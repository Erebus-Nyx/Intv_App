import sys
import argparse
import json
import os
import time
from pathlib import Path
from . import __version__

def get_parser():
    parser = argparse.ArgumentParser(
        description="intv: Interview Automation & Document Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # General options
    general = parser.add_argument_group('General Options')
    general.add_argument('-f', '--file', required=False, help='Path to the document file')
    general.add_argument('--audio', required=False, help='Path to the audio file for transcription')
    general.add_argument('-o', '--output', help='Output file path (optional)')
    general.add_argument('--config', default=None, help='Path to config.yaml (overrides default config location)')
    general.add_argument('--gui', action='store_true', default=False, help='Enable web-based GUI (if not set, defaults to terminal mode)')
    general.add_argument('--remotetunnel', action='store_true', default=False, help='Start a Cloudflare tunnel for remote access')
    general.add_argument('--shutdown', action='store_true', default=False, help='Shutdown all running intv services and exit')
    general.add_argument('--version', action='store_true', default=False, help='Show version and exit')
    general.add_argument('--about', action='store_true', default=False, help='Show about information and exit')
    general.add_argument('--mic', action='store_true', default=False, help='Use microphone for live/streaming transcription')
    general.add_argument('--module', required=False, help='Module/interview type to use (e.g., adult, ar, child, etc.). If set, bypasses interactive selection.')
    # All other file locations (vars, policy, etc.) are set in config.yaml only
    return parser

# Helper to load config from settings.json
def load_settings(config_path=None):
    settings_path = config_path or str(Path(__file__).parent.parent / 'settings.json')
    try:
        with open(settings_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def parse_cli_args():
    parser = get_parser()
    # Fast path: print help/version/about before any config loading or import
    if len(sys.argv) == 1 or any(arg in sys.argv for arg in ["-h", "--help"]):
        parser.print_help()
        sys.exit(0)
    if "--version" in sys.argv:
        print(f"intv version {__version__}")
        sys.exit(0)
    if "--about" in sys.argv:
        print("intv: Interview Automation & Document Analysis CLI\n"
              "Modular, production-ready CLI for document and audio analysis with RAG and LLM support.")
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Handle shutdown flag early
    if args.shutdown:
        from .server_utils import shutdown_services
        print("[INFO] Shutting down all intv services...")
        shutdown_services()
        print("[INFO] Shutdown complete.")
        sys.exit(0)
    # Load config values from settings.json (or config.yaml via settings.json)
    settings = load_settings(getattr(args, 'config', None))
    # Set config-driven options (no CLI override for file locations)
    args.log_level = settings.get('log_level', 'INFO')
    args.llm_provider = settings.get('llm_provider', 'llama.cpp')
    args.llm_api_base = settings.get('llm_api_base', None)
    args.llm_api_key = settings.get('llm_api_key', None)
    args.llm_api_port = settings.get('llm_api_port', None)
    args.chunk_size = settings.get('chunk_size', 1200)
    args.output_format = settings.get('output_format', 'txt')  # Accepts: txt, json, srt, vtt. Used for both transcription and results.
    args.rag_mode = settings.get('rag_mode', 'embedded')
    args.llm_mode = settings.get('llm_mode', 'embedded')
    args.llm_provider = settings.get('llm_provider', '')

    # Inject config-driven defaults for all new settings
    args.whisper_model = settings.get('whisper_model', 'base')
    args.enable_vad = settings.get('enable_vad', True)
    args.enable_diarization = settings.get('enable_diarization', True)
    args.audio_sample_rate = settings.get('audio_sample_rate', 16000)
    args.vad_min_segment_ms = settings.get('vad_min_segment_ms', 500)
    args.vad_aggressiveness = settings.get('vad_aggressiveness', 2)
    args.diarization_model = settings.get('diarization_model', '')
    args.llm_mode = settings.get('llm_mode', 'embedded')
    args.llm_model = settings.get('llm_model', '')
    args.llm_temperature = settings.get('llm_temperature', 0.7)
    args.llm_context_size = settings.get('llm_context_size', 'auto')
    args.llm_max_tokens = settings.get('llm_max_tokens', 1024)
    args.llm_top_p = settings.get('llm_top_p', 1.0)
    args.llm_frequency_penalty = settings.get('llm_frequency_penalty', 0.0)
    args.llm_presence_penalty = settings.get('llm_presence_penalty', 0.0)
    args.llm_stop = settings.get('llm_stop', '')
    args.rag_model = settings.get('rag_model', '')
    args.rag_chunk_size = settings.get('rag_chunk_size', 500)
    args.rag_top_k = settings.get('rag_top_k', 5)
    args.rag_score_threshold = settings.get('rag_score_threshold', 0.0)
    args.rag_prompt_template = settings.get('rag_prompt_template', '')
    args.rag_context_window = settings.get('rag_context_window', 2048)

    # Failsafe: If llm_mode is 'external', check for required external LLM config
    if args.llm_mode == 'external':
        missing = []
        if not settings.get('llm_provider'):
            missing.append('llm_provider')
        if not settings.get('llm_api_base'):
            missing.append('llm_api_base')
        if settings.get('llm_api_key', None) is None:
            missing.append('llm_api_key')
        if settings.get('llm_api_port', None) is None:
            missing.append('llm_api_port')
        if missing:
            print(f"[ERROR] llm_mode is 'external' but the following required config values are missing or commented out in config.yaml: {', '.join(missing)}")
            print("Please edit config/config.yaml and ensure all required fields are set for external LLM usage.")
            sys.exit(1)

    # If using koboldcpp as an external LLM provider, ignore any model specified in config or CLI
    if args.llm_mode == 'external' and args.llm_provider.lower() == 'koboldcpp' and hasattr(args, 'llm_model'):
        args.llm_model = None  # KoboldCpp will use whatever model is loaded in its app

    # Skip interactive prompt for tunnel-only operations
    tunnel_only_operation = getattr(args, 'remotetunnel', False) and not getattr(args, 'file', None) and not getattr(args, 'audio', None)
    
    # Only run interactive prompt if not help/version/about/tunnel-only and if type is missing
    if not tunnel_only_operation and (not hasattr(args, 'type') or not args.type) and (not hasattr(args, 'module') or not args.module):
        try:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            try:
                from .module_utils import get_available_interview_types
            except ImportError:
                from module_utils import get_available_interview_types
            available_types = get_available_interview_types()
            type_keys = [t['key'] for t in available_types]
            menu_types = [t.get('display', t['key']) for t in available_types]
            print("\nSelect an interview/module type:")
            for idx, label in enumerate(menu_types, 1):
                print(f"{idx}. {label}")
            print(f"{len(menu_types)+1}. Cancel")
            while True:
                choice = input(f"Enter choice [1-{len(menu_types)+1}]: ")
                if choice.isdigit() and 1 <= int(choice) <= len(menu_types)+1:
                    choice = int(choice)
                    if choice == len(menu_types)+1:
                        print("Cancelled.")
                        sys.exit(0)
                    selected_type = type_keys[choice-1]
                    setattr(args, 'type', selected_type)
                    break
                print(f"Invalid choice, please select a number between 1 and {len(menu_types)+1}.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    # If --module is provided, set args.type to args.module for downstream logic
    if hasattr(args, 'module') and args.module:
        args.type = args.module
    return args

def main():
    args = parse_cli_args()
    try:
        from .output_and_cache import ensure_output_dir, purge_cache
    except ImportError:
        # Fallback for when running as main module
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from output_and_cache import ensure_output_dir, purge_cache
    config_path = getattr(args, 'config', None)
    import json
    from pathlib import Path
    settings_path = config_path or str(Path(__file__).parent.parent / 'settings.json')
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except Exception:
        settings = {}
    output_dir = settings.get('output_dir', 'output/')
    cache_dir = settings.get('cache_dir', '.cache/')
    retention_days = int(settings.get('cache_retention_days', 7))
    ensure_output_dir(output_dir)

    # Handle remote tunnel flag
    if args.remotetunnel:
        from .server_utils import ensure_cloudflared_running, is_port_in_use, get_cloudflared_binary
        import subprocess
        import os
        
        # Check if cloudflared is already running
        def is_cloudflared_running():
            try:
                import psutil
                for proc in psutil.process_iter(['name', 'cmdline', 'status']):
                    try:
                        if (
                            'cloudflared' in proc.info['name'] or
                            (proc.info['cmdline'] and any('cloudflared' in c for c in proc.info['cmdline']))
                        ) and proc.info.get('status', '').lower() not in ('zombie', 'defunct'):
                            return True, proc.pid
                    except Exception:
                        continue
                return False, None
            except ImportError:
                # Fallback: use pgrep if psutil is not available
                result = subprocess.run(['pgrep', '-f', 'cloudflared'], capture_output=True, text=True)
                if result.returncode == 0:
                    pid = result.stdout.strip().split('\n')[0]
                    return True, int(pid) if pid.isdigit() else None
                return False, None
        
        tunnel_running, tunnel_pid = is_cloudflared_running()
        
        if tunnel_running:
            # Tunnel already running - show status and exit
            print(f"[INFO] Cloudflare tunnel is already running (PID: {tunnel_pid})")
            
            # Try to get tunnel URL from logs
            log_path = "/tmp/cloudflared_3773.log"
            tunnel_url = None
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        for line in f:
                            if "trycloudflare.com" in line and "http" in line:
                                parts = line.split()
                                for part in parts:
                                    if "trycloudflare.com" in part:
                                        tunnel_url = part.strip()
                                        break
                                if tunnel_url:
                                    break
                except Exception:
                    pass
            
            if tunnel_url:
                print(f"[INFO] Tunnel URL: {tunnel_url}")
            else:
                print("[INFO] Tunnel URL not found in logs. Check /tmp/cloudflared_3773.log")
            
            print("[INFO] To stop the tunnel, use: python -m intv.cli --shutdown")
            sys.exit(0)
        else:
            # Start tunnel in background
            print("[INFO] Starting Cloudflare tunnel in background...")
            
            # Ensure FastAPI is running first if port is not active
            if not is_port_in_use(3773):
                print("[INFO] Port 3773 not active. You may need to start your FastAPI server first.")
                print("[INFO] Starting a simple HTTP server on port 3773 for tunnel testing...")
                # Start a simple server in background for testing
                subprocess.Popen([
                    'python3', '-c', 
                    'import http.server; import socketserver; '
                    'handler = http.server.SimpleHTTPRequestHandler; '
                    'httpd = socketserver.TCPServer(("", 3773), handler); '
                    'print("Test server running on port 3773"); '
                    'httpd.serve_forever()'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                import time
                time.sleep(2)  # Give server time to start
            
            # Start cloudflared in background
            try:
                from .server_utils import get_cloudflared_binary
                cloudflared_bin = get_cloudflared_binary()
                log_path = "/tmp/cloudflared_3773.log"
                
                # Start cloudflared in background
                proc = subprocess.Popen([
                    cloudflared_bin, 'tunnel', '--url', 'http://localhost:3773'
                ], stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
                
                print(f"[INFO] Cloudflared started in background (PID: {proc.pid})")
                print("[INFO] Waiting for tunnel URL...")
                
                # Wait for tunnel URL (but don't block too long)
                for i in range(15):  # Wait up to 15 seconds
                    if os.path.exists(log_path):
                        try:
                            with open(log_path, 'r') as f:
                                for line in f:
                                    if "trycloudflare.com" in line and "http" in line:
                                        parts = line.split()
                                        for part in parts:
                                            if "trycloudflare.com" in part:
                                                print(f"[INFO] Tunnel URL: {part.strip()}")
                                                print("[INFO] Tunnel is running in background.")
                                                print("[INFO] To stop: python -m intv.cli --shutdown")
                                                sys.exit(0)
                        except Exception:
                            pass
                    print(".", end="", flush=True)
                    time.sleep(1)
                
                print(f"\n[INFO] Tunnel started but URL not yet available. Check logs: {log_path}")
                print("[INFO] To stop: python -m intv.cli --shutdown")
                
            except Exception as e:
                print(f"[ERROR] Failed to start tunnel: {e}")
                sys.exit(1)
        
        # Exit after handling tunnel operation
        sys.exit(0)

    # --- Actual pipeline integration ---
    ai_summary = "[AI summary goes here]"
    formatted_interview = "[Formatted interview output goes here]"
    transcript_notes = "[Transcript/notes go here]"
    if getattr(args, 'file', None):
        # Determine module type (from args or prompt)
        module_key = getattr(args, 'module', None)
        if not module_key:
            # For automated mode, if no module is specified, default to 'adult'
            module_key = 'adult'
            print("[INFO] No module specified, defaulting to 'adult'")
        # Call the pipeline
        try:
            from .llm import rag_llm_pipeline
        except ImportError:
            from llm import rag_llm_pipeline
        pipeline_output = rag_llm_pipeline(
            document_path=args.file,
            module_key=module_key,
            model=args.llm_model,
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            api_port=args.llm_api_port,
            provider=args.llm_provider,
            output_path=None,  # Will trigger save-as dialog if not set
            config=settings
        )
        # Parse output blocks
        block1 = pipeline_output.get('block1', {})
        block2 = pipeline_output.get('block2', {})
        block3 = pipeline_output.get('block3', '')
        ai_summary = block1.get('llm_summary', '')
        formatted_interview = block3
        # For transcript/notes, show all variables and extra info
        transcript_notes = json.dumps(block2, indent=2, ensure_ascii=False)
        if block1.get('extra_info'):
            transcript_notes += f"\n\nExtra Info:\n{block1['extra_info']}"

    # For automated mode (when --module is specified), only print the summary
    if hasattr(args, 'module') and args.module:
        print(ai_summary)
        return

    # Compose output in the requested format for interactive mode
    output_content = (
        "------ AI Summary ------\n"
        f"{ai_summary}\n\n"
        "------ Formatted Interview ------\n"
        f"{formatted_interview}\n\n"
        "------ Transcript / Notes ------\n"
        f"{transcript_notes}\n"
    )

    # Save output to output_dir as usual
    output_path = os.path.join(output_dir, 'result.txt')
    with open(output_path, 'w') as f:
        f.write(output_content)

    # Offer a save-as dialog for the user (if running in an environment with a display)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        save_path = filedialog.asksaveasfilename(
            title="Save As",
            initialdir=output_dir,
            initialfile='result.txt',
            defaultextension='.txt',
            filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')]
        )
        if save_path:
            with open(save_path, 'w') as f:
                f.write(output_content)
            print(f"Output also saved as: {save_path}")
    except Exception as e:
        print(f"[INFO] Save-as dialog not available: {e}")

    # At the end of main(), before exiting:
    purge_cache(cache_dir, retention_days)

if __name__ == "__main__":
    main()
