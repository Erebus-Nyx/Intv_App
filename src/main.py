import argparse
from utils import is_valid_filetype, chunk_document
from rag import process_with_rag
from llm import analyze_chunks
import yaml
import sys
import threading
import time
from pathlib import Path
from contextlib import nullcontext
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
except ImportError:
    Console = None
    Live = None
    Table = None
import subprocess
import glob
import json
import os

def transcribe_audio_file(audio_path, model_size="small", diarization=False, vad=False, output_format="txt"):
    from pathlib import Path
    """
    Transcribe an audio file using WhisperModel.
    Supports diarization, VAD, and output format selection.
    Returns transcript and metadata.
    """
    model = WhisperModel(model_size)
    segments, info = model.transcribe(audio_path, vad_filter=vad, word_timestamps=True)
    transcript = ""
    metadata = []
    for seg in segments:
        speaker = getattr(seg, 'speaker', None)
        start = getattr(seg, 'start', None)
        end = getattr(seg, 'end', None)
        text = seg.text
        transcript += text
        metadata.append({
            'start': start,
            'end': end,
            'speaker': speaker,
            'text': text
        })
    # Output format selection
    out_path = Path(audio_path).with_suffix(f'.{output_format}')
    if output_format == "json":
        import json
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'transcript': transcript, 'segments': metadata, 'info': info}, f, indent=2, ensure_ascii=False)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
    return transcript, metadata

def stream_microphone_transcription(output_path, model_size="small", diarization=False, vad=False, output_format="txt"):
    from pathlib import Path
    """
    Stream microphone input, save to file, and transcribe using WhisperModel.
    Supports diarization, VAD, and output format selection.
    Returns transcript and metadata.
    """
    if sd is None or sf is None or WhisperModel is None:
        print("Required packages for microphone transcription are not installed.")
        sys.exit(1)
    model = WhisperModel(model_size)
    duration = 0
    running = True
    paused = False
    transcript = ""
    metadata = []
    console = Console() if Console else None
    def record_callback(indata, frames, time_info, status):
        nonlocal transcript, duration
        if not paused:
            sf.write(output_path, indata, 16000, format='WAV', append=True)
            # For demo: just update duration
            duration += frames / 16000
    def show_ui():
        with Live(refresh_per_second=2) if Live else nullcontext():
            while running:
                if console:
                    table = Table()
                    table.add_column("Status")
                    table.add_column("Duration (s)")
                    table.add_row("Recording" if not paused else "Paused", f"{duration:.1f}")
                    table.add_row("Transcript", transcript[-80:])
                    console.clear()
                    console.print(table)
                time.sleep(0.5)
    # Start UI thread
    ui_thread = threading.Thread(target=show_ui, daemon=True)
    ui_thread.start()
    # Start recording
    with sd.InputStream(samplerate=16000, channels=1, callback=record_callback):
        print("Press 'p' to pause/resume, 's' to stop.")
        while running:
            c = sys.stdin.read(1)
            if c == 'p':
                paused = not paused
            elif c == 's':
                running = False
    # After recording, transcribe
    segments, info = model.transcribe(str(output_path), vad_filter=vad, word_timestamps=True)
    transcript = ""
    metadata = []
    for seg in segments:
        speaker = getattr(seg, 'speaker', None)
        start = getattr(seg, 'start', None)
        end = getattr(seg, 'end', None)
        text = seg.text
        transcript += text
        metadata.append({
            'start': start,
            'end': end,
            'speaker': speaker,
            'text': text
        })
    out_path = Path(str(output_path) + f'.{output_format}')
    if output_format == "json":
        import json
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'transcript': transcript, 'segments': metadata, 'info': info}, f, indent=2, ensure_ascii=False)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
    return transcript, metadata

def start_cloudflared_tunnel(port=3773):
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    public_url = None
    for line in process.stdout:
        if "trycloudflare.com" in line:
            print(line, end="")
            if not public_url:
                import re
                m = re.search(r'(https://[\w\-]+\.trycloudflare.com)', line)
                if m:
                    public_url = m.group(1)
                    print(f"[Cloudflared] Public URL: {public_url}")
        elif "failed to sufficiently increase receive buffer size" in line:
            continue  # Suppress this warning
        else:
            print(line, end="")
    return public_url

def detect_filetype_from_extension(file_path: Path):
    from pathlib import Path
    ext = file_path.suffix.lower().lstrip('.')
    supported_types = ['txt', 'rtf', 'docx', 'pdf', 'mp4', 'm4a', 'jpg']
    if ext in supported_types:
        return ext
    raise ValueError(f"Unsupported file extension: {ext}. Supported types: {supported_types}")

def get_available_interview_types():
    config_dir = os.path.join(os.path.dirname(__file__), 'modules')
    pattern = os.path.join(config_dir, '*_vars.json')
    files = glob.glob(pattern)
    types = []
    for f in files:
        base = os.path.basename(f)
        type_key = base.replace('_vars.json', '')
        # Try to get a display name from the first key or use the type_key
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                # Use the first key's hint as display name if available
                if isinstance(data, dict) and data:
                    first_key = next(iter(data))
                    hint = data[first_key].get('hint') if isinstance(data[first_key], dict) else None
                    display = hint if hint else type_key.replace('_', ' ').title()
                else:
                    display = type_key.replace('_', ' ').title()
        except Exception:
            display = type_key.replace('_', ' ').title()
        types.append({'key': type_key, 'display': display})
    return types

def main():
    from pathlib import Path
    import psutil
    def is_process_running(name, check_status=False):
        for proc in psutil.process_iter(['name', 'cmdline'] + (['status'] if check_status else [])):
            try:
                if name in proc.info['name'] or (proc.info['cmdline'] and any(name in c for c in proc.info['cmdline'])):
                    if check_status:
                        if proc.info.get('status', '').lower() in ('zombie', 'defunct'):
                            continue
                    return True
            except Exception:
                continue
        return False
    """
    Main entrypoint for document/audio analysis and RAG/LLM workflow.
    Handles argument parsing, input validation, and pipeline orchestration.
    """
    parser = argparse.ArgumentParser(description='Document Analysis with RAG and LLM', add_help=True)
    parser.add_argument('--file', required=False, help='Path to the document')
    parser.add_argument('--format', required=False, choices=['pdf', 'docx', 'txt', 'rtf', 'mp4', 'm4a', 'jpg'], help='Document file format (optional, will be auto-detected from file extension if not provided)')
    parser.add_argument('--model', required=False, default='hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q6_K_XL', help='Model name or ID (default: hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q6_K_XL)')
    parser.add_argument('--rag-mode', choices=['embedded', 'external'], default='embedded', help='RAG mode')
    parser.add_argument('--llm-provider', default='koboldcpp', help='LLM provider: openai, ollama, koboldcpp, etc. (default: koboldcpp)')
    parser.add_argument('--llm-api-base', default=None, help='Base URL for LLM API (e.g., https://api.openai.com or http://localhost)')
    parser.add_argument('--llm-api-key', default=None, help='API key for LLM provider (if required)')
    parser.add_argument('--llm-api-port', default=None, type=int, help='Port for LLM API (if required, default: 5001 for koboldcpp)')
    parser.add_argument('--nowada', action='store_false', dest='disable_cuda', help='Disable CUDA (GPU) acceleration for supported LLMs (AR mode, disables CUDA)')
    parser.add_argument('--config', default=None, help='Path to config.json (overrides defaults)')
    parser.add_argument('--gui', action='store_true', help='Enable web-based GUI (if not set, defaults to terminal mode)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only mode (disables CUDA and CUTLASS, for AMD/ARM compatibility)')
    parser.add_argument('--audio', type=str, help='Path to audio file for transcription')
    parser.add_argument('--mic', action='store_true', help='Use microphone for streaming transcription')
    parser.add_argument('--diarization', action='store_true', help='Enable speaker diarization')
    parser.add_argument('--vad', action='store_true', help='Enable voice activity detection (VAD)')
    parser.add_argument('--output-format', choices=['txt', 'json'], default='txt', help='Transcription output format')
    parser.add_argument('--defaults', action='store_true', help='Bypass data source and use default values for all variables')
    parser.add_argument('--remotetunnel', action='store_true', help='Start a Cloudflare tunnel for remote access')
    parser.add_argument('--output', required=False, default=None, help='Optional output file path for module results (default: print to stdout)')
    parser.add_argument('--shutdown', action='store_true', help='Shutdown all FastAPI and Cloudflared services (Linux: uses scripts/run_and_info.sh --exit)')
    parser.add_argument('--tunnelinfo', action='store_true', help='Print the current Cloudflared public tunnel link (if available) and exit')
    parser.set_defaults(disable_cuda=False)  # Default: CUDA ON
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)
    # --- DYNAMIC TYPE ARGUMENT ---
    available_types = get_available_interview_types()
    type_keys = [t['key'] for t in available_types]
    parser.add_argument('--type', required=False, choices=type_keys, help=f"Interview type/module: {', '.join(type_keys)}")
    args = parser.parse_args()

    # --- Ensure FastAPI is running for all commands except shutdown/tunnelinfo/remotetunnel ---
    skip_fastapi = any([
        getattr(args, 'shutdown', False),
        getattr(args, 'tunnelinfo', False),
        getattr(args, 'remotetunnel', False)
    ])
    if not skip_fastapi:
        import shutil
        fastapi_running = is_process_running('uvicorn')
        if not fastapi_running:
            print('[INFO] FastAPI service not running. Starting FastAPI...')
            fastapi_cmd = shutil.which('uvicorn') or 'uvicorn'
            fastapi_proc = subprocess.Popen([
                fastapi_cmd,
                'src.modules.gui.app:app',
                '--host', '0.0.0.0',
                '--port', '3773',
                '--workers', '4'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Wait for FastAPI to be ready
            import socket
            import time
            for _ in range(30):
                try:
                    with socket.create_connection(("127.0.0.1", 3773), timeout=1):
                        print('[INFO] FastAPI is now running.')
                        break
                except Exception:
                    time.sleep(1)
            else:
                print('[ERROR] FastAPI did not start within 30 seconds.')
                sys.exit(1)
        cloudflared_running = is_process_running('cloudflared')
        if not cloudflared_running:
            print('[INFO] Cloudflared not running. Starting cloudflared...')
            cloudflared_bin = shutil.which('cloudflared') or './scripts/cloudflared-linux-amd64'
            subprocess.Popen([
                cloudflared_bin,
                'tunnel', '--url', 'http://localhost:3773'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
    # Handle --tunnelinfo argument
    if getattr(args, 'tunnelinfo', False):
        import glob
        import os
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        url_files = glob.glob(os.path.join(cache_dir, 'cloudflared_url_*.txt'))
        if not url_files:
            print("[INFO] No Cloudflared tunnel link found in .cache.")
            sys.exit(0)
        # Print all found links
        for url_file in url_files:
            with open(url_file, 'r', encoding='utf-8') as f:
                link = f.read().strip()
                print(f"[Cloudflared Tunnel] {os.path.basename(url_file)}: {link}")
        sys.exit(0)
    # Handle --remotetunnel argument
    if getattr(args, 'remotetunnel', False):
        import subprocess
        import re
        from datetime import datetime
        import os
        import shutil
        import psutil
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        os.makedirs(cache_dir, exist_ok=True)
        port = 3773  # Default FastAPI port
        # Check if cloudflared is already running (hardened: skip zombies/defunct)
        def is_process_running(name):
            for proc in psutil.process_iter(['name', 'cmdline', 'status']):
                try:
                    if (
                        (name in proc.info['name'] or (proc.info['cmdline'] and any(name in c for c in proc.info['cmdline'])))
                        and proc.info.get('status', '').lower() not in ('zombie', 'defunct')
                    ):
                        return True
                except Exception:
                    continue
            return False
        # Check for cached URLs
        url_files = []
        try:
            url_files = [f for f in os.listdir(cache_dir) if f.startswith('cloudflared_url_') and f.endswith('.txt')]
        except Exception:
            pass
        if is_process_running('cloudflared'):
            if url_files:
                print("[INFO] Cloudflared tunnel already running. Cached public URL(s):")
                for url_file in url_files:
                    url_path = os.path.join(cache_dir, url_file)
                    with open(url_path, 'r', encoding='utf-8') as f:
                        link = f.read().strip()
                        print(f"[Cloudflared Tunnel] {url_file}: {link}")
                sys.exit(0)
            else:
                # If no cached URL and process is running, warn and allow new tunnel
                print("[WARN] Cloudflared process detected but no cached URL found. Starting a new tunnel anyway...")
        elif url_files:
            # If no process but cache exists, warn and allow new tunnel
            print("[WARN] Cached Cloudflared URL(s) found but no process running. Removing stale cache and starting new tunnel...")
            for url_file in url_files:
                try:
                    os.remove(os.path.join(cache_dir, url_file))
                except Exception:
                    pass
        # Find cloudflared binary robustly
        cloudflared_bin = shutil.which('cloudflared')
        if not cloudflared_bin:
            alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/cloudflared-linux-amd64'))
            if os.path.isfile(alt_path) and os.access(alt_path, os.X_OK):
                cloudflared_bin = alt_path
            else:
                print("[ERROR] cloudflared not found in PATH or scripts/. Please install cloudflared or place the binary at scripts/cloudflared-linux-amd64 and make it executable.")
                sys.exit(1)
        # Kill any zombie/defunct cloudflared processes before starting a new tunnel
        for proc in psutil.process_iter(['name', 'cmdline', 'status']):
            try:
                if (
                    'cloudflared' in proc.info['name'] or (proc.info['cmdline'] and any('cloudflared' in c for c in proc.info['cmdline']))
                ) and proc.info.get('status', '').lower() in ('zombie', 'defunct'):
                    proc.kill()
            except Exception:
                pass
        print(f"[INFO] Starting Cloudflared tunnel for http://localhost:{port} ...")
        proc = subprocess.Popen([
            cloudflared_bin, 'tunnel', '--url', f'http://localhost:{port}'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        public_url = None
        for line in proc.stdout:
            if "trycloudflare.com" in line:
                print(line, end="")
                m = re.search(r'(https://[\w\-]+\.trycloudflare.com)', line)
                if m:
                    public_url = m.group(1)
                    print(f"[Cloudflared] Public URL: {public_url}")
                    # Verify the public URL is reachable before caching (retry up to 5 times)
                    import requests
                    import time as _time
                    for attempt in range(5):
                        try:
                            resp = requests.get(public_url, timeout=10)
                            if resp.status_code < 400:
                                break  # Success
                            else:
                                print(f"[WARN] Cloudflared public URL returned HTTP {resp.status_code} (attempt {attempt+1}/5). Retrying...")
                        except Exception as e:
                            print(f"[WARN] Could not reach Cloudflared public URL (attempt {attempt+1}/5): {e}. Retrying...")
                        _time.sleep(2)
                    else:
                        print(f"[ERROR] Could not reach Cloudflared public URL after 5 attempts. Not caching this URL.")
                        public_url = None
                        break
                    # Cache the URL if reachable
                    dtstr = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                    url_path = os.path.join(cache_dir, f'cloudflared_url_{dtstr}.txt')
                    with open(url_path, 'w', encoding='utf-8') as f:
                        f.write(public_url)
                    print(f"[INFO] Cloudflared public URL cached at {url_path}")
                    break
            elif "failed to sufficiently increase receive buffer size" in line:
                continue
            else:
                print(line, end="")
        proc.terminate()
        if not public_url:
            print("[ERROR] No valid Cloudflared public URL was established. Please check your tunnel and try again.")
            sys.exit(1)
        sys.exit(0)
    # Handle shutdown immediately after parsing args
    if getattr(args, 'shutdown', False):
        import subprocess
        import platform
        import glob
        import os
        if platform.system().lower().startswith('win'):
            script = 'scripts/run_and_info_win.bat'
            cmd = [script, '--exit']
        else:
            script = 'scripts/run_and_info.sh'
            cmd = ['bash', script, '--exit']
        print(f"[INFO] Shutting down FastAPI and Cloudflared using {script} ...")
        subprocess.run(cmd)
        # Remove cached tunnel URLs after shutdown
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        url_files = glob.glob(os.path.join(cache_dir, 'cloudflared_url_*.txt'))
        for url_file in url_files:
            try:
                os.remove(url_file)
            except Exception:
                pass
        print("[INFO] Removed cached Cloudflared tunnel URLs from .cache.")
        sys.exit(0)
    sources = [s for s in [args.audio, args.mic] if s]
    if len(sources) > 1:
        print("[ERROR] Multiple audio sources provided. Please specify only one of --audio or --mic.")
        print("Usage: --audio <file> OR --mic (not both)")
        sys.exit(1)
    # File format detection logic
    if args.file:
        file_path = Path(args.file)
        # Auto-detect format if not provided
        if not args.format:
            try:
                args.format = detect_filetype_from_extension(file_path)
                print(f"[INFO] Auto-detected file format: {args.format}")
            except ValueError as e:
                print(f"[ERROR] {e}")
                sys.exit(1)
    # --- Handle --mic: record, transcribe, and cache before LLM ---
    if getattr(args, 'mic', False):
        from datetime import datetime
        transcript, metadata = stream_microphone_transcription('mic_recording.wav')
        # Save to .cache/yyyy-mm-dd_hhmm_recording.json
        cache_dir = Path('.cache')
        cache_dir.mkdir(exist_ok=True)
        dtstr = datetime.now().strftime('%Y-%m-%d_%H%M')
        cache_path = cache_dir / f"{dtstr}_recording.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'transcript': transcript, 'segments': metadata}, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Recording and transcript saved to {cache_path}")
        # Optionally, set args.file to the cache_path for downstream processing
        args.file = str(cache_path)
        args.format = 'json'
        # If you want to skip LLM evaluation for mic, you could return here
        # return

    # Get dynamic interview/module types
    available_types = get_available_interview_types()
    type_keys = [t['key'] for t in available_types]
    # Menu for interview type selection if not provided
    if not args.type or args.type not in type_keys:
        print("\nSelect an interview/module type:")
        for idx, t in enumerate(available_types, 1):
            print(f"{idx}. {t['display']} ({t['key']})")
        print(f"{len(available_types)+1}. Cancel")
        while True:
            choice = input(f"Enter choice [1-{len(available_types)+1}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available_types)+1:
                break
            print(f"Invalid choice. Please enter a number 1-{len(available_types)+1}.")
        if int(choice) == len(available_types)+1:
            print("Exiting.")
            return
        args.type = available_types[int(choice)-1]['key']

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists() or not is_valid_filetype(file_path, args.format):
            raise ValueError('Invalid file or file format')
        try:
            chunks = chunk_document(file_path, args.format)
        except NotImplementedError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    def show_rag_progress():
        count = 0
        while not hasattr(show_rag_progress, 'done'):
            print(f"[RAG] Processing... (chunks processed: {count})", end='\r')
            time.sleep(0.5)
            count += 1
        print(f"[RAG] Processing complete. Total chunks: {count}")

    if args.gui:
        print("[INFO] GUI mode enabled. (Web interface setup placeholder)")
        # Placeholder: Launch web server here in the future
    else:
        print("[INFO] Terminal mode (default)")

    # Start RAG progress indicator in a thread
    progress_thread = threading.Thread(target=show_rag_progress)
    progress_thread.start()

    if args.rag_mode == 'external':
        rag_results = process_with_rag(
            chunks,
            mode=args.rag_mode,
            filetype=args.format,
            file_path=str(file_path)
        )
    else:
        rag_results = process_with_rag(chunks, mode=args.rag_mode)

    # Signal progress thread to stop
    show_rag_progress.done = True
    progress_thread.join()

    # Load config and optionally override with command-line args
    from config import load_config, save_config
    config = load_config()
    # Load from YAML if present
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            config.update({k: v for k, v in yaml_config.items() if v is not None})
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open('r', encoding='utf-8') as f:
                user_config = json.load(f)
            config.update(user_config)
    # Override config with command-line args if provided
    for k in ['llm_api_base', 'llm_api_key', 'llm_api_port', 'llm_provider', 'model', 'external_rag', 'purge_variables', 'name']:
        v = getattr(args, k, None)
        if v is not None:
            config[k] = v
    # Ensure correct default port for provider if not set or invalid
    if config.get('llm_provider', '').lower() == 'koboldcpp' and not (isinstance(config.get('llm_api_port'), int) and config['llm_api_port'] > 0):
        config['llm_api_port'] = 5001
    elif config.get('llm_provider', '').lower() == 'ollama' and not (isinstance(config.get('llm_api_port'), int) and config['llm_api_port'] > 0):
        config['llm_api_port'] = 11434
    # Force correct port if provider/port combo is mismatched
    if config.get('llm_provider', '').lower() == 'koboldcpp' and config.get('llm_api_port') == 11434:
        config['llm_api_port'] = 5001
    elif config.get('llm_provider', '').lower() == 'ollama' and config.get('llm_api_port') == 5001:
        config['llm_api_port'] = 11434
    # Hardware acceleration detection
    import platform
    import os
    use_gpu = not args.cpu
    extra_params = {}
    # Try to detect if CUDA or CUTLASS is available
    cuda_available = False
    cutlass_available = False
    if use_gpu:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass
        # Check for CUTLASS (if supported by backend)
        # This is a placeholder; actual detection may depend on backend
        cutlass_available = os.environ.get('CUTLASS_PATH') is not None
        if cuda_available:
            extra_params['cuda'] = True
        if cutlass_available:
            extra_params['cutlass'] = True

    # --- Koboldcpp model info check (if using koboldcpp) ---
    backend_info = {}
    if config.get('llm_provider', '').lower() == 'koboldcpp':
        import requests
        koboldcpp_url = f"http://localhost:{config.get('llm_api_port', 5001)}"
        model_name = None
        for endpoint in ["/v1/model", "/api/v1/model", "/v1/info", "/api/v1/info"]:
            try:
                resp = requests.get(koboldcpp_url + endpoint, timeout=3)
                if resp.ok:
                    data = resp.json()
                    # Try common keys
                    if isinstance(data, dict):
                        model_name = data.get('model') or data.get('model_name') or data.get('model_path')
                        # If still not found, try 'result' key
                        if not model_name and 'result' in data:
                            model_name = data['result']
                    else:
                        model_name = str(data)
                    backend_info['koboldcpp_model'] = model_name
                    backend_info['koboldcpp_endpoint'] = koboldcpp_url + endpoint
                    break
            except Exception as e:
                continue
        if not model_name:
            backend_info['koboldcpp_model'] = 'Unknown (could not query koboldcpp API)'
        print(f"[INFO] koboldcpp backend model: {backend_info.get('koboldcpp_model')}")
    else:
        backend_info['koboldcpp_model'] = None

    # Debug: print resolved config for LLM
    print(f"[DEBUG] LLM provider: {config.get('llm_provider')}, LLM port: {config.get('llm_api_port')}, LLM base: {config.get('llm_api_base')}")
    llm_output = analyze_chunks(
        rag_results,
        model=config['model'],
        api_base=config['llm_api_base'],
        api_key=config['llm_api_key'],
        api_port=config['llm_api_port'],
        provider=config['llm_provider'],
        extra_params=extra_params if extra_params else None
    )

    # Always treat each run as a unique instance: clear any cached LLM variable values for this file/interview
    from llm_db import clear_llm_variables
    if args.file:
        clear_llm_variables(str(args.file))

    # After menu selection, call the corresponding module
    if not getattr(args, 'gui', False):
        from modules.dynamic_module import dynamic_module_output
        module_key = args.type
        provided_data = None
        if hasattr(args, f'{module_key}_data'):
            provided_data = getattr(args, f'{module_key}_data')
        result = dynamic_module_output(lookup_id=args.file, output_path=args.output, module_key=module_key, provided_data=provided_data)
        # Attach backend_info to output for user visibility
        if isinstance(result, dict):
            result['backend_info'] = backend_info
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

if __name__ == '__main__':
    main()
