# Main entry point for the INTV CLI and pipeline
import argparse
from intv.utils import is_valid_filetype
from intv.rag import chunk_text, chunk_document, batch_chunk_documents, process_with_retriever_and_llm
from intv.llm import analyze_chunks
import yaml
import sys
import os
import json
import threading
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from intv.cli import parse_cli_args as parse_args
from intv.server_utils import ensure_fastapi_running, ensure_cloudflared_running
from intv.audio_transcribe import transcribe_audio_fastwhisper
from intv.module_utils import get_available_interview_types, detect_filetype_from_extension


def main():
    args = parse_args()
    # Parse CLI arguments and handle file dialog if requested

    # --- Handle --file-dialog (must be before file logic) ---
    if getattr(args, 'file_dialog', False):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select a file to analyze",
                filetypes=[
                    ("Supported files", "*.pdf *.docx *.txt *.rtf *.mp4 *.m4a *.jpg"),
                    ("All files", "*.*")
                ]
            )
            if not file_path:
                print("[INFO] No file selected. Exiting.")
                sys.exit(0)
            args.file = file_path
            print(f"[INFO] Selected file: {args.file}")
        except Exception as e:
            print(f"[ERROR] Could not open file dialog: {e}")
            sys.exit(1)
    # --- Ensure FastAPI is running for all commands except shutdown/tunnelinfo/remotetunnel ---
    skip_fastapi = any([
        getattr(args, 'shutdown', False),
        getattr(args, 'tunnelinfo', False),
        getattr(args, 'remotetunnel', False)
    ])
    if not skip_fastapi:
        ensure_fastapi_running()
        ensure_cloudflared_running()

    # --- Handle tunnel info: print all cached Cloudflared tunnel URLs and exit ---
    if getattr(args, 'tunnelinfo', False):
        import glob
        # Print all cached Cloudflared tunnel URLs for user reference
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        url_files = glob.glob(os.path.join(cache_dir, 'cloudflared_url_*.txt'))
        if not url_files:
            print("[INFO] No Cloudflared tunnel link found in .cache.")
            sys.exit(0)
        for url_file in url_files:
            with open(url_file, 'r', encoding='utf-8') as f:
                link = f.read().strip()
                print(f"[Cloudflared Tunnel] {os.path.basename(url_file)}: {link}")
        sys.exit(0)

    if getattr(args, 'remotetunnel', False):
        # Start a new cloudflared tunnel if needed, print public URL
        import subprocess, re, shutil, psutil
        from datetime import datetime
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        os.makedirs(cache_dir, exist_ok=True)
        port = 3773
        def is_process_running(name):
            # Check if a process with the given name is running and not defunct
            for proc in psutil.process_iter(['name', 'cmdline', 'status']):
                try:
                    if ((name in proc.info['name'] or (proc.info['cmdline'] and any(name in c for c in proc.info['cmdline'])))
                        and proc.info.get('status', '').lower() not in ('zombie', 'defunct')):
                        return True
                except Exception:
                    continue
            return False
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
                print("[WARN] Cloudflared process detected but no cached URL found. Starting a new tunnel anyway...")
        elif url_files:
            print("[WARN] Cached Cloudflared URL(s) found but no process running. Removing stale cache and starting new tunnel...")
            for url_file in url_files:
                try:
                    os.remove(os.path.join(cache_dir, url_file))
                except Exception:
                    pass
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
                if ('cloudflared' in proc.info['name'] or (proc.info['cmdline'] and any('cloudflared' in c for c in proc.info['cmdline']))) and proc.info.get('status', '').lower() in ('zombie', 'defunct'):
                    proc.kill()
            except Exception:
                pass
        print(f"[INFO] Starting Cloudflared tunnel for http://localhost:{port} ...")
        proc = subprocess.Popen([
            cloudflared_bin, 'tunnel', '--url', f'http://localhost:{port}'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        public_url = None
        # Parse cloudflared output for the public URL and verify reachability
        for line in proc.stdout:
            if "trycloudflare.com" in line:
                print(line, end="")
                m = re.search(r'(https://[\w\-]+\.trycloudflare.com)', line)
                if m:
                    public_url = m.group(1)
                    print(f"[Cloudflared] Public URL: {public_url}")
                    import requests, time as _time
                    for attempt in range(5):
                        try:
                            resp = requests.get(public_url, timeout=10)
                            if resp.status_code < 400:
                                break
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

    # --- Handle shutdown: stop all background services and clean up ---
    if getattr(args, 'shutdown', False):
        import subprocess, platform, glob
        if platform.system().lower().startswith('win'):
            script = 'scripts/run_and_info_win.bat'
            cmd = [script, '--exit']
        else:
            script = 'scripts/run_and_info.sh'
            cmd = ['bash', script, '--exit']
        print(f"[INFO] Shutting down FastAPI and Cloudflared using {script} ...")
        subprocess.run(cmd)
        cache_dir = os.path.join(os.path.dirname(__file__), '../.cache')
        url_files = glob.glob(os.path.join(cache_dir, 'cloudflared_url_*.txt'))
        for url_file in url_files:
            try:
                os.remove(url_file)
            except Exception:
                pass
        print("[INFO] Removed cached Cloudflared tunnel URLs from .cache.")
        sys.exit(0)

    # --- Validate audio/mic arguments ---
    sources = [s for s in [args.audio, args.mic] if s]
    if len(sources) > 1:
        print("[ERROR] Multiple audio sources provided. Please specify only one of --audio or --mic.")
        print("Usage: --audio <file> OR --mic (not both)")
        sys.exit(1)

    # --- Detect file type if not provided ---
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

    # --- Microphone streaming not implemented (placeholder for future feature) ---
    if getattr(args, 'mic', False):
        from datetime import datetime
        # Placeholder: streaming microphone transcription is not implemented
        raise NotImplementedError("Microphone streaming transcription is not implemented in the current codebase. Please use --audio for file transcription.")
        # transcript, metadata = stream_microphone_transcription('mic_recording.wav')
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
    # --- Handle --audio: transcribe and cache before RAG ---
    if getattr(args, 'audio', False):
        import importlib.util, datetime
        audio_path = args.audio
        print(f"[INFO] Transcribing audio file: {audio_path}")
        # Dynamically import audio_transcribe
        transcribe_spec = importlib.util.spec_from_file_location(
            "audio_transcribe",
            str(Path(__file__).parent.parent / "intv" / "audio_transcribe.py")
        )
        audio_transcribe = importlib.util.module_from_spec(transcribe_spec)
        transcribe_spec.loader.exec_module(audio_transcribe)
        # Dynamically import audio_diarization
        diarize_spec = importlib.util.spec_from_file_location(
            "audio_diarization",
            str(Path(__file__).parent.parent / "intv" / "audio_diarization.py")
        )
        audio_diarization = importlib.util.module_from_spec(diarize_spec)
        diarize_spec.loader.exec_module(audio_diarization)
        # Transcribe audio
        segments = audio_transcribe.transcribe_audio_fastwhisper(audio_path)
        try:
            diarization = audio_diarization.diarize_audio(audio_path)
        except Exception as e:
            print(f"[WARN] Diarization failed: {e}")
            diarization = []
        # Merge diarization with transcription (simple overlap matching)
        for seg in segments:
            seg['speaker'] = None
            for d in diarization:
                if d['start'] <= seg['start'] < d['end']:
                    seg['speaker'] = d['speaker']
                    break
        # Save to .cache/yyyy-mm-dd_hhmm_audio.json
        cache_dir = Path('.cache')
        cache_dir.mkdir(exist_ok=True)
        dtstr = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
        cache_path = cache_dir / f"{dtstr}_audio.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'segments': segments}, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Audio transcription and diarization saved to {cache_path}")
        # Set args.file to the cache_path for downstream RAG processing
        args.file = str(cache_path)
        args.format = 'json'

    # --- Main RAG/LLM pipeline logic ---
    available_types = get_available_interview_types()
    type_keys = [t['key'] for t in available_types]
    if not available_types:
        print("[ERROR] No modules found. Ensure *_vars.json files exist in src/modules/.")
        sys.exit(1)
    if not args.type or args.type not in type_keys:
        print("\nSelect an interview/module type:")
        menu_types = []
        for idx, t in enumerate(available_types, 1):
            label = t.get('display', t['key'])
            menu_types.append(label)
            print(f"{idx}. {label}")
        print(f"{len(menu_types)+1}. Cancel")
        while True:
            choice = input(f"Enter choice [1-{len(menu_types)+1}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(menu_types)+1:
                break
            print(f"Invalid choice. Please enter a number 1-{len(menu_types)+1}.")
        if int(choice) == len(menu_types)+1:
            print("Exiting.")
            return
        args.type = available_types[int(choice)-1]['key']

    # --- Validate input file and chunk document ---
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists() or not is_valid_filetype(file_path, args.format):
            print(f"[ERROR] Invalid file or file format: {file_path}, {args.format}")
            raise ValueError('Invalid file or file format')
        try:
            chunks = chunk_document(file_path, args.format)
        except NotImplementedError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # --- Show progress indicator for RAG processing ---
    def show_rag_progress():
        import time
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
    progress_thread = threading.Thread(target=show_rag_progress)
    progress_thread.start()

    # --- RAG pipeline: external or embedded mode ---
    if args.rag_mode == 'external':
        rag_results = process_with_retriever_and_llm(
            chunks,
            mode=args.rag_mode,
            filetype=args.format,
            file_path=str(file_path)
        )
    else:
        rag_results = process_with_retriever_and_llm(chunks, mode=args.rag_mode)

    show_rag_progress.done = True
    progress_thread.join()

    # --- Load config and override with CLI args if provided ---
    from config import load_config, save_config
    config = load_config()
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
    # Force correct port if provider/port combo is mismatched
    if config.get('llm_provider', '').lower() == 'koboldcpp' and config.get('llm_api_port') == 5001:
        config['llm_api_port'] = 5001
    # Hardware acceleration detection
    import platform
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
        cutlass_available = os.environ.get('CUTLASS_PATH') is not None
        if cuda_available:
            extra_params['cuda'] = True
        if cutlass_available:
            extra_params['cutlass'] = True

    # --- Query backend for model info if using koboldcpp ---
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
                    if isinstance(data, dict):
                        model_name = data.get('model') or data.get('model_name') or data.get('model_path')
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

    # --- Call the selected module for postprocessing/output ---
    if not getattr(args, 'gui', False):
        from modules.dynamic_module import dynamic_module_output
        module_key = args.type
        provided_data = None
        if hasattr(args, f'{module_key}_data'):
            provided_data = getattr(args, f'{module_key}_data')
        try:
            result = dynamic_module_output(lookup_id=args.file, output_path=args.output, module_key=module_key, provided_data=provided_data)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        # Attach backend_info to output for user visibility
        if isinstance(result, dict):
            result['backend_info'] = backend_info
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # --- Optionally run the full RAG+LLM pipeline for the selected module ---
    from intv.llm import rag_llm_pipeline
    if args.file and args.type and not getattr(args, 'gui', False):
        rag_llm_pipeline(
            document_path=args.file,
            module_key=args.type,
            vars_json_path=getattr(args, 'vars_json_path', None),
            policy_prompt_path=getattr(args, 'policy_prompt_path', None),
            model=getattr(args, 'model', None),
            api_base=getattr(args, 'llm_api_base', None),
            api_key=getattr(args, 'llm_api_key', None),
            api_port=getattr(args, 'llm_api_port', None),
            provider=getattr(args, 'llm_provider', None)
        )
        return

if __name__ == "__main__":
    main()
