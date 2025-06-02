import argparse
from src.utils import is_valid_filetype, chunk_document
from src.rag import process_with_rag
from src.llm import analyze_chunks
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

def transcribe_audio_file(audio_path, model_size="small", diarization=False, vad=False, output_format="txt"):
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

def main():
    """
    Main CLI entrypoint for document/audio analysis and RAG/LLM workflow.
    Handles argument parsing, input validation, and pipeline orchestration.
    """
    parser = argparse.ArgumentParser(description='Document Analysis with RAG and LLM')
    parser.add_argument('--file', required=True, help='Path to the document')
    parser.add_argument('--type', required=True, choices=['pdf', 'docx', 'txt'], help='Document type')
    parser.add_argument('--model', required=False, default='hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M', help='Model name or ID (default: hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M)')
    parser.add_argument('--rag-mode', choices=['embedded', 'external'], default='embedded', help='RAG mode')
    parser.add_argument('--llm-provider', default='ollama', help='LLM provider: openai, ollama, koboldcpp, etc. (default: ollama)')
    parser.add_argument('--llm-api-base', default=None, help='Base URL for LLM API (e.g., https://api.openai.com or http://localhost)')
    parser.add_argument('--llm-api-key', default=None, help='API key for LLM provider (if required)')
    parser.add_argument('--llm-api-port', default=None, type=int, help='Port for LLM API (if required)')
    parser.add_argument('--nowada', action='store_false', dest='disable_cuda', help='Disable CUDA (GPU) acceleration for supported LLMs (AR mode, disables CUDA)')
    parser.add_argument('--config', default=None, help='Path to config.json (overrides defaults)')
    parser.add_argument('--gui', action='store_true', help='Enable web-based GUI (if not set, defaults to CLI only)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only mode (disables CUDA and CUTLASS, for AMD/ARM compatibility)')
    parser.add_argument('--audio', type=str, help='Path to audio file for transcription')
    parser.add_argument('--mic', action='store_true', help='Use microphone for streaming transcription')
    parser.add_argument('--diarization', action='store_true', help='Enable speaker diarization')
    parser.add_argument('--vad', action='store_true', help='Enable voice activity detection (VAD)')
    parser.add_argument('--output-format', choices=['txt', 'json'], default='txt', help='Transcription output format')
    parser.add_argument('--defaults', action='store_true', help='Bypass data source and use default values for all variables')
    parser.add_argument('--remotetunnel', action='store_true', help='Start a Cloudflare tunnel for remote access (like koboldcpp)')
    parser.set_defaults(disable_cuda=False)  # Default: CUDA ON
    args = parser.parse_args()
    sources = [s for s in [args.audio, args.mic] if s]
    if len(sources) > 1:
        print("[ERROR] Multiple audio sources provided. Please specify only one of --audio or --mic.")
        print("Usage: --audio <file> OR --mic (not both)")
        sys.exit(1)
    if args.audio:
        print(f"Transcribing audio file: {args.audio}")
        transcript, metadata = transcribe_audio_file(args.audio, diarization=args.diarization, vad=args.vad, output_format=args.output_format)
        transcript_path = Path(args.audio).with_suffix(f'.{args.output_format}')
        print(f"Transcription saved to {transcript_path}")
        # Pass to RAG pipeline here
        # rag.process(transcript_path, metadata=metadata)
    elif args.mic:
        print("Starting microphone streaming transcription...")
        output_path = Path("mic_recording.wav")
        transcript, metadata = stream_microphone_transcription(output_path, diarization=args.diarization, vad=args.vad, output_format=args.output_format)
        print(f"Transcription saved to {output_path}.{args.output_format}")
        # Pass to RAG pipeline here
        # rag.process(str(output_path) + f'.{args.output_format}', metadata=metadata)
    else:
        print("No audio source provided. Use --audio <file> or --mic.")
        sys.exit(1)

    # CLI menu if not in GUI mode
    if not getattr(args, 'gui', False):
        print("\nSelect an option:")
        print("1. Adult interview")
        print("2. Child interview")
        print("3. Alternative response")
        print("4. Collateral interview")
        print("5. Cancel")
        while True:
            choice = input("Enter choice [1-5]: ").strip()
            if choice in {'1','2','3','4','5'}:
                break
            print("Invalid choice. Please enter a number 1-5.")
        if choice == '5':
            print("Exiting.")
            return
        # Map choice to module/type for downstream logic
        if choice == '1':
            args.type = 'adult'
        elif choice == '2':
            args.type = 'child'
        elif choice == '3':
            args.type = 'ar'
        elif choice == '4':
            args.type = 'col'

    file_path = Path(args.file)
    if not file_path.exists() or not is_valid_filetype(file_path, args.type):
        raise ValueError('Invalid file or file type')

    chunks = chunk_document(file_path, args.type)

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
        print("[INFO] CLI mode (default)")

    # Start RAG progress indicator in a thread
    progress_thread = threading.Thread(target=show_rag_progress)
    progress_thread.start()

    if args.rag_mode == 'external':
        rag_results = process_with_rag(
            chunks,
            mode=args.rag_mode,
            filetype=args.type,
            file_path=str(file_path)
        )
    else:
        rag_results = process_with_rag(chunks, mode=args.rag_mode)

    # Signal progress thread to stop
    show_rag_progress.done = True
    progress_thread.join()

    # Load config and optionally override with CLI args
    from src.config import load_config, save_config
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
        import json
        from pathlib import Path
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open('r', encoding='utf-8') as f:
                user_config = json.load(f)
            config.update(user_config)
    # Override config with CLI args if provided
    for k in ['llm_api_base', 'llm_api_key', 'llm_api_port', 'llm_provider', 'model', 'external_rag', 'purge_variables', 'name']:
        v = getattr(args, k, None)
        if v is not None:
            config[k] = v
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

    llm_output = analyze_chunks(
        rag_results,
        model=config['model'],
        api_base=config['llm_api_base'],
        api_key=config['llm_api_key'],
        api_port=config['llm_api_port'],
        provider=config['llm_provider'],
        extra_params=extra_params if extra_params else None
    )

    # After LLM variable extraction and narrative review, enforce mandatory content/questions
    if args.type == 'child':
        # For child interview, ensure truth/lie and pre-interview topics are explicitly asked
        mandatory_questions = [
            ("Did you explain the difference between telling the truth and a lie to the child? (yes/no)", "explained_truth_lie"),
            ("Did the child demonstrate understanding of the difference between truth and lie? (yes/no)", "child_understands_truth_lie"),
            ("Did the child agree to only tell the truth during the interview? (yes/no)", "child_agreed_truth"),
        ]
        for q, var in mandatory_questions:
            val = get_llm_variable(args.file, var)
            if not val:
                user_val = input(f"[MANDATORY] {q} ")
                set_llm_variable(args.file, var, user_val)
    elif args.type == 'adult':
        # For adult interview, ensure 3010 advisement and agreement to speak are explicitly asked
        mandatory_questions = [
            ("Did you advise the adult of the 3010 warning? (yes/no)", "advised_3010"),
            ("Did the adult agree to speak with you? (yes/no)", "adult_agreed_speak"),
        ]
        for q, var in mandatory_questions:
            val = get_llm_variable(args.file, var)
            if not val:
                user_val = input(f"[MANDATORY] {q} ")
                set_llm_variable(args.file, var, user_val)
    # Now print the final output
    print(llm_output)

    # After menu selection, call the corresponding module
    if not getattr(args, 'gui', False):
        from src.llm_db import get_needed_variables_for_module, set_llm_variable, get_llm_variable, get_module_yaml_defaults
        from src.llm import analyze_chunks
        # Map module name
        module_map = {
            'adult': ('doc_intv_adult', 'intv_adult_output'),
            'child': ('doc_intv_child', 'intv_child_output'),
            'ar': ('doc_intv_ar', 'intv_ar_output'),
            'col': ('doc_intv_col', 'intv_col_output'),
        }
        mod_name, func_name = module_map[args.type]
        needed_vars = get_needed_variables_for_module(mod_name)
        # Analyze context with LLM to extract variable values
        print(f"[INFO] Extracting variables for {mod_name} using LLM...")
        context = '\n'.join(chunks)
        # Calculate degree of relevance based on variable hints
        keywords = set()
        for var, meta in needed_vars.items():
            if isinstance(meta, dict) and 'hint' in meta:
                for word in meta['hint'].split():
                    if len(word) > 3:
                        keywords.add(word.lower().strip('.,:;()'))
        found_keywords = [kw for kw in keywords if kw in context.lower()]
        degree = len(found_keywords) / max(1, len(keywords))
        print(f"[INFO] Initial evaluation: {len(found_keywords)} of {len(keywords)} relevant keywords found ({degree:.0%} relevance)")
        print(f"[INFO] Relevant keywords found: {found_keywords}")
        # Ask for confirmation
        while True:
            confirm = input("Continue with LLM extraction? (y/n): ").strip().lower()
            if confirm in {'y', 'n'}:
                break
        if confirm == 'n':
            print("[INFO] Operation cancelled. Returning to CLI menu.")
            return
        # Proceed with LLM extraction
        # Load YAML defaults for the selected module
        yaml_defaults = get_module_yaml_defaults(mod_name)
        # Prefer DB value, then LLM, then YAML default
        for var, meta in needed_vars.items():
            # Prefer DB value, then LLM, then YAML default
            prev_val = get_llm_variable(args.file, var)
            llm_result = None
            if not prev_val:
                prompt = f"Extract the value for '{var}' ({meta['hint']}) from the following context. If not found, return None. Context:\n{context}"
                llm_result = analyze_chunks([prompt], model=config['model'], api_base=config['llm_api_base'], api_key=config['llm_api_key'], api_port=config['llm_api_port'], provider=config['llm_provider'])
                if llm_result:
                    set_llm_variable(args.file, var, llm_result)
            # If still not set, use YAML default
            if not get_llm_variable(args.file, var) and var in yaml_defaults and yaml_defaults[var] not in (None, ""):
                set_llm_variable(args.file, var, yaml_defaults[var])
            # If still not set, prompt user
            if not get_llm_variable(args.file, var):
                user_val = input(f"No value found for '{var}' ({meta['hint']}). Please provide a value: ")
                set_llm_variable(args.file, var, user_val)
        # LLM variable extraction and narrative refinement
        for var, meta in needed_vars.items():
            # First pass: extract value, focusing on direct relevance (e.g., Peterborough info)
            prompt = (
                f"Extract the value for '{var}' ({meta['hint']}) from the following context. "
                f"Focus on information directly relevant to Peterborough or the interview subject. "
                f"If not found, return None. Context:\n{context}"
            )
            llm_result = None
            prev_val = get_llm_variable(args.file, var)
            if not prev_val:
                llm_result = analyze_chunks([prompt], model=config['model'], api_base=config['llm_api_base'], api_key=config['llm_api_key'], api_port=config['llm_api_port'], provider=config['llm_provider'])
                if llm_result:
                    set_llm_variable(args.file, var, llm_result)
            # Second pass: cross-check for related/cross-relevant material
            if not get_llm_variable(args.file, var):
                cross_prompt = (
                    f"Review the context for any cross-relevant or supporting information for '{var}' ({meta['hint']}). "
                    f"If found, append it to the value. Do not invent or add information not present in the context. Context:\n{context}"
                )
                cross_result = analyze_chunks([cross_prompt], model=config['model'], api_base=config['llm_api_base'], api_key=config['llm_api_key'], api_port=config['llm_api_port'], provider=config['llm_provider'])
                if cross_result:
                    set_llm_variable(args.file, var, cross_result)
            # Use YAML default if still not set
            if not get_llm_variable(args.file, var) and var in yaml_defaults and yaml_defaults[var] not in (None, ""):
                set_llm_variable(args.file, var, yaml_defaults[var])
            # Prompt user if still not set
            if not get_llm_variable(args.file, var):
                user_val = input(f"No value found for '{var}' ({meta['hint']}). Please provide a value: ")
                set_llm_variable(args.file, var, user_val)
        # Final narrative review and professionalization
        final_narrative = get_llm_variable(args.file, 'narrative')
        review_prompt = (
            "Review the following narrative for professional tone, ease of reading, and clarity. "
            "Highlight any areas where clarifying information may be needed. "
            "Ensure that no information is added by the AI or any outside source other than the user. "
            "If any information appears to be fabricated or not present in the original context, flag it.\n"
            f"Narrative:\n{final_narrative}"
        )
        reviewed_narrative = analyze_chunks([review_prompt], model=config['model'], api_base=config['llm_api_base'], api_key=config['llm_api_key'], api_port=config['llm_api_port'], provider=config['llm_provider'])
        set_llm_variable(args.file, 'narrative', reviewed_narrative)
        # Call the selected module's output function
        import importlib
        mod = importlib.import_module(f"src.{mod_name}")
        func = getattr(mod, func_name)
        print(f"[INFO] Running {mod_name} module...")
        if args.defaults:
            print("Bypassing data source: all variables will use default values from config.")
            provided_data = {var: meta.get('default', '') for var, meta in needed_vars.items()}
        else:
            provided_data = None
        result = func(lookup_id=args.file, output_path=args.output, **{f'{args.type}_data': provided_data})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

if __name__ == '__main__':
    main()
