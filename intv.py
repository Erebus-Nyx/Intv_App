from . import __version__

def main():
    import sys
    import os
    from pathlib import Path
    # Ensure project root and src/ are in sys.path for src imports
    project_root = Path(__file__).parent.resolve()
    src_path = project_root / 'src'
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from rag_llm import RagLLM
    from config import Config
    from logger import logger
    from utils import print_banner
    import argparse
    parser = argparse.ArgumentParser(description='Document Analysis with RAG and LLM', add_help=True)
    parser.add_argument('--file', required=False, help='Path to the document')
    parser.add_argument('--type', required=False, choices=['pdf', 'docx', 'txt'], help='Document type')
    parser.add_argument('--model', required=False, default='hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M', help='Model name or ID (default: hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M)')
    parser.add_argument('--rag-mode', choices=['embedded', 'external'], default='embedded', help='RAG mode')
    parser.add_argument('--llm-provider', default='koboldcpp', help='LLM provider: koboldcpp (default: koboldcpp)')
    parser.add_argument('--llm-api-base', default=None, help='Base URL for LLM API (e.g., http://localhost)')
    parser.add_argument('--llm-api-key', default=None, help='API key for LLM provider (if required)')
    parser.add_argument('--llm-api-port', default=None, type=int, help='Port for LLM API (if required)')
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
    parser.add_argument('--shutdown', action='store_true', help='Shutdown all running INTV services and exit')
    parser.set_defaults(disable_cuda=False)
    args = parser.parse_args()

    if args.shutdown:
        import subprocess
        import signal
        import os
        import sys
        print("[INFO] Shutting down all INTV services...")
        try:
            subprocess.run(["pkill", "-f", "run_and_info"], check=False)
            subprocess.run(["pkill", "-f", "cloudflared"], check=False)
            subprocess.run(["pkill", "-f", "uvicorn"], check=False)
        except Exception as e:
            print(f"[WARN] Error during shutdown: {e}")
        print("[INFO] Shutdown signal sent. Exiting.")
        sys.exit(0)

    print_banner()
    logger.info(f"Starting Document Analysis with RAG and LLM, version {__version__}")

    config = Config(args.config)
    rag_llm = RagLLM(config)

    if args.file:
        rag_llm.load_document(args.file, args.type)

    if args.model:
        rag_llm.set_model(args.model, args.rag_mode, args.llm_provider, args.llm_api_base, args.llm_api_key, args.llm_api_port)

    # --- RAG mode selection ---
    if args.rag_mode == 'external':
        # Use external RAG app/service (e.g., via API call)
        if hasattr(rag_llm, 'run_external_rag'):
            rag_llm.run_external_rag()
        else:
            print("[ERROR] External RAG mode selected, but 'run_external_rag' is not implemented in RagLLM.")
            exit(1)
    elif args.gui:
        rag_llm.run_gui()
    else:
        rag_llm.run_terminal()

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / 'intv'))
    import importlib
    cli_module = importlib.import_module('cli')
    cli_module.main()
