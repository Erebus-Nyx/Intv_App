import argparse
import sys

__version__ = "1.0.0"  # Update as needed

def get_parser():
    parser = argparse.ArgumentParser(
        description="Intv_App: Interview Automation & Document Analysis CLI"
    )
    parser.add_argument('--file', required=False, help='Path to the document or audio file')
    parser.add_argument('--type', required=False, choices=['pdf', 'docx', 'txt', 'audio'], help='Input file type')
    parser.add_argument('--model', required=False, default='hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M', help='Model name or ID (default: hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M)')
    parser.add_argument('--rag-mode', choices=['embedded', 'external'], default='embedded', help='RAG mode (default: embedded)')
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
    parser.add_argument('--shutdown', action='store_true', help='Shutdown all running Intv_App services and exit')
    parser.add_argument('--log-level', default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--version', action='store_true', help='Show version and exit')
    parser.add_argument('--about', action='store_true', help='Show about information and exit')
    parser.add_argument('--vars-json-path', required=False, help='Path to custom _vars.json file for variable config (optional)')
    parser.add_argument('--policy-prompt-path', required=False, help='Path to custom policy_prompt.yaml (optional)')
    # Add more arguments as needed for your workflow
    parser.set_defaults(disable_cuda=False)
    return parser

def parse_cli_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.version:
        print(f"Intv_App CLI version {__version__}")
        sys.exit(0)
    if args.about:
        print("Intv_App: Interview Automation & Document Analysis CLI\n" \
              "Modular, production-ready CLI for document and audio analysis with RAG and LLM support.")
        sys.exit(0)
    # Interactive prompt for intv type if not provided
    if not args.type:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from module_utils import get_available_interview_types
            available_types = get_available_interview_types()
            type_keys = [t['key'] for t in available_types]
            menu_types = [t.get('display', t['key']) for t in available_types]
            print("\nSelect an interview/module type:")
            for idx, label in enumerate(menu_types, 1):
                print(f"{idx}. {label}")
            print(f"{len(menu_types)+1}. Cancel")
            while True:
                choice = input(f"Enter choice [1-{len(menu_types)+1}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(menu_types)+1:
                    break
                print(f"Invalid choice. Please enter a number 1-{len(menu_types)+1}.")
            if int(choice) == len(menu_types)+1:
                print("Exiting.")
                sys.exit(0)
            args.type = type_keys[int(choice)-1]
        except Exception as e:
            print(f"[ERROR] Could not load interview/module types: {e}")
            sys.exit(1)
    return args

def main():
    args = parse_cli_args()
    # Pass args to main pipeline
    from main import main as app_main
    app_main(args)

if __name__ == '__main__':
    main()
