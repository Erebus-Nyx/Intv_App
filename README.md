# INTV: Interview Automation & Document Analysis

> **Warning**
> 
> This project is in **alpha** status and is still under active development. Not all features are fully functional or stable. Expect breaking changes, incomplete modules, and evolving APIs. Use at your own risk and see the issues tracker for known limitations.

This project provides a robust, modular system for document analysis using Retrieval Augmented Generation (RAG) and LLMs. It is designed to process TXT, PDF, and DOCX files, extract structured variables, and generate narrative outputs for various interview and assessment modules.

---

## ⚠️ Host System Dependencies (Local/Non-Docker Installs)

Before running INTV locally (outside Docker), you **must** install the following system packages and tools on your host:

> **Note:** If you use the Linux startup script (`scripts/run_and_info.sh`), all required apt packages will be automatically installed for you on first run. You may still review the list below for reference or for manual setup.

### Linux (Debian/Ubuntu)
```sh
sudo apt update && sudo apt install -y python3 python3-venv python3-pip tesseract-ocr poppler-utils cloudflared python3-tk
```

### macOS (Homebrew)
```sh
brew install python@3.10 tesseract poppler cloudflared
# Tkinter is included with the official Python.org installer; if using Homebrew Python, also run:
brew install tcl-tk
```

### Windows
- **Python 3.10+**: [Download from python.org](https://www.python.org/downloads/)
- **Tkinter**: Included with the official Python installer (make sure to check the box during install)
- **Tesseract-OCR**: [Download installer](https://github.com/tesseract-ocr/tesseract/wiki)
- **Poppler**: [Download binaries](http://blog.alivate.com.au/poppler-windows/), add `bin/` to your PATH
- **cloudflared**: [Download from Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)

### LLM Backends (Optional, for local LLM inference)
- **KoboldCpp**: [Releases & setup](https://github.com/LostRuins/koboldcpp)

### Python Packages
After installing system dependencies, create a virtual environment and install Python requirements:
```sh
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:**
> - `tkinter` is a system package, not a pip package. If you see errors about missing `tkinter`, install it via your OS package manager.
> - `cloudflared` must be in your PATH for tunnel features to work.
> - For OCR and PDF support, both `tesseract-ocr` and `poppler-utils` are required.
> - If you use the web UI or CLI tunnel features, you must have `cloudflared` installed.
> - For LLM inference, install and run either Ollama or KoboldCpp as needed.

---

## 🚀 Quick Start

### Local Usage
1. Edit `config/config.yaml` and any `config/*.json` as needed.
2. (Recommended) Create and activate a Python virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   python -m src.main --file yourfile.pdf --type pdf
   ```
   - Use `--cpu` to force CPU-only mode (disables CUDA and CUTLASS, for AMD/ARM compatibility).
   - Use `--gui` for the web interface (FastAPI, see below).
   - Use `--cloudflare` to enable a public Cloudflare tunnel for the web UI (see below).

### Local Web UI (FastAPI)
- Start the web server:
  ```sh
  ./scripts/cloudflared-entrypoint.sh --gui [--cloudflare]
  ```
  - `--cloudflare` enables a public Cloudflare tunnel (see below).
  - `--no-cloudflare` disables the tunnel (default).
  - `--cloudflare-env` uses the `USE_CLOUDFLARE_TUNNEL` environment variable.
- Access the UI at [http://localhost:3773](http://localhost:3773)
- If Cloudflare is enabled, a public URL will be shown in the logs.

### Docker Compose (GPU or CPU)
- Edit `.env` to set your environment variables (see below for Cloudflare options).
- Build and run (GPU):
  ```sh
  docker-compose up --build -d
  ```
- For CPU-only, change the Dockerfile in `docker-compose.yml` to `docker/Dockerfile.cpu`.
- The app will be available at [http://localhost:3773](http://localhost:3773)
- To enable Cloudflare tunnel in Docker, set `USE_CLOUDFLARE_TUNNEL=true` in `.env`.

---

## 🌐 Cloudflare Tunnel Integration & Security
- **Default:** Cloudflare tunnel is **off** unless enabled.
- **Enable in Docker:** Set `USE_CLOUDFLARE_TUNNEL=true` in `.env`.
- **Enable in CLI:** Pass `--cloudflare` to the entrypoint script, or use `--cloudflare-env` to use the environment variable.
- The tunnel will forward the public URL to your running app on port 3773.
- **Access Split:**
  - All HTTP(S) endpoints (including `/api`, `/api/v1`, `/api/admin`, `/api/data`) can be protected by Cloudflare Access or WAF.
  - The WebSocket endpoint (`/ws`) is public by default, but can be restricted via Cloudflare Access/WAF if needed.
- No config warning will appear; a minimal config is created automatically for free tunnels.

---

## 🗂️ Directory Structure
- `src/` - All Python source code
- `config/` - All user-editable configuration and per-module defaults (JSON only)
- `docker/` - Docker deployment files (`Dockerfile.gpu`, `Dockerfile.cpu`)
- `scripts/` - Entrypoint and utility scripts
- `requirements.txt` - All dependencies
- `setup.py` - Project install/setup

---

## 🛠️ Features & Modules
- Modular narrative modules (Adult, Child, AR, Collateral, Home Assessment, Allegations, Dispo, EA, Staffing)
- Unified variable sourcing: DB → JSON defaults (in config/) → config.yaml → user prompt
- Automatic chunking for TXT, PDF (with OCR fallback), and DOCX
- **Default LLM provider:** Now defaults to **KoboldCpp** (`llm_provider: koboldcpp`, port `5001`). To use OpenAI, set `--llm-provider` and `--llm-api-port` as needed.
- LLM and RAG integration (OpenAI, KoboldCpp)
- SQLite backend for variable persistence and LLM reference
- User-prompt fallback for missing variables, with clarification/finalization logic
- All modules and configs enforce narrative, variable, and compliance standards per guidelines/template
- API endpoints under `/api/v1/`, `/api/admin/`, `/api/data/` (see FastAPI docs at `/api`)
- WebSocket endpoint at `/ws` for real-time UI updates (public by default)
- No login or multi-user logic; admin features are toggled in the UI by a button

---

## 📄 Supported Input Types

- **Audio:** WAV, MP3, M4A, MP4, PDF (for transcription)
- **Text/Document:** TXT, RTF, DOCX, PDF (for document analysis)
- **Images:** JPG (basic support; OCR can be added as needed)

You can use the CLI or web UI to process these files. The backend will automatically route audio files and PDF files to the transcription pipeline, and text/doc files (including PDF) to the RAG/documentation pipeline.

### CLI Usage Examples

- **Transcribe audio:**
  ```sh
  python src/main.py --audio path/to/audiofile.wav
  ```
- **Process document (TXT, RTF, DOCX):**
  ```sh
  python src/main.py --file path/to/document.docx
  ```
- **Record from microphone:**
  ```sh
  python src/main.py --mic
  ```
- **Interactive mode (prompt for file or mic):**
  ```sh
  python src/main.py
  ```

### Web UI
- Upload TXT, RTF, DOCX, or audio files directly in the browser.
- The backend will process the file appropriately (RAG for text/docs, transcription for audio).
- Admin features are toggled by a button in the UI (no login required).

---

## 📝 Configuration & Compliance
- All config and defaults are user-editable and reloadable at runtime.
- LLM policy, writing style, and compliance are controlled by `config/policy_prompt.yaml`.
- See the compliance checklist and automated check script in `src/compliance_check.py`.

---

## 🔒 Security & Admin
- Admin mode is toggled in the UI (no login or password required).
- API endpoints are split into `/api/v1`, `/api/admin`, and `/api/data`.
- Protect HTTP(S) endpoints with Cloudflare Access or WAF as needed.
- WebSocket endpoint (`/ws`) is public by default, but can be restricted via Cloudflare if desired.
- `.gitignore` and `.dockerignore` are hardened to prevent secrets, uploads, and sensitive files from being committed or included in images.

---

## 📦 Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies.
- For Docker: NVIDIA GPU (for GPU image), or use CPU image for ARM/AMD/CPU-only.

---

## 🧑‍💻 Development & Extending
- Add new modules in `src/modules/` and corresponding config in `config/`.
- Update `llm_db.py` for variable hints and defaults.
- Follow the compliance checklist for all new modules/configs.

---

## 🆘 Troubleshooting
- For OCR issues: ensure `tesseract-ocr` and `pdf2image` are installed.
- For LLM/RAG issues: check your config and provider/model settings.
- For GPU: ensure NVIDIA drivers and CUDA are installed. Use `--cpu` to disable GPU.

---

## Windows/WSL Startup Automation

The script `scripts/run_and_info_win.bat` now supports automatic startup registration for WSL:

- To add the script to your WSL startup (so it runs automatically when WSL launches):
  ```sh
  scripts/run_and_info_win.bat --startup true
  ```
- To remove the script from WSL startup:
  ```sh
  scripts/run_and_info_win.bat --startup false
  ```
This works by adding/removing a line in your `~/.bashrc` that calls the batch script via bash.

## Running as a Background Service

Both the Linux (`scripts/run_and_info.sh`) and Windows (`scripts/run_and_info_win.bat`) scripts are now functionally identical:

- They start FastAPI and Cloudflared in the background, automatically freeing ports 3773/3774 as needed.
- If `cloudflared` is missing, it will be downloaded automatically for your platform.
- PID files are created in `/tmp/` (Linux) or `%TEMP%\` (Windows) for process management.
- Use the `--exit` argument to stop all related FastAPI and Cloudflared processes and clean up PID files.
- Logs are written to `fastapi_<port>.log` and `cloudflared_<port>.log` in the current directory.
- On Windows, public URL detection is not automatic; check the log file for the Cloudflare public URL.

Example usage:

```sh
# Start services (Linux)
./scripts/run_and_info.sh

# Start services (Windows)
scripts\run_and_info_win.bat

# Stop all services (Linux)
./scripts/run_and_info.sh --exit
# Stop all services (Windows)
scripts\run_and_info_win.bat --exit
```

## API Endpoint to Trigger CLI

A new API endpoint is available:

- `POST /api/generate` — Triggers the CLI script in the background. Returns the process output. 

---

## LLM Provider Setup

The system supports two LLM providers:
- **KoboldCpp** (local, OpenAI-compatible API)
- **OpenAI** (cloud)

---

### Configuration

- Default provider: `koboldcpp`
- To use OpenAI, set `llm_provider` to `openai` in your config or CLI.
- For KoboldCpp, ensure the API is running and set the correct port (default: 5001).

---

## 🧠 Recommended LLM Models for Reasoning (7B–30B)

You can use any of the following models for strong reasoning performance with this app (KoboldCpp, OpenAI):

1. **Phi-3 Mini (8B)**
   - Excellent reasoning, strong performance for its size.
   - Model: `microsoft/Phi-3-mini-128k-instruct` (GGUF: `Phi-3-mini-128k-instruct.Q4_K_M.gguf`)

2. **Llama-3 8B Instruct**
   - Very strong reasoning and instruction-following.
   - Model: `meta-llama/Meta-Llama-3-8B-Instruct` (GGUF: `Llama-3-8B-Instruct.Q4_K_M.gguf`)

3. **Mistral 7B Instruct**
   - Compact, fast, and surprisingly capable at reasoning.
   - Model: `mistralai/Mistral-7B-Instruct-v0.2` (GGUF: `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf`)

4. **Nous Hermes 2 - Llama-3 8B**
   - Fine-tuned for reasoning and conversation.
   - Model: `NousResearch/Nous-Hermes-2-Llama-3-8B` (GGUF: `Nous-Hermes-2-Llama-3-8B.Q4_K_M.gguf`)

5. **Llama-2 13B Chat**
   - Larger context and strong reasoning.
   - Model: `meta-llama/Llama-2-13b-chat-hf` (GGUF: `Llama-2-13B-chat.Q4_K_M.gguf`)

6. **Qwen2 18B Chat**
   - Excellent reasoning and multi-turn ability, strong at 18B.
   - Model: `Qwen/Qwen2-18B-Chat` (GGUF: `Qwen2-18B-Chat.Q4_K_M.gguf`)

7. **Mixtral 8x22B (MoE, ~22B active)**
   - State-of-the-art mixture-of-experts, very strong at reasoning.
   - Model: `mistralai/Mixtral-8x22B-Instruct-v0.1` (GGUF: `Mixtral-8x22B-Instruct-v0.1.Q4_K_M.gguf`)

8. **Current Default: Phi-4 Reasoning Plus (Q6_K_XL)**
   - Outstanding at reasoning, especially for professional and structured tasks.
   - Model: `hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q6_K_XL`
   - This is the default in your app (`--model` argument).

> All of these are available in GGUF format. For best results, use Q4_K_M or Q6_K_S quantizations for a balance of speed and reasoning quality.

---

## License
See `LICENSE` for details.

---

## Credits
- Inspired by best practices from open-source LLM, RAG, and document automation projects.
- Cloudflare integration modeled after KoboldCpp and similar projects.

---

## 🐳 Docker Entrypoint & Modes

- The Docker image now supports both GUI (web) and CLI (terminal) modes via the `APP_MODE` environment variable.
- **Default:** `APP_MODE=gui` (FastAPI web UI)
- To run the CLI pipeline in Docker, set `APP_MODE=cli` in your `.env` or `docker-compose.yml`.
- The entrypoint script (`scripts/cloudflared-entrypoint.sh`) will launch the correct mode automatically.
- Example:
  ```yaml
  environment:
    APP_MODE: cli  # or gui
  ```
- All other environment variables (LLM, RAG, admin, etc.) are passed through as before.

---

## 🛠️ Process Management & Service Control

- The app now includes robust process management for FastAPI and Cloudflared services.
- The Docker entrypoint script (`cloudflared-entrypoint.sh`) and Python utilities (`server_utils.py`) will:
  - Detect and avoid duplicate cloudflared tunnels.
  - Start cloudflared and FastAPI if not running.
  - Provide user-friendly status messages and progress bars for tunnel startup.
  - Allow graceful shutdown of all related services (CLI, Docker, or script).
- Use the CLI or scripts to start/stop services, or call `shutdown_services()` from Python.

---

## 🧪 Testing & Robustness

- Test utilities are provided for all major pipeline components:
  - `test_server_utils.py`: Tests process management, health checks, and logging.
  - `test_pipeline_utils.py`: Tests RAG chunking, OCR, audio transcription, microphone streaming, and LLM chunk analysis.
  - `test_cli_and_pipeline.py`: Tests CLI argument handling and error cases.
  - `test_docker_entrypoint.sh`: Tests Docker entrypoint logic for both CLI and GUI modes.
- To run all tests:
  ```sh
  pytest test_src/
  bash test_src/test_docker_entrypoint.sh
  ```
- Ensure `pytest` is installed and run from the project root for correct imports.

---
