# Intv_App: Interview Automation & Document Analysis

> **Warning**
> 
> This project is in **alpha** status and is still under active development. Not all features are fully functional or stable. Expect breaking changes, incomplete modules, and evolving APIs. Use at your own risk and see the issues tracker for known limitations.

This project provides a robust, modular CLI and web-based system for document analysis using Retrieval Augmented Generation (RAG) and LLMs. It is designed to process TXT, PDF, and DOCX files, extract structured variables, and generate narrative outputs for various interview and assessment modules.

---

## 🚀 Quick Start

### Local CLI Usage
1. Edit `config/config.yaml` and any `config/*.json` as needed.
2. (Recommended) Create and activate a Python virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Run the CLI:
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

## 🌐 Cloudflare Tunnel Integration
- **Default:** Cloudflare tunnel is **off** unless enabled.
- **Enable in Docker:** Set `USE_CLOUDFLARE_TUNNEL=true` in `.env`.
- **Enable in CLI:** Pass `--cloudflare` to the entrypoint script, or use `--cloudflare-env` to use the environment variable.
- The tunnel will forward the public URL to your running app on port 3773.
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
- LLM and RAG integration (OpenAI, Ollama, KoboldCpp, etc.)
- SQLite backend for variable persistence and LLM reference
- User-prompt fallback for missing variables, with clarification/finalization logic
- All modules and configs enforce narrative, variable, and compliance standards per guidelines/template
- API endpoints under `/api/v1/` (see FastAPI docs at `/api/v1/docs`)

---

## 📝 Configuration & Compliance
- All config and defaults are user-editable and reloadable at runtime.
- LLM policy, writing style, and compliance are controlled by `config/policy_prompt.yaml`.
- See the compliance checklist and automated check script in `src/compliance_check.py`.

---

## 🔒 Security & Admin
- Admin user/password set via `.env` or Docker Compose environment.
- All API endpoints require authentication.

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

## License
See `LICENSE` for details.

---

## Credits
- Inspired by best practices from open-source LLM, RAG, and document automation projects.
- Cloudflare integration modeled after KoboldCpp and similar projects.

---
