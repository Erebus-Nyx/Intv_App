"""
README.md - Project documentation for python-cli-app

> **Warning**
> 
> This project is in **alpha** status and is still under active development. Not all features are fully functional or stable. Expect breaking changes, incomplete modules, and evolving APIs. Use at your own risk and see the issues tracker for known limitations.

This project provides a robust, modular CLI and  web-based system for document analysis using Retrieval Augmented Generation (RAG) and LLMs. It is designed to process TXT, PDF, and DOCX files, extract structured variables, and generate narrative outputs for various interview and assessment modules.

## Key Features
- Modular narrative modules (Adult, Child, AR, Collateral, Home Assessment, Allegations, Dispo, EA, Staffing)
- Unified variable sourcing: DB → JSON defaults (in config/) → config.yaml → user prompt
- Centralized user configuration in `config/config.yaml` and per-module defaults in `config/*.json`
- Automatic chunking for TXT, PDF (with OCR fallback), and DOCX
- LLM and RAG integration (OpenAI, Ollama, KoboldCpp, etc.)
- CLI menu for module selection; future support for web GUI (`--gui`)
- SQLite backend for variable persistence and LLM reference
- User-prompt fallback for missing variables, with clarification/finalization logic
- All modules and configs enforce narrative, variable, and compliance standards per guidelines/template
- Extensive inline code comments for troubleshooting and maintainability

## Directory Structure
- `src/` - All Python source code
  - `mod_*.py` - Modular support files (remain in src/)
  - `modules/` - All document/narrative modules (no 'doc_' prefix)
  - other core modules (main.py, utils.py, etc.)
- `config/` - All user-editable configuration and per-module defaults (JSON only)
- `docker/` - Docker deployment files
  - `Dockerfile.gpu` - Dockerfile for GPU-enabled deployment
- `requirements.txt` - All dependencies, including OCR and LLM support
- `setup.py` - Project install/setup

## Usage
1. Edit `config/config.yaml` and any `config/*.json` as needed.
2. Run the CLI: `python -m src.main --file yourfile.pdf --type pdf`
3. Use `--cpu` to force CPU-only mode (disables CUDA and CUTLASS, for AMD/ARM compatibility). By default, CUDA and CUTLASS are enabled if available.
4. Follow prompts for module selection and variable confirmation.
5. For missing variables, you will be prompted interactively. Modules will indicate if clarification is needed or if the output is final.
6. Use `--gui` for future web interface (currently placeholder).

## Local Setup: Using a Virtual Environment
It is recommended to use a Python virtual environment for local installs and dependency management:

1. Create a virtual environment (Windows):
   ```sh
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Windows (cmd):
     ```sh
     .venv\Scripts\activate
     ```
   - On Windows (PowerShell):
     ```sh
     .venv\Scripts\Activate.ps1
     ```
   - On WSL/Linux/Mac:
     ```sh
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Proceed with the usage instructions below.

## Troubleshooting
- All major modules are documented and use consistent variable sourcing logic.
- Inline comments in each module explain the flow and variable resolution order.
- For OCR issues, ensure `pytesseract` and `pdf2image` are installed and Tesseract is available on your system.
- For LLM/RAG issues, check your config and ensure the correct provider/model is set.
- For GPU acceleration, ensure you have a compatible NVIDIA GPU and CUDA drivers. CUTLASS support is enabled if available. Use `--cpu` to disable GPU support.

## Extending
- Add new modules by creating a Python file in `src/modules/` and a corresponding JSON defaults file in `config/` (e.g., `intv_newtype_vars.json`).
- Update `llm_db.py` with variable hints and defaults for new modules.
- All config and defaults are user-editable and reloadable at runtime.
- Ensure all new modules and configs:
  - Require all variables and narrative sections as specified in the latest guidelines and template.
  - Enforce first-person, professional, narrative (paragraph) style with clear party identification and demographic info.
  - Never use bullet points, lists, or headers in the narrative body.
  - Include logic for clarification prompts and finalization flags.

## LLM Policy & Writing Guidelines
- LLM behavior, compliance, and writing style are controlled by a structured policy prompt in `config/policy_prompt.yaml`.
- This YAML file contains clear sections such as `behavior`, `writing_guidelines`, and `compliance`.
- The system automatically loads and concatenates these sections for every LLM call, ensuring consistent output and compliance regardless of backend or model.
- Edit `config/policy_prompt.yaml` to update LLM policy, writing rules, or compliance requirements at any time.

**Policy Prompt Requirements:**
- All documentation must be in first-person, professional, and narrative (no bullet points, no headers).
- Parties must be clearly identified (no abbreviations, state relation/role/age as appropriate).
- All demographic and personal information (stated or denied) must be included.
- Language must be clear, concise, formal, and avoid speculation or unsupported statements.
- Narrative must follow the story-like, paragraph structure as in the template.

Example `policy_prompt.yaml`:
```yaml
behavior: |
  Always act professionally, neutrally, and with privacy compliance.
  Never provide legal or medical advice.
  Always clarify when uncertain.

writing_guidelines: |
  Write in first-person narrative from the perspective of the investigator.
  Use clear, concise, and formal language.
  Do not use bullet points, lists, or headers in the narrative body.
  Structure all documentation as a story in paragraph form.
  Clearly identify all parties by name and role/relation/age as appropriate.
  Include all demographic and personal information provided or denied.
  Avoid speculation and unsupported statements.
  Maintain a professional tone at all times.

compliance: |
  Adhere to all organizational and regulatory requirements.
  Do not output or request sensitive personal information unless explicitly required for the task.
  Ensure all documentation is suitable for legal and professional review.
```

## Narrative Structure & Required Content
- Each module's narrative must include all required sections and variables as specified in the guidelines and template, such as:
  - Name, Role/Relation, Age, Location, Contact Method, Rapport, Observation, Family, Chores, Allegation, Screening, Summary, etc.
- All modules output `clarification_needed`, `pending_questions`, and `is_final` flags to support interactive completion.
- See the documentation template and interview guidelines for full narrative structure and compliance requirements.

## Module/Config Compliance Checklist
To ensure every module and config remains fully compliant with the interview guidelines and narrative template, use the following checklist when creating or updating modules:

- [ ] **All required variables and narrative sections are present**
  - Name, Role/Relation, Age, Location, Address, OthersPresent, Rapport, Observation, Family, Chores, Allegation, Screening, Summary, etc. (see template/guidelines for your module)
- [ ] **Config file is JSON** (not YAML, except for `policy_prompt.yaml`)
- [ ] **Each variable has a clear `hint` and, if possible, a default**
- [ ] **Section configs in code match the config JSON and narrative template**
- [ ] **Data-driven variable resolution order is followed**
  - Provided → DB → config JSON default → user prompt
- [ ] **Narrative output is always in first-person, professional, paragraph/story form**
  - No bullet points, lists, or headers
  - Clear party identification (no abbreviations; state relation/role/age as appropriate)
  - All demographic and personal information (stated or denied) included
  - Language is clear, concise, formal, and avoids speculation/unsupported statements
- [ ] **Module outputs `clarification_needed`, `pending_questions`, and `is_final` flags**
- [ ] **LLM policy prompt is loaded and enforced for all LLM calls**
- [ ] **No pronoun or name logic in code; LLM handles context**
- [ ] **All configs and modules are user-editable and reloadable at runtime**
- [ ] **Documentation and inline comments are updated as needed**

**Tip:** Use this checklist as a reference for code reviews, onboarding, or when extending the system. For automated checks, use the provided script:

### Automated Compliance Check
- Run `python src/compliance_check.py` to automatically validate config structure, required keys, and hints for all modules/configs.
- The script will flag missing keys, missing hints, and provide reminders for manual narrative/code review.
- This is the preferred method for ongoing compliance validation.

For more details, see the script source at `src/compliance_check.py`.

## API & Web GUI (FastAPI)

- The app now exposes all API endpoints under a versioned path: `/api/v1/`
- Example endpoints:
  - `POST /api/v1/login` — User login
  - `POST /api/v1/add_user` — Add a user (in-memory)
  - `POST /api/v1/admin/run-cli` — Run the CLI for document analysis (admin only)
  - `GET /api/v1/whoami` — Auth/user info
  - `GET /api/v1/logout` — Logout
  - `GET /api/v1/` — Main web UI (if present)
  - `WebSocket /api/v1/ws` — Real-time workflow
- All endpoints require the `/api/v1/` prefix for forward compatibility.
- See `/api/v1/docs` for interactive OpenAPI documentation (if enabled).

## Requirements

- All dependencies are listed in `requirements.txt`.
- Make sure to install with `pip install -r requirements.txt` after any update.
- Notable dependencies: `fastapi`, `uvicorn[standard]`, `websockets`, `python-jose[cryptography]`, `itsdangerous`, `pyjwt`, `python-multipart`, `requests`.
