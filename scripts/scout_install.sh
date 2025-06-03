#!/bin/bash
# Intv_App Scout Installer: Automated local setup for Linux/macOS
# Usage: bash /absolute/path/to/scripts/scout_install.sh
set -e

# Get absolute project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Check for Python 3.10+
PYTHON_BIN=$(command -v python3 || command -v python)
if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.10+ is required. Please install Python and rerun."; exit 1
fi
PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc) -eq 1 ]]; then
  echo "Python 3.10+ is required. Found $PYTHON_VERSION. Please upgrade."; exit 1
fi

# 2. Create virtual environment if missing (absolute path)
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  $PYTHON_BIN -m venv "$PROJECT_ROOT/.venv"
fi

# 3. Activate venv for this shell session
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "ERROR: venv activation script not found at $VENV_ACTIVATE"; exit 1
fi
source "$VENV_ACTIVATE"

# 4. Ensure pip is up to date in venv
pip install --upgrade pip

# 5. Install required packages from requirements.txt
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "ERROR: requirements.txt not found at $REQ_FILE"; exit 1
fi
pip install -r "$REQ_FILE"

# 5b. Install cloudflared if missing
if ! command -v cloudflared &>/dev/null; then
  echo "cloudflared not found. Installing..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update && sudo apt-get install -y cloudflared
  else
    echo "Please install cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
    exit 1
  fi
fi

# 6. Create config files if missing (absolute path)
mkdir -p "$PROJECT_ROOT/config"
[ ! -f "$PROJECT_ROOT/config/config.yaml" ] && cp "$PROJECT_ROOT/config/config_example.yaml" "$PROJECT_ROOT/config/config.yaml" 2>/dev/null || true
[ ! -f "$PROJECT_ROOT/config/users.yaml" ] && cp "$PROJECT_ROOT/config/users_example.yaml" "$PROJECT_ROOT/config/users.yaml" 2>/dev/null || true
[ ! -f "$PROJECT_ROOT/config/policy_prompt.yaml" ] && cp "$PROJECT_ROOT/config/policy_prompt_example.yaml" "$PROJECT_ROOT/config/policy_prompt.yaml" 2>/dev/null || true
[ ! -f "$PROJECT_ROOT/config/vars.json" ] && cp "$PROJECT_ROOT/config/vars_example.json" "$PROJECT_ROOT/config/vars.json" 2>/dev/null || true

# 7. Print next steps
cat <<EOF

✅ Intv_App local environment is ready!

To run locally:
  source "$PROJECT_ROOT/.venv/bin/activate"
  python -m src.main --file yourfile.pdf --type pdf

To run the Web UI:
  ./scripts/cloudflared-entrypoint.sh --gui
  # or add --cloudflare for public tunnel

To stop, press Ctrl+C or close the terminal.

See README.md for more options and troubleshooting.
EOF

echo "\nWould you like to automatically start Intv_App on WSL launch? (y/n)"
read -r AUTO_START
if [ "$AUTO_START" = "y" ] || [ "$AUTO_START" = "Y" ]; then
  BASHRC="$HOME/.bashrc"
  START_LINE="bash $(pwd)/scripts/run_and_info.sh"
  if ! grep -qF "$START_LINE" "$BASHRC"; then
    echo "$START_LINE" >> "$BASHRC"
    echo "Added Intv_App auto-start to ~/.bashrc."
  else
    echo "Intv_App auto-start already present in ~/.bashrc."
  fi
else
  echo "Auto-start on WSL launch not enabled. You can add it later by appending to ~/.bashrc."
fi
