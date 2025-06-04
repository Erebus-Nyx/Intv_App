#!/bin/bash
# intv_startup.sh: Unified installer and runner for Intv_App (Linux/WSL/macOS)
set -e

# Fast path: handle --exit before any install/startup logic
if [[ "$1" == "--exit" ]]; then
  echo "[INFO] Stopping all FastAPI and cloudflared processes on ports 3773 and 3774..."
  for P in 3773 3774; do
    for PIDFILE in /tmp/intvapp_fastapi_${P}.pid /tmp/intvapp_cloudflared_${P}.pid; do
      if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 $PID 2>/dev/null; then
          echo "Killing process $PID from $PIDFILE..."
          kill -9 $PID 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
      fi
    done
  done
  # Fallback: kill by process name if any remain
  pkill -f uvicorn || true
  pkill -f cloudflared || true
  echo "[INFO] All relevant processes stopped."
  exit 0
fi

# Move to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CACHE_DIR="$PROJECT_ROOT/.cache"
INSTALL_MARKER="$CACHE_DIR/installed.ok"
mkdir -p "$CACHE_DIR"

# Check for --refresh to force dependency check
if [[ "$1" == "--refresh" ]]; then
  echo "[INFO] Forcing dependency check (--refresh specified)."
  rm -f "$INSTALL_MARKER"
fi

# If install marker exists, skip dependency installation
if [ -f "$INSTALL_MARKER" ]; then
  echo "[INFO] Dependencies previously installed. Skipping install steps. Use --refresh to force."
else
  # 1. Check for Python 3.10+
  PYTHON_BIN=$(command -v python3 || command -v python)
  if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.10+ is required. Please install Python and rerun."; exit 1
  fi
  PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
  if [[ $(echo "$PYTHON_VERSION < 3.10" | bc) -eq 1 ]]; then
    echo "Python 3.10+ is required. Found $PYTHON_VERSION. Please upgrade."; exit 1
  fi

  # 2. Create virtual environment if missing
  if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    $PYTHON_BIN -m venv "$PROJECT_ROOT/.venv"
  fi

  # 3. Activate venv
  VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
  if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: venv activation script not found at $VENV_ACTIVATE"; exit 1
  fi
  source "$VENV_ACTIVATE"

  # 4. Ensure pip is up to date
  pip install --upgrade pip



  # 6. Install required system packages (Linux/apt)
  REQUIRED_PKGS=(python3 python3-venv python3-pip tesseract-ocr poppler-utils python3-tk libtesseract-dev libleptonica-dev cups)
  MISSING_PKGS=()
  for pkg in "${REQUIRED_PKGS[@]}"; do
      if ! dpkg -s "$pkg" &>/dev/null; then
          MISSING_PKGS+=("$pkg")
      fi
  done
  if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
      echo "[INFO] Installing missing system dependencies: ${MISSING_PKGS[*]}"
      sudo apt-get update && sudo apt-get install -y "${MISSING_PKGS[@]}"
  fi

  # 7. Ensure config files exist
  mkdir -p "$PROJECT_ROOT/config"
  [ ! -f "$PROJECT_ROOT/config/config.yaml" ] && cp "$PROJECT_ROOT/config/config_example.yaml" "$PROJECT_ROOT/config/config.yaml" 2>/dev/null || true
  [ ! -f "$PROJECT_ROOT/config/users.yaml" ] && cp "$PROJECT_ROOT/config/users_example.yaml" "$PROJECT_ROOT/config/users.yaml" 2>/dev/null || true
  [ ! -f "$PROJECT_ROOT/config/policy_prompt.yaml" ] && cp "$PROJECT_ROOT/config/policy_prompt_example.yaml" "$PROJECT_ROOT/config/policy_prompt.yaml" 2>/dev/null || true
  [ ! -f "$PROJECT_ROOT/config/vars.json" ] && cp "$PROJECT_ROOT/config/vars_example.json" "$PROJECT_ROOT/config/vars.json" 2>/dev/null || true

  # Mark install as complete
  touch "$INSTALL_MARKER"
  echo "[INFO] Install complete. Marker written to $INSTALL_MARKER."
fi

# Always ensure the latest package is installed in editable mode
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
  VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
  if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
  fi
  pip install -e .
fi

# Always activate venv before running backend
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
fi

# Always run backend in background, no frontend/GUI option

# Set default ports
API_PORT=3773
API_HOST=0.0.0.0

# Function to check and free port
free_port() {
  local PORT=$1
  PORT_PIDS=$(lsof -t -i :$PORT 2>/dev/null || netstat -nlp 2>/dev/null | grep :$PORT | awk '{print $7}' | cut -d'/' -f1)
  if [ -n "$PORT_PIDS" ]; then
    echo "Port $PORT is in use by PIDs: $PORT_PIDS. Stopping them..."
    for PID in $PORT_PIDS; do
      if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null || true
      fi
    done
    sleep 1
  fi
  # Double-check port is free
  PORT_PIDS2=$(lsof -t -i :$PORT 2>/dev/null || netstat -nlp 2>/dev/null | grep :$PORT | awk '{print $7}' | cut -d'/' -f1)
  if [ -n "$PORT_PIDS2" ]; then
    return 1
  fi
  return 0
}

# Try to free 3773, else use 3774
if free_port 3773; then
  API_PORT=3773
  echo "Using port 3773."
else
  echo "Port 3773 unavailable, trying 3774..."
  if free_port 3774; then
    API_PORT=3774
    echo "Using port 3774."
  else
    echo "Neither port 3773 nor 3774 is available. Exiting."
    exit 1
  fi
fi

# Check for cloudflared, download if missing
CLOUDFLARED_BIN="cloudflared"
CLOUDFLARED_LOCAL="$(dirname "$0")/cloudflared-linux-amd64"
if command -v cloudflared &>/dev/null; then
  CLOUDFLARED_BIN="cloudflared"
else
  if [ ! -f "$CLOUDFLARED_LOCAL" ]; then
    echo "cloudflared not found in PATH. Downloading latest Linux binary to $CLOUDFLARED_LOCAL..."
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$CLOUDFLARED_LOCAL"
    chmod +x "$CLOUDFLARED_LOCAL"
  fi
  CLOUDFLARED_BIN="$CLOUDFLARED_LOCAL"
fi

# Start FastAPI backend in the background and save PID
nohup uvicorn src.modules.gui.app:app --host "$API_HOST" --port "$API_PORT" --workers 4 > .logs/fastapi_${API_PORT}.log 2>&1 &
echo $! > /tmp/intvapp_fastapi_${API_PORT}.pid

# Start cloudflared in the background and save PID
nohup "$CLOUDFLARED_BIN" tunnel --url http://localhost:$API_PORT > .logs/cloudflared_${API_PORT}.log 2>&1 &
echo $! > /tmp/intvapp_cloudflared_${API_PORT}.pid

# Print info and exit
sleep 2
echo "==============================="
echo "FastAPI and cloudflared started in background."
echo "Check .logs/fastapi_${API_PORT}.log and .logs/cloudflared_${API_PORT}.log for output."
echo "-------------------------------"
echo "Local API:   http://localhost:${API_PORT}/api/v1/"
echo "Swagger UI:  http://localhost:${API_PORT}/api"
# Print Cloudflare public URL if available
tunnel_url=$(grep -Eo 'https://[a-zA-Z0-9\-]+\.trycloudflare.com' .logs/cloudflared_${API_PORT}.log | tail -1)
if [ -n "$tunnel_url" ]; then
  echo "Cloudflare Tunnel: $tunnel_url"
  echo "Tunnel API:        $tunnel_url/api/v1/"
  echo "Tunnel Swagger UI: $tunnel_url/api"
else
  echo "Cloudflare Tunnel: (not available yet, check .logs/cloudflared_${API_PORT}.log)"
fi
echo "==============================="
exit 0
