#!/bin/bash

set -e

# Move to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Check and free up API port if in use (robust, cross-platform)
API_PORT=3773
API_HOST="${API_HOST:-0.0.0.0}"

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

# Parse CLI arguments
if [[ "$1" == "--exit" ]]; then
  echo "Stopping all FastAPI and cloudflared processes on ports 3773 and 3774..."
  for P in 3773 3774; do
    PIDS=$(lsof -t -i :$P 2>/dev/null || netstat -nlp 2>/dev/null | grep :$P | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$PIDS" ]; then
      for PID in $PIDS; do
        if [ -n "$PID" ]; then
          echo "Killing PID $PID on port $P..."
          kill -9 $PID 2>/dev/null || true
        fi
      done
    fi
  done
  # Also kill any background uvicorn or cloudflared started by this script (using PID files)
  for PIDFILE in /tmp/intvapp_fastapi_*.pid /tmp/intvapp_cloudflared_*.pid; do
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      if kill -0 $PID 2>/dev/null; then
        echo "Killing process $PID from $PIDFILE..."
        kill -9 $PID 2>/dev/null || true
      fi
      rm -f "$PIDFILE"
    fi
  done
  echo "All relevant processes stopped."
  exit 0
fi

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
CLOUDFLARED_LOCAL="$SCRIPT_DIR/cloudflared-linux-amd64"

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

# Start FastAPI app in the background (only one instance)
uvicorn src.modules.gui.app:app --host "$API_HOST" --port "$API_PORT" --workers 4 &
FASTAPI_PID=$!
echo $FASTAPI_PID > /tmp/intvapp_fastapi_${API_PORT}.pid

# Start cloudflared with a random tunnel and capture output (only one instance)
CLOUDFLARED_LOG=/tmp/cloudflared_${API_PORT}.log
"$CLOUDFLARED_BIN" tunnel --url http://localhost:$API_PORT 2>&1 | tee $CLOUDFLARED_LOG &
CLOUDFLARED_PID=$!
echo $CLOUDFLARED_PID > /tmp/intvapp_cloudflared_${API_PORT}.pid

# Wait for cloudflared to print the public URL, then announce it in the logs
(
  while ! grep -qE 'https://[a-z0-9\-]+\.trycloudflare.com' $CLOUDFLARED_LOG; do sleep 1; done
  LINK=$(grep -Eo 'https://[a-z0-9\-]+\.trycloudflare.com' $CLOUDFLARED_LOG | head -1)
  echo "$LINK" > /tmp/cloudflared_url_${API_PORT}.txt
  echo "\n==============================="
  echo "Cloudflared public URL: $LINK"
  echo "Local API listening at: http://$API_HOST:$API_PORT/app/v1"
  echo "API docs (Swagger UI): http://$API_HOST:$API_PORT/docs"
  echo "Cloudflared API endpoint: $LINK/app/v1"
  echo "Cloudflared API docs: $LINK/docs"
  echo "===============================\n"
) &

# Script exits here, leaving FastAPI and cloudflared running in the background
exit 0
