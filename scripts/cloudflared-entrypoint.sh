#!/bin/sh
set -e

CONFIG_FILE="./config/config.cloudflare.yaml"


# Function to extract tunnel value from YAML
# POSIX-compliant shell: use grep/sed/awk, avoid [[ ... ]]
tunnel_name=""
if [ -f "$CONFIG_FILE" ]; then
    tunnel_name=$(grep '^tunnel:' "$CONFIG_FILE" | awk '{print $2}' | xargs)
fi

# Default: do not enable cloudflare unless --cloudflare is present
CLOUDFLARE_ENABLED="false"

# Parse CLI args for --cloudflare
for arg in "$@"; do
    if [ "$arg" = "--cloudflare" ]; then
        CLOUDFLARE_ENABLED="true"
    fi
    if [ "$arg" = "--no-cloudflare" ]; then
        CLOUDFLARE_ENABLED="false"
    fi
    if [ "$arg" = "--cloudflare-env" ]; then
        CLOUDFLARE_ENABLED=$(echo "$USE_CLOUDFLARE_TUNNEL" | tr '[:upper:]' '[:lower:]')
    fi
    # If no cloudflare-related flag is present, leave CLOUDFLARE_ENABLED as false
    # (no action needed, default is off)
done

# Find cloudflared binary (system or local)
CLOUDFLARED_BIN="cloudflared"
CLOUDFLARED_LOCAL="$(dirname "$0")/cloudflared-linux-amd64"
if ! command -v cloudflared >/dev/null 2>&1; then
    if [ -f "$CLOUDFLARED_LOCAL" ]; then
        CLOUDFLARED_BIN="$CLOUDFLARED_LOCAL"
    else
        echo "[ERROR] cloudflared not found. Please ensure it is installed or downloaded." >&2
        exit 1
    fi
fi

# Ensure log directory exists
LOG_DIR="${PROJECT_ROOT:-.}/.logs"
mkdir -p "$LOG_DIR"

if [ "$CLOUDFLARE_ENABLED" = "true" ]; then
    # Set DNS for cloudflared at runtime (Cloudflare 1.1.1.1)
    # export CLOUDLFARED_DNS="1.1.1.1,8.8.8.8"
    # export RESOLV_CONF="/etc/resolv.conf"
    # # Overwrite /etc/resolv.conf for this session (restores on reboot)
    # if [ -w /etc/resolv.conf ]; then
    #     echo -e "nameserver 1.1.1.1\nnameserver 8.8.8.8" > /etc/resolv.conf
    fi
    if [ -n "$tunnel_name" ]; then
        echo "[cloudflared-test] Found tunnel: $tunnel_name. Starting cloudflared with defined tunnel."
        "$CLOUDFLARED_BIN" tunnel run "$tunnel_name" >> "$LOG_DIR/cloudflared.log" 2>&1 &
    else
        # Create a minimal config file to suppress warning
        mkdir -p ./config/cloudflared
        echo "tunnel: free" > ./config/cloudflared/config.yml
        echo "url: http://localhost:3773" >> ./config/cloudflared/config.yml
        echo "[cloudflared-test] No tunnel defined or config missing. Starting free cloudflared tunnel with dummy config..."
        "$CLOUDFLARED_BIN" tunnel --config ./config/cloudflared/config.yml --url http://localhost:3773 >> "$LOG_DIR/cloudflared.log" 2>&1 &
    fi
    sleep 2
fi

# Start the main app (uvicorn or CLI)
if [ "$APP_MODE" = "cli" ]; then
    echo "[Entrypoint] Starting CLI pipeline (python src/main.py)" | tee -a "$LOG_DIR/entrypoint.log"
    exec python src/main.py >> "$LOG_DIR/cli.log" 2>&1
else
    echo "[Entrypoint] Starting FastAPI GUI (uvicorn src.modules.gui.app:app)" | tee -a "$LOG_DIR/entrypoint.log"
    exec uvicorn src.modules.gui.app:app --host 0.0.0.0 --port 3773 --workers 4 --proxy-headers --forwarded-allow-ips=* >> "$LOG_DIR/fastapi.log" 2>&1
fi
