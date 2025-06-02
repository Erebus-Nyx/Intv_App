#!/bin/sh
set -e

CONFIG_FILE="./config/config.cloudflare.yaml"
GUI_OUTPUT="./web-apps/interview summary.json"

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
done

if [ "$CLOUDFLARE_ENABLED" = "true" ]; then
    if [ -n "$tunnel_name" ]; then
        echo "[cloudflared-test] Found tunnel: $tunnel_name. Starting cloudflared with defined tunnel."
        cloudflared tunnel run "$tunnel_name" &
    else
        # Create a minimal config file to suppress warning
        mkdir -p ./config/cloudflared
        echo "tunnel: free" > ./config/cloudflared/config.yml
        echo "url: http://localhost:3773" >> ./config/cloudflared/config.yml
        echo "[cloudflared-test] No tunnel defined or config missing. Starting free cloudflared tunnel with dummy config..."
        cloudflared tunnel --config ./config/cloudflared/config.yml --url http://localhost:3773 2>&1 | tee ./web-apps/cloudflared.log &
    fi
    sleep 2
fi

# Start the main app (uvicorn)
exec uvicorn src.modules.gui.app:app --host 0.0.0.0 --port 3773 --workers 4
