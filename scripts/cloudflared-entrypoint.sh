#!/bin/bash
set -e

CONFIG_FILE="/mnt/g/WSL/nyx/workspace/python-cli-app/config/config.cloudflare.yaml"
GUI_OUTPUT="/mnt/g/WSL/nyx/web-apps/interview summary.json"

# Function to extract tunnel value from YAML
tunnel_name=$(grep '^tunnel:' "$CONFIG_FILE" | awk '{print $2}' | xargs)

if [[ -n "$tunnel_name" ]]; then
    echo "[cloudflared-test] Found tunnel: $tunnel_name. Starting cloudflared with defined tunnel."
    cloudflared tunnel run "$tunnel_name"
else
    echo "[cloudflared-test] No tunnel defined. Starting free cloudflared tunnel..."
    # Start a free tunnel and capture the output
    output=$(cloudflared tunnel --url http://localhost:8080 2>&1 | tee /tmp/cloudflared.log)
    # Extract the assigned public URL from the output
    assigned_url=$(echo "$output" | grep -Eo 'https://[a-z0-9\-]+\.trycloudflare.com')
    if [[ -n "$assigned_url" ]]; then
        echo "[cloudflared-test] Assigned public URL: $assigned_url"
        # Output to CLI
        echo "Cloudflared public URL: $assigned_url"
        # Output to GUI (write to a JSON file for GUI to read)
        echo "{\"cloudflared_url\": \"$assigned_url\"}" > "$GUI_OUTPUT"
    else
        echo "[cloudflared-test] Failed to obtain public URL from cloudflared output."
        exit 1
    fi
fi
