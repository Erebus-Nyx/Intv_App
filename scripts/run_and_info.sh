#!/bin/bash

set -e

# Paths
COMPOSE_FILE="docker/docker-compose.yml"
CLOUDFLARED_ENTRYPOINT="scripts/cloudflared-entrypoint.sh"
APP_PORT=3773

# Ensure entrypoint is executable
chmod +x "$CLOUDFLARED_ENTRYPOINT"

echo "[INFO] Entrypoint script is executable."

# Rebuild and restart containers
docker compose -f "$COMPOSE_FILE" down
docker compose -f "$COMPOSE_FILE" up -d --build

echo "[INFO] Containers rebuilt and started."

# Wait for services to be up
sleep 5

# Print app info
echo "---- App Info ----"
echo "App is listening on: http://localhost:$APP_PORT (inbound: 0.0.0.0:$APP_PORT)"
echo "WebSocket endpoint: ws://localhost:$APP_PORT/api/v1/ws"
echo "API base URL: http://localhost:$APP_PORT/api/v1/docs"
echo "App request endpoint (admin only): POST http://localhost:$APP_PORT/api/v1/admin/run-cli"
echo "Compatibility API: (Check your FastAPI OpenAPI docs for available endpoints)"

# Print Cloudflare info, even if not in logs
CF_CONTAINER=$(docker compose -f "$COMPOSE_FILE" ps -q cloudflared-test)
CF_LOG=$(docker logs "$CF_CONTAINER" 2>&1 | grep -Eo 'https://[a-zA-Z0-9.-]+\\.trycloudflare\\.com' | tail -n1)

if [ -n "$CF_LOG" ]; then
    echo "Cloudflare public URL: $CF_LOG"
    echo "Accessible API: $CF_LOG/api/v1/docs"
    echo "Accessible WebSocket: ${CF_LOG/https:/wss:}/api/v1/ws"
else
    echo "Cloudflare tunnel URL not found in logs."
    echo "If you know your tunnel domain, access: https://<your-tunnel>.trycloudflare.com"
    echo "Accessible API: https://<your-tunnel>.trycloudflare.com/api/v1/docs"
    echo "Accessible WebSocket: wss://<your-tunnel>.trycloudflare.com/api/v1/ws"
fi

echo "-------------------"
