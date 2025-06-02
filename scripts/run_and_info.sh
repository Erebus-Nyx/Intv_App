#!/bin/bash

set -e

# Use env vars or defaults for host/port
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-3773}"

# Start FastAPI app in the background
uvicorn src.modules.gui.app:app --host "$API_HOST" --port "$API_PORT" --workers 4 &
FASTAPI_PID=$!

# Start cloudflared with a random tunnel and capture output
CLOUDFLARED_LOG=/tmp/cloudflared.log
cloudflared tunnel --url http://localhost:$API_PORT 2>&1 | tee $CLOUDFLARED_LOG &
CLOUDFLARED_PID=$!

# Wait for cloudflared to print the public URL, then announce it in the logs
(
  while ! grep -qE 'https://[a-z0-9\-]+\.trycloudflare.com' $CLOUDFLARED_LOG; do sleep 1; done
  LINK=$(grep -Eo 'https://[a-z0-9\-]+\.trycloudflare.com' $CLOUDFLARED_LOG | head -1)
  echo "$LINK" > /tmp/cloudflared_url.txt
  echo "\n==============================="
  echo "Cloudflared public URL: $LINK"
  echo "Local API listening at: http://$API_HOST:$API_PORT/app/v1"
  echo "API docs (Swagger UI): http://$API_HOST:$API_PORT/docs"
  echo "Cloudflared API endpoint: $LINK/app/v1"
  echo "Cloudflared API docs: $LINK/docs"
  echo "===============================\n"
) &

# Wait for FastAPI and cloudflared to exit
wait $FASTAPI_PID
wait $CLOUDFLARED_PID
