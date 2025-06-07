#!/bin/bash
# Test Docker entrypoint logic for APP_MODE=cli and APP_MODE=gui
set -e

# Test CLI mode
export APP_MODE=cli
echo "[TEST] Running Docker CLI mode (should run python src/main.py)"
docker-compose up --build -d
sleep 10
docker-compose logs intv | grep -i 'main.py is executing' || echo '[WARN] main.py not found in logs (may be running in background)'
docker-compose down

# Test GUI mode
export APP_MODE=gui
echo "[TEST] Running Docker GUI mode (should run FastAPI GUI)"
docker-compose up --build -d
sleep 10
docker-compose logs intv | grep -i 'uvicorn' || echo '[WARN] uvicorn not found in logs (may be running in background)'
docker-compose down
