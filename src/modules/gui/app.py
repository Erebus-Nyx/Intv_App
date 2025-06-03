"""
WebSocket-enabled FastAPI GUI for Intv_App
- Serves a web UI for document/narrative workflow
- Uses WebSockets for real-time communication
- Designed for Docker, NGINX proxy, and Cloudflare tunnel
- Listens on port 3773
"""
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Form, Cookie, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
import json
from pathlib import Path
import importlib
import secrets
import jwt
import base64
import requests
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from typing import Optional
from src.config import load_config
from fastapi import Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import shutil
from fastapi import APIRouter
import yaml
import threading
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse as StarlettePlainTextResponse
from fastapi import BackgroundTasks
import platform
from src.main import get_available_interview_types

# Secret key for session cookies
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
app = FastAPI(docs_url="/api", redoc_url=None, openapi_url="/api/openapi.json")

# Set session cookie policy based on environment
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
if SESSION_COOKIE_SECURE:
    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="session", same_site="none", https_only=True)
else:
    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="session", same_site="lax", https_only=False)

# Allow CORS for Cloudflare tunnel and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Main app page at /
@app.get("/", include_in_schema=False)
def root():
    static_index = static_dir / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index), media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)

# --- Routers ---
from fastapi import APIRouter

api_v1 = APIRouter(prefix="/api/v1")
api_admin = APIRouter(prefix="/api/admin")
api_data = APIRouter(prefix="/api/data")

# Admin state and helpers
CURRENT_ADMIN = False

# Allowed file types and safe directory
ALLOWED_TYPES = {"pdf", "docx", "txt", "mp3", "wav", "mp4", "m4a"}
SAFE_INPUT_DIR = Path("/app/config/uploads")
SAFE_OUTPUT_DIR = Path("/app/config/outputs")

# Accept PDF files for transcription as well as document analysis
TRANSCRIBE_EXTENSIONS = {"pdf", "mp3", "wav", "mp4", "m4a"}

def is_admin():
    global CURRENT_ADMIN
    if not CURRENT_ADMIN:
        raise JSONResponse({"error": "Admin access required"}, status_code=403)
    return True

# Core API endpoints (United compatible)
@api_v1.get("/", response_class=JSONResponse)
def api_v1_root():
    return {"status": "ok", "message": "API v1 root. See /api for docs."}

@api_v1.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "message": "API is healthy."}

# Example: whoami endpoint (could be moved to admin if needed)
@api_v1.get("/whoami")
def whoami():
    global CURRENT_ADMIN
    return {"admin": CURRENT_ADMIN}

# Admin endpoints
@api_admin.post("/login")
async def admin_login():
    global CURRENT_ADMIN
    CURRENT_ADMIN = True
    return {"status": "ok", "admin": True}

@api_admin.post("/logout")
async def admin_logout():
    global CURRENT_ADMIN
    CURRENT_ADMIN = False
    return {"status": "ok", "admin": False}

# Data recall endpoints (stub)
@api_data.get("/example")
def data_example():
    return {"data": "This is a data recall endpoint."}

# Mount routers
app.include_router(api_v1)
app.include_router(api_admin)
app.include_router(api_data)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Dynamically get modules
    modules = [t['key'] for t in get_available_interview_types()]
    await websocket.send_text(json.dumps({"type": "modules", "modules": modules}))
    try:
        while True:
            data = await websocket.receive_text()
            # You can implement your workflow logic here, or forward to a handler
            # For now, just echo the message
            await websocket.send_text(data)
    except WebSocketDisconnect:
        pass

class WebSocketOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/ws" and request.headers.get("upgrade", "").lower() != "websocket":
            return StarlettePlainTextResponse("WebSocket endpoint. Use a WebSocket client to connect.", status_code=400)
        return await call_next(request)

app.add_middleware(WebSocketOnlyMiddleware)

# In your /api/generate or relevant endpoint, add logic to treat PDF as transcribable
@app.post("/api/generate")
def api_generate(background_tasks: BackgroundTasks):
    import subprocess
    import sys
    import os
    # Choose script based on OS
    if platform.system() == "Windows":
        script = os.path.join("scripts", "run_and_info_win.bat")
        cmd = ["cmd.exe", "/c", script]
    else:
        script = os.path.join("scripts", "run_and_info.sh")
        cmd = ["bash", script]
    # You can add logic here to check file extension and route PDF to transcription if needed
    try:
        # Run in background, return immediately
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"status": "started", "pid": process.pid, "script": script}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3773))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
