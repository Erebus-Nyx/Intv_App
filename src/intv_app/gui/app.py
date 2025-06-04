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
from config import load_config
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
from intv_app.main import get_available_interview_types

SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
app = FastAPI(docs_url="/api", redoc_url=None, openapi_url="/api/openapi.json")

SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
if SESSION_COOKIE_SECURE:
    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="session", same_site="none", https_only=True)
else:
    app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="session", same_site="lax", https_only=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
def root():
    static_index = static_dir / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index), media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)

from fastapi import APIRouter

api_v1 = APIRouter(prefix="/api/v1")
api_admin = APIRouter(prefix="/api/admin")
api_data = APIRouter(prefix="/api/data")

CURRENT_ADMIN = False
ALLOWED_TYPES = {"pdf", "docx", "txt", "mp3", "wav", "mp4", "m4a"}
SAFE_INPUT_DIR = Path("/app/config/uploads")
SAFE_OUTPUT_DIR = Path("/app/config/outputs")
TRANSCRIBE_EXTENSIONS = {"pdf", "mp3", "wav", "mp4", "m4a"}

def is_admin():
    global CURRENT_ADMIN
    if not CURRENT_ADMIN:
        raise JSONResponse({"error": "Admin access required"}, status_code=403)
    return True

@api_v1.get("/", response_class=JSONResponse)
def api_v1_root():
    return {"status": "ok", "message": "API v1 root. See /api for docs."}

@api_v1.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "message": "API is healthy."}

@api_v1.get("/whoami")
def whoami():
    global CURRENT_ADMIN
    return {"admin": CURRENT_ADMIN}

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

@api_data.get("/example")
def data_example():
    return {"data": "This is a data recall endpoint."}

app.include_router(api_v1)
app.include_router(api_admin)
app.include_router(api_data)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    modules = [t['key'] for t in get_available_interview_types()]
    await websocket.send_text(json.dumps({"type": "modules", "modules": modules}))
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        pass

class WebSocketOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/ws" and request.headers.get("upgrade", "").lower() != "websocket":
            return StarlettePlainTextResponse("WebSocket endpoint. Use a WebSocket client to connect.", status_code=400)
        return await call_next(request)

app.add_middleware(WebSocketOnlyMiddleware)

@app.post("/api/generate")
def api_generate(background_tasks: BackgroundTasks):
    import subprocess
    import sys
    import os
    if platform.system() == "Windows":
        script = os.path.join("scripts", "run_and_info_win.bat")
        cmd = ["cmd.exe", "/c", script]
    else:
        script = os.path.join("scripts", "run_and_info.sh")
        cmd = ["bash", script]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"status": "started", "pid": process.pid, "script": script}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

from fastapi.responses import RedirectResponse

@app.get("/api", include_in_schema=False)
async def api_root_redirect(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/")
    return JSONResponse({"message": "API root. See /api/v1 for endpoints."})

@app.get("/api/v1", include_in_schema=False)
async def api_v1_redirect(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/")
    return JSONResponse({"message": "API v1 root. See /api/v1/health for status."})

@app.get("/docs", include_in_schema=False)
async def docs_redirect(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/")
    return JSONResponse({"message": "Docs are available at /api (Swagger UI)."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3773))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
