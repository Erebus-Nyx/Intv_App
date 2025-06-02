"""
WebSocket-enabled FastAPI GUI for python-cli-app
- Serves a web UI for document/narrative workflow
- Uses WebSockets for real-time communication
- Designed for Docker, NGINX proxy, and Cloudflare tunnel
- Listens on port 3773
"""
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Form, Cookie, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
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

# Secret key for session cookies
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="session")

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

api_v1 = APIRouter(prefix="/api/v1")

@api_v1.get("/", response_class=JSONResponse)
def index():
    return {"status": "ok", "message": "API root. See /docs for documentation."}

@api_v1.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "message": "API is healthy."}

# Add support for loading users from config or environment

def get_users_from_config():
    config = load_config()
    users = {}
    # Default user from config
    default_user = config.get('default_user', {})
    if default_user:
        users[default_user.get('username', 'admin')] = {
            'password': default_user.get('password', 'password123'),
            'role': default_user.get('role', 'admin'),
            'name': default_user.get('name', 'User')
        }
    # Additional users (future multi-user support)
    for user in config.get('users', []):
        users[user['username']] = {
            'password': user['password'],
            'role': user.get('role', 'user'),
            'name': user.get('name', user['username'])
        }
    # ENV override (for admin)
    env_user = os.environ.get('APP_ADMIN_USER')
    env_pass = os.environ.get('APP_ADMIN_PASS')
    env_name = os.environ.get('APP_ADMIN_NAME', env_user or 'Admin')
    if env_user and env_pass:
        users[env_user] = {'password': env_pass, 'role': 'admin', 'name': env_name}
    return users

# Replace dummy USERS with config/env-driven users
USERS = get_users_from_config()

# Add function to add users at runtime (in-memory only)
def add_user(username, password, role='user'):
    USERS[username] = {'password': password, 'role': role}

# Cloudflare Access JWT public keys endpoint
CF_JWT_KEYS_URL = "https://YOUR_DOMAIN.cloudflareaccess.com/cdn-cgi/access/certs"
CF_AUD = os.environ.get("CF_AUD", "")  # Set your Cloudflare Access AUD

# Helper: verify Cloudflare Access JWT
cf_jwt_keys = None

def verify_cf_jwt(token: str) -> Optional[dict]:
    global cf_jwt_keys
    if not cf_jwt_keys:
        try:
            resp = requests.get(CF_JWT_KEYS_URL)
            cf_jwt_keys = resp.json()["keys"]
        except Exception:
            return None
    for key in cf_jwt_keys:
        try:
            payload = jwt.decode(token, key=jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key)), algorithms=[key["alg"]], audience=CF_AUD)
            return payload
        except Exception:
            continue
    return None

# Helper: get current user from session or Cloudflare JWT
async def get_current_user(request: Request) -> Optional[str]:
    # 1. Check session cookie
    user = request.session.get("user")
    if user:
        return user
    # 2. Check Cloudflare Access JWT
    cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get("CF_Authorization")
    if cf_jwt:
        payload = verify_cf_jwt(cf_jwt)
        if payload:
            return payload.get("email") or payload.get("sub")
    return None

# Login endpoint
@api_v1.post("/login")
async def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    user = USERS.get(username)
    if user and user['password'] == password:
        request.session["user"] = username
        request.session["role"] = user.get('role', 'user')
        request.session["name"] = user.get('name', username)
        response = RedirectResponse(url="/api/v1/", status_code=status.HTTP_302_FOUND)
        return response
    return JSONResponse({"error": "Invalid credentials"}, status_code=401)

# Logout endpoint
@api_v1.get("/logout")
async def logout(request: Request, response: Response):
    request.session.clear()
    response = RedirectResponse(url="/api/v1/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("session")
    response.delete_cookie("user")
    return response

# Whoami endpoint
@api_v1.get("/whoami")
async def whoami(user: Optional[str] = Depends(get_current_user)):
    # Allow unauthenticated healthcheck: if no user, return a simple healthy response
    if user:
        user_obj = USERS.get(user)
        return {"user": user, "name": user_obj.get("name", user), "role": user_obj.get("role", "user")}
    # For healthcheck, return 200 with minimal info
    return {"status": "ok", "user": None}

# WebSocket endpoint for real-time workflow
@api_v1.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Authenticate: check session or Cloudflare JWT
    session_cookie = websocket.cookies.get("session")
    cf_jwt = websocket.headers.get("Cf-Access-Jwt-Assertion") or websocket.cookies.get("CF_Authorization")
    user = None
    if session_cookie:
        user = websocket.cookies.get("user")
    elif cf_jwt:
        payload = verify_cf_jwt(cf_jwt)
        if payload:
            user = payload.get("email") or payload.get("sub")
    if not user:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            if req.get("type") == "start_workflow":
                modules = [
                    "intv_adult", "intv_child", "intv_ar", "intv_col",
                    "homeassess", "allegations", "close_dispo", "close_ea", "close_staffing"
                ]
                await websocket.send_json({"type": "modules", "modules": modules})
            elif req.get("type") == "run_module":
                module = req.get("module")
                input_data = req.get("input", {})
                use_defaults = req.get("defaults", False)
                await websocket.send_json({"type": "progress", "msg": f"Running {module}...", "percent": 5})
                try:
                    mod = importlib.import_module(f"src.modules.{module}")
                    output_func = getattr(mod, f"{module}_output")
                    # If use_defaults, load config and pass all defaults as input
                    if use_defaults:
                        config_path = Path(__file__).parent.parent.parent / "config" / f"{module}_vars.json"
                        with open(config_path, "r", encoding="utf-8") as f:
                            logic_tree = json.load(f)
                        input_data = {var: meta.get("default", "") for var, meta in logic_tree.items()}
                    # Simulate progress
                    await websocket.send_json({"type": "progress", "msg": "Processing input...", "percent": 30})
                    result = output_func(lookup_id=None, output_path=None, **{f"{module.split('_')[0]}_data": input_data})
                    await websocket.send_json({"type": "progress", "msg": "Finalizing...", "percent": 90})
                    await websocket.send_json({
                        "type": "result",
                        "narrative": result.get("narrative", ""),
                        "clarification_needed": result.get("clarification_needed", False),
                        "pending_questions": result.get("pending_questions", []),
                        "is_final": result.get("is_final", False)
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "msg": f"Error running module: {str(e)}"})
            elif req.get("type") == "recording":
                # Simulate recording indicator
                await websocket.send_json({"type": "recording", "active": req.get("active", False)})
            else:
                await websocket.send_json({"type": "error", "msg": "Unknown request type."})
    except WebSocketDisconnect:
        pass

# API to add users (for testing, in-memory only)
@api_v1.post("/add_user")
async def add_user_api(username: str = Form(...), password: str = Form(...), role: str = Form('user'), name: str = Form(None)):
    add_user(username, password, role)
    if name:
        USERS[username]['name'] = name
    return {"status": "success", "user": username, "role": role, "name": USERS[username].get('name', username)}

security = HTTPBasic()

# Restrict CLI emulation endpoint to admin users only
def is_admin(user: Optional[str] = Depends(get_current_user)):
    if not user or USERS.get(user, {}).get('role') != 'admin':
        raise JSONResponse({"error": "Admin access required"}, status_code=403)
    return user

# Allowed file types and safe directory
ALLOWED_TYPES = {"pdf", "docx", "txt", "mp3", "wav", "mp4", "m4a"}
SAFE_INPUT_DIR = Path("/app/config/uploads")
SAFE_OUTPUT_DIR = Path("/app/config/outputs")

@api_v1.post("/admin/run-cli")
async def run_cli(
    file: str = Form(...),
    type: str = Form(...),
    extra_args: str = Form(""),
    user: str = Depends(is_admin)
):
    # Only allow files in SAFE_INPUT_DIR
    file_path = SAFE_INPUT_DIR / file
    if not file_path.exists() or not file_path.is_file():
        return JSONResponse({"error": "File not found or not allowed."}, status_code=404)
    if type not in ALLOWED_TYPES:
        return JSONResponse({"error": "Type not allowed."}, status_code=400)
    # Build CLI command
    cmd = ["python3", "-m", "src.main", "--file", str(file_path), "--type", type]
    if extra_args:
        cmd += extra_args.split()
    # Run CLI and capture output
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Optionally restrict output file access here
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# New endpoint to expose Cloudflare public URL
@api_v1.get("/cloudflared_url", response_class=JSONResponse)
def cloudflared_url():
    url_path = "/tmp/cloudflared_url.txt"
    if os.path.exists(url_path):
        with open(url_path, "r") as f:
            url = f.read().strip()
        if url:
            return {"cloudflared_url": url}
    return {"cloudflared_url": None, "error": "URL not available yet"}

app.include_router(api_v1)

@app.get("/", include_in_schema=False)
def root():
    # Serve the GUI static index.html
    static_index = static_dir / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index), media_type="text/html")
    return RedirectResponse(url="/api/v1/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3773))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
