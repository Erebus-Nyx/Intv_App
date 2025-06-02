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
import yaml
import threading

# Secret key for session cookies
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
app = FastAPI()

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

api_v1 = APIRouter(prefix="/api/v1")

@api_v1.get("/", response_class=JSONResponse)
def api_v1_root():
    return {"status": "ok", "message": "API v1 root. See /api/v1/docs for documentation."}

@api_v1.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "message": "API is healthy."}

# Remove all user/role/multi-user logic
CURRENT_ADMIN = False

@api_v1.post("/admin_login")
async def admin_login():
    global CURRENT_ADMIN
    CURRENT_ADMIN = True
    return {"status": "ok", "admin": True}

@api_v1.post("/admin_logout")
async def admin_logout():
    global CURRENT_ADMIN
    CURRENT_ADMIN = False
    return {"status": "ok", "admin": False}

@api_v1.get("/whoami")
async def whoami():
    global CURRENT_ADMIN
    return {"admin": CURRENT_ADMIN}

# Update is_admin to check CURRENT_ADMIN
def is_admin():
    global CURRENT_ADMIN
    if not CURRENT_ADMIN:
        raise JSONResponse({"error": "Admin access required"}, status_code=403)
    return True

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

# Helper to check if admin is using default password
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin"
USERS_YAML_PATH = Path("/app/config/users.yaml")

# Helper to update users.yaml
users_yaml_lock = threading.Lock()
def update_user_password_in_yaml(username, new_password):
    with users_yaml_lock:
        if USERS_YAML_PATH.exists():
            with open(USERS_YAML_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            user_list = data.get('users', [])
            for user in user_list:
                if user['username'] == username:
                    user['password'] = new_password
            data['users'] = user_list
            with open(USERS_YAML_PATH, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True)

# Remove/disable login and session logic
# Add a global CURRENT_USER variable
CURRENT_USER = None

@api_v1.post("/set_user")
async def set_user(name: str = Form(...)):
    global CURRENT_USER
    CURRENT_USER = name.strip()
    return {"status": "ok", "user": CURRENT_USER}

@api_v1.get("/whoami")
async def whoami():
    global CURRENT_USER
    return {"user": CURRENT_USER}

# Update is_admin to check CURRENT_USER
def is_admin():
    global CURRENT_USER
    if CURRENT_USER != "admin":
        raise JSONResponse({"error": "Admin access required"}, status_code=403)
    return CURRENT_USER

# Only admin endpoints use is_admin()

security = HTTPBasic()

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
    # Serve the GUI static index.html as the default landing page
    static_index = static_dir / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index), media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3773))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
