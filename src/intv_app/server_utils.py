import subprocess
import socket
import os
import time

try:
    import psutil
except ImportError:
    psutil = None

def is_port_in_use(port):
    """
    Check if a port is in use on localhost.
    Returns True if the port is open, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def ensure_fastapi_running(port=3773):
    """
    Ensure FastAPI server is running on the given port. Start if not running.
    Launches uvicorn if needed.
    """
    if not is_port_in_use(port):
        subprocess.Popen(['uvicorn', 'intv_app.api:app', '--port', str(port)])
        print(f"[server_utils] Started FastAPI on port {port}")
    else:
        print(f"[server_utils] FastAPI already running on port {port}")

def ensure_cloudflared_running(port=3773, wait_for_url=True, progress=True):
    """
    Ensure cloudflared tunnel is running. Start if not running.
    - Checks for running cloudflared process (psutil or pgrep fallback).
    - If not running, starts cloudflared for the given port.
    - Optionally waits for the public URL to appear in logs (progress bar).
    """
    def is_cloudflared_running():
        if psutil:
            for proc in psutil.process_iter(['name', 'cmdline', 'status']):
                try:
                    if (
                        'cloudflared' in proc.info['name'] or
                        (proc.info['cmdline'] and any('cloudflared' in c for c in proc.info['cmdline']))
                    ) and proc.info.get('status', '').lower() not in ('zombie', 'defunct'):
                        return True
                except Exception:
                    continue
            return False
        else:
            # Fallback: use pgrep if psutil is not available
            return subprocess.call(['pgrep', '-f', 'cloudflared'], stdout=subprocess.DEVNULL) == 0

    if is_cloudflared_running():
        print("[server_utils] Cloudflared tunnel already running.")
        return
    print(f"[server_utils] Starting cloudflared tunnel for http://localhost:{port} ...")
    log_path = f"/tmp/cloudflared_{port}.log"
    proc = subprocess.Popen([
        'cloudflared', 'tunnel', '--url', f'http://localhost:{port}'
    ], stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    if progress:
        print("[server_utils] Waiting for cloudflared public URL...", end="", flush=True)
        for i in range(30):
            if os.path.exists(log_path):
                with open(log_path) as f:
                    for line in f:
                        if "trycloudflare.com" in line:
                            print(" done.")
                            print(f"[server_utils] {line.strip()}")
                            return
            print(".", end="", flush=True)
            time.sleep(1)
        print("\n[server_utils] Warning: Cloudflared public URL not detected after 30s. Check logs.")
    else:
        print("[server_utils] Cloudflared started (no progress bar).")

def shutdown_services():
    """
    Gracefully shutdown FastAPI and cloudflared services.
    - Kills cloudflared, uvicorn, and FastAPI python processes (psutil or pkill fallback).
    - Prints status messages for user clarity.
    """
    print("[server_utils] Shutting down FastAPI and cloudflared services...")
    killed = 0
    def kill_by_name(name):
        nonlocal killed
        if psutil:
            for proc in psutil.process_iter(['name', 'cmdline', 'status']):
                try:
                    if name in proc.info['name'] or (proc.info['cmdline'] and any(name in c for c in proc.info['cmdline'])):
                        print(f"[server_utils] Killing {name} (PID {proc.pid})")
                        proc.kill()
                        killed += 1
                except Exception:
                    continue
        else:
            subprocess.call(['pkill', '-f', name])
            killed += 1
    kill_by_name('cloudflared')
    kill_by_name('uvicorn')
    kill_by_name('python')  # Only if running FastAPI as python process
    print(f"[server_utils] Shutdown complete. {killed} processes killed.")

def check_health_endpoint(url='http://localhost:3773/health'):
    """Ping a health endpoint to verify FastAPI readiness."""
    try:
        import requests
        resp = requests.get(url, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False

def setup_logging(log_file='server.log'):
    """Centralized logging setup for server events."""
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger('server_utils')
