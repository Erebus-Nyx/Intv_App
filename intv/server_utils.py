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
        subprocess.Popen(['uvicorn', 'intv.api:app', '--port', str(port)])
        print(f"[server_utils] Started FastAPI on port {port}")
    else:
        print(f"[server_utils] FastAPI already running on port {port}")

def get_cloudflared_binary():
    """Find cloudflared binary, prefer system installation over local binary."""
    import shutil
    from pathlib import Path
    
    # Check if cloudflared is in PATH
    if shutil.which('cloudflared'):
        return 'cloudflared'
    
    # Check for local binary in scripts directory
    script_dir = Path(__file__).parent.parent.parent / 'scripts'
    local_binary = script_dir / 'cloudflared-linux-amd64'
    if local_binary.exists() and local_binary.is_file():
        return str(local_binary)
    
    raise FileNotFoundError("cloudflared binary not found. Please install cloudflared or ensure scripts/cloudflared-linux-amd64 exists.")

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
    
    try:
        cloudflared_bin = get_cloudflared_binary()
        print(f"[server_utils] Starting cloudflared tunnel for http://localhost:{port} ...")
        log_path = f"/tmp/cloudflared_{port}.log"
        proc = subprocess.Popen([
            cloudflared_bin, 'tunnel', '--url', f'http://localhost:{port}'
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
    except FileNotFoundError as e:
        print(f"[server_utils] ERROR: {e}")
        print("[server_utils] Please install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/")

def shutdown_services():
    """
    Gracefully shutdown FastAPI and cloudflared services.
    - Kills cloudflared, uvicorn, and relevant INTV python processes (psutil or pkill fallback).
    - Prints status messages for user clarity.
    """
    print("[server_utils] Shutting down FastAPI and cloudflared services...")
    killed = 0
    def kill_by_name(name, selective=False):
        nonlocal killed
        if psutil:
            for proc in psutil.process_iter(['name', 'cmdline', 'status']):
                try:
                    if name in proc.info['name'] or (proc.info['cmdline'] and any(name in c for c in proc.info['cmdline'])):
                        # For python processes, be more selective - only kill INTV related processes
                        if selective and proc.info['cmdline']:
                            cmdline_str = ' '.join(proc.info['cmdline'])
                            if not any(keyword in cmdline_str for keyword in ['intv', 'uvicorn', 'fastapi']):
                                continue
                        print(f"[server_utils] Killing {name} (PID {proc.pid})")
                        proc.kill()
                        killed += 1
                except Exception:
                    continue
        else:
            if selective:
                subprocess.call(['pkill', '-f', 'intv'])
            else:
                subprocess.call(['pkill', '-f', name])
            killed += 1
    kill_by_name('cloudflared')
    kill_by_name('uvicorn')
    kill_by_name('python', selective=True)  # Only kill INTV related python processes
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

# In all relevant functions, use config-driven settings for LLM, RAG, output_format, VAD, diarization, etc., falling back to config.yaml defaults if not present.
