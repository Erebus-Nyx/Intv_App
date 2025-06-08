"""
INTV Server Utilities - Cloudflare tunnel management and service control
"""
import os
import sys
import platform
import subprocess
import time
import psutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def get_cloudflared_binary() -> Optional[str]:
    """Get the path to cloudflared binary"""
    # Check common locations
    possible_paths = [
        "/usr/local/bin/cloudflared",
        "/usr/bin/cloudflared",
        "/opt/homebrew/bin/cloudflared",
        os.path.expanduser("~/.local/bin/cloudflared"),
        "./cloudflared",
        "./bin/cloudflared",
        "cloudflared"  # In PATH
    ]
    
    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    # Try to find in PATH
    try:
        result = subprocess.run(['which', 'cloudflared'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None

def ensure_cloudflared_running(config_path: str = None) -> Dict[str, Any]:
    """Ensure cloudflared is running with the given config"""
    cloudflared_bin = get_cloudflared_binary()
    
    if not cloudflared_bin:
        return {
            'success': False,
            'error': 'cloudflared binary not found',
            'message': 'Install cloudflared from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/'
        }
    
    # Check if already running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'cloudflared' in proc.info['name']:
                return {
                    'success': True,
                    'pid': proc.info['pid'],
                    'message': 'cloudflared already running'
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Start cloudflared
    try:
        if config_path and os.path.exists(config_path):
            cmd = [cloudflared_bin, 'tunnel', '--config', config_path, 'run']
        else:
            # Default config
            cmd = [cloudflared_bin, 'tunnel', '--url', 'localhost:3773', 'run']
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait a moment for startup
        time.sleep(2)
        
        if process.poll() is None:  # Still running
            return {
                'success': True,
                'pid': process.pid,
                'message': 'cloudflared started successfully'
            }
        else:
            stdout, stderr = process.communicate()
            return {
                'success': False,
                'error': f'cloudflared failed to start: {stderr.decode()}',
                'stdout': stdout.decode()
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to start cloudflared: {str(e)}'
        }

def shutdown_services():
    """Shutdown all INTV-related services"""
    services_killed = []
    
    # Kill cloudflared processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'cloudflared' in proc.info['name']:
                proc.terminate()
                services_killed.append(f"cloudflared (PID: {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Kill any INTV GUI processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'intv' in cmdline and ('uvicorn' in cmdline or 'fastapi' in cmdline):
                proc.terminate()
                services_killed.append(f"INTV GUI (PID: {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Wait for graceful termination
    time.sleep(2)
    
    # Force kill if necessary
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'cloudflared' in proc.info['name']:
                proc.kill()
                services_killed.append(f"cloudflared (PID: {proc.info['pid']}) - force killed")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return services_killed

def get_service_status() -> Dict[str, Any]:
    """Get status of all INTV services"""
    status = {
        'cloudflared': False,
        'gui': False,
        'processes': []
    }
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'cloudflared' in proc.info['name']:
                status['cloudflared'] = True
                status['processes'].append({
                    'name': 'cloudflared',
                    'pid': proc.info['pid'],
                    'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                })
            
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'intv' in cmdline and ('uvicorn' in cmdline or 'fastapi' in cmdline):
                status['gui'] = True
                status['processes'].append({
                    'name': 'intv-gui',
                    'pid': proc.info['pid'],
                    'cmdline': cmdline
                })
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return status
