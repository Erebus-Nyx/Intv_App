import os
import time
import pytest
from intv_app import server_utils

def test_is_port_in_use_unused():
    # Pick a high port unlikely to be in use
    assert not server_utils.is_port_in_use(54321)

def test_is_port_in_use_used(monkeypatch):
    import socket
    s = socket.socket()
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    try:
        assert server_utils.is_port_in_use(port)
    finally:
        s.close()

def test_ensure_cloudflared_running(monkeypatch):
    # Simulate cloudflared not running, then running
    called = {}
    def fake_is_cloudflared_running():
        if not called.get('ran'):
            called['ran'] = True
            return False
        return True
    monkeypatch.setattr(server_utils, 'psutil', None)  # fallback to pgrep
    monkeypatch.setattr(server_utils, 'subprocess', server_utils.subprocess)
    monkeypatch.setattr(server_utils, 'is_port_in_use', lambda port: True)
    # Should not raise
    server_utils.ensure_cloudflared_running(progress=False)

def test_shutdown_services(monkeypatch):
    killed = []
    def fake_kill_by_name(name):
        killed.append(name)
    monkeypatch.setattr(server_utils, 'psutil', None)
    monkeypatch.setattr(server_utils, 'subprocess', server_utils.subprocess)
    # Should not raise
    server_utils.shutdown_services()
    assert 'cloudflared' in killed or 'uvicorn' in killed or 'python' in killed
