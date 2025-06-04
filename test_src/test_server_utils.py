import os
import socket
import subprocess
import sys
import time
import types
import pytest
from intv_app import server_utils

def test_is_port_in_use_unused():
    # Pick a high port unlikely to be in use
    assert not server_utils.is_port_in_use(54321)

def test_is_port_in_use_used():
    s = socket.socket()
    s.bind(('localhost', 0))
    port = s.getsockname()[1]
    try:
        assert server_utils.is_port_in_use(port)
    finally:
        s.close()

def test_ensure_fastapi_running(monkeypatch):
    called = {}
    def fake_is_port_in_use(port):
        if not called.get('ran'):
            called['ran'] = True
            return False
        return True
    monkeypatch.setattr(server_utils, 'is_port_in_use', fake_is_port_in_use)
    monkeypatch.setattr(server_utils.subprocess, 'Popen', lambda *a, **k: None)
    server_utils.ensure_fastapi_running(port=54322)
    # Should print started, then print already running
    server_utils.ensure_fastapi_running(port=54322)

def test_ensure_cloudflared_running(monkeypatch):
    # Simulate cloudflared not running, then running
    called = {'ran': False}
    def fake_is_cloudflared_running():
        if not called['ran']:
            called['ran'] = True
            return False
        return True
    monkeypatch.setattr(server_utils, 'psutil', None)
    monkeypatch.setattr(server_utils, 'subprocess', server_utils.subprocess)
    monkeypatch.setattr(server_utils, 'is_port_in_use', lambda port: True)
    # Should not raise
    server_utils.ensure_cloudflared_running(progress=False)

def test_shutdown_services(monkeypatch):
    killed = []
    def fake_kill_by_name(name):
        killed.append(name)
    # Patch psutil to None and subprocess to a dummy
    monkeypatch.setattr(server_utils, 'psutil', None)
    monkeypatch.setattr(server_utils, 'subprocess', server_utils.subprocess)
    # Should not raise
    server_utils.shutdown_services()
    # We can't assert killed here, but should not error

def test_check_health_endpoint(monkeypatch):
    class FakeResp:
        status_code = 200
    monkeypatch.setattr('requests.get', lambda url, timeout=3: FakeResp())
    assert server_utils.check_health_endpoint('http://localhost:3773')

def test_setup_logging(tmp_path):
    log_file = tmp_path / 'test.log'
    logger = server_utils.setup_logging(str(log_file))
    logger.info('test log')
    assert log_file.exists()
    with open(log_file) as f:
        assert 'test log' in f.read()
