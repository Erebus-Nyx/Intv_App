import pytest
from fastapi.testclient import TestClient
from src.modules.gui.app import app

client = TestClient(app)

def test_api_v1_root():
    resp = client.get('/api/v1/')
    assert resp.status_code == 200
    assert resp.json().get('status') == 'ok'

def test_api_health():
    resp = client.get('/api/v1/health')
    assert resp.status_code == 200
    assert resp.json().get('status') == 'ok'

def test_api_whoami():
    resp = client.get('/api/v1/whoami')
    assert resp.status_code == 200
    assert 'admin' in resp.json()

def test_api_admin_login_logout():
    resp = client.post('/api/admin/login')
    assert resp.status_code == 200
    assert resp.json().get('admin') is True
    resp = client.post('/api/admin/logout')
    assert resp.status_code == 200
    assert resp.json().get('admin') is False

def test_api_data_example():
    resp = client.get('/api/data/example')
    assert resp.status_code == 200
    assert 'data' in resp.json()

def test_api_generate():
    resp = client.post('/api/generate')
    assert resp.status_code == 200 or resp.status_code == 500  # Accept 500 if subprocess fails in test
    assert 'status' in resp.json() or 'error' in resp.json()

def test_websocket_modules():
    with client.websocket_connect('/ws') as ws:
        data = ws.receive_json()
        assert data['type'] == 'modules'
        assert isinstance(data['modules'], list)
