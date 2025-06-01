const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host.replace(/:.*$/, ':3773') + '/ws';
let ws;

function connectWS(customUrl) {
  if (ws) ws.close();
  wsUrl = customUrl || wsUrl;
  ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    document.getElementById('status').textContent = 'Connected.';
    document.getElementById('progressBarContainer').style.display = 'none';
  };
  ws.onclose = () => {
    document.getElementById('status').textContent = 'Disconnected. Retrying...';
    setTimeout(() => connectWS(wsUrl), 2000);
  };
  ws.onerror = (e) => {
    document.getElementById('status').textContent = 'WebSocket error.';
  };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'modules') {
      const sel = document.getElementById('modules');
      sel.innerHTML = '';
      msg.modules.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        sel.appendChild(opt);
      });
      document.getElementById('moduleSelect').style.display = '';
    } else if (msg.type === 'progress') {
      document.getElementById('progress').style.display = '';
      document.getElementById('progress').textContent = msg.msg;
      document.getElementById('progressBarContainer').style.display = '';
      let percent = msg.percent || 0;
      document.getElementById('progressBar').style.width = percent + '%';
    } else if (msg.type === 'result') {
      document.getElementById('progress').style.display = 'none';
      document.getElementById('progressBarContainer').style.display = 'none';
      document.getElementById('result').style.display = '';
      document.getElementById('result').textContent = JSON.stringify(msg, null, 2);
      document.getElementById('recordingIndicator').style.display = 'none';
    } else if (msg.type === 'recording') {
      document.getElementById('recordingIndicator').style.display = msg.active ? '' : 'none';
    } else if (msg.type === 'error') {
      alert(msg.msg);
      document.getElementById('progressBarContainer').style.display = 'none';
      document.getElementById('recordingIndicator').style.display = 'none';
    }
  };
}

document.addEventListener('DOMContentLoaded', () => {
  // Check authentication
  fetch('/whoami').then(async r => {
    if (r.ok) {
      const data = await r.json();
      document.getElementById('loginPanel').style.display = 'none';
      document.getElementById('app').style.display = '';
      connectWS();
      // --- Admin CLI Panel ---
      if (data.role === 'admin') {
        document.getElementById('adminPanel').style.display = '';
      }
    } else {
      document.getElementById('loginPanel').style.display = '';
      document.getElementById('app').style.display = 'none';
    }
  });

  document.getElementById('loginForm').onsubmit = async (e) => {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    const resp = await fetch('/login', { method: 'POST', body: formData, credentials: 'same-origin' });
    if (resp.redirected || resp.ok) {
      document.getElementById('loginError').style.display = 'none';
      location.reload();
    } else {
      const err = await resp.json();
      document.getElementById('loginError').textContent = err.error || 'Login failed';
      document.getElementById('loginError').style.display = '';
    }
  };

  document.getElementById('logoutBtn').onclick = () => {
    fetch('/logout').then(() => location.reload());
  };

  document.getElementById('startBtn').onclick = () => {
    ws.send(JSON.stringify({type: 'start_workflow'}));
  };
  document.getElementById('runBtn').onclick = () => {
    const module = document.getElementById('modules').value;
    const useDefaults = document.getElementById('defaultsCheckbox').checked;
    ws.send(JSON.stringify({type: 'run_module', module: module, input: {}, defaults: useDefaults}));
    document.getElementById('progress').style.display = '';
    document.getElementById('progress').textContent = 'Running...';
    document.getElementById('result').style.display = 'none';
  };

  document.getElementById('wssSetBtn').onclick = () => {
    const url = document.getElementById('wssInput').value.trim();
    if (url) connectWS(url);
  };

  // Admin CLI form submission
  const adminForm = document.getElementById('adminCliForm');
  if (adminForm) {
    adminForm.onsubmit = async (e) => {
      e.preventDefault();
      const file = document.getElementById('cliFile').value;
      const type = document.getElementById('cliType').value;
      const extra = document.getElementById('cliExtra').value;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', type);
      formData.append('extra_args', extra);
      const resp = await fetch('/admin/run-cli', { method: 'POST', body: formData, credentials: 'same-origin' });
      const out = document.getElementById('cliOutput');
      if (resp.ok) {
        const result = await resp.json();
        out.textContent = 'STDOUT:\n' + result.stdout + '\nSTDERR:\n' + result.stderr;
      } else {
        const err = await resp.json();
        out.textContent = 'Error: ' + (err.error || 'Unknown error');
      }
    };
  }

  // Add recording button for demonstration (admin only)
  if (document.getElementById('adminPanel')) {
    const recBtn = document.createElement('button');
    recBtn.textContent = 'Toggle Recording';
    recBtn.style.marginTop = '1em';
    recBtn.onclick = () => {
      ws.send(JSON.stringify({type: 'recording', active: document.getElementById('recordingIndicator').style.display === 'none'}));
    };
    document.getElementById('adminPanel').appendChild(recBtn);
  }
});
