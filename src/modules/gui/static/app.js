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
  // Remove user selector logic
  document.getElementById('app').style.display = '';

  // Add Admin Login/Logout button
  const adminDiv = document.createElement('div');
  adminDiv.style.marginBottom = '1em';
  adminDiv.innerHTML = `
    <button id="adminLoginBtn">Admin Login</button>
    <button id="adminLogoutBtn" style="display:none;">Admin Logout</button>
    <span id="adminStatus"></span>
  `;
  document.getElementById('app').prepend(adminDiv);

  async function refreshAdmin() {
    const resp = await fetch('/api/v1/whoami');
    const data = await resp.json();
    if (data.admin) {
      document.getElementById('adminPanel').style.display = '';
      document.getElementById('adminLoginBtn').style.display = 'none';
      document.getElementById('adminLogoutBtn').style.display = '';
      document.getElementById('adminStatus').textContent = ' (Admin mode)';
    } else {
      document.getElementById('adminPanel').style.display = 'none';
      document.getElementById('adminLoginBtn').style.display = '';
      document.getElementById('adminLogoutBtn').style.display = 'none';
      document.getElementById('adminStatus').textContent = '';
    }
  }

  document.getElementById('adminLoginBtn').onclick = async () => {
    await fetch('/api/v1/admin_login', { method: 'POST' });
    await refreshAdmin();
  };
  document.getElementById('adminLogoutBtn').onclick = async () => {
    await fetch('/api/v1/admin_logout', { method: 'POST' });
    await refreshAdmin();
  };

  refreshAdmin();

  // Password show/hide toggle with SVG eye icons
  const passwordInput = document.getElementById('password');
  const togglePassword = document.getElementById('togglePassword');
  if (togglePassword && passwordInput) {
    const eyeOpen = document.getElementById('eyeOpen');
    const eyeClosed = document.getElementById('eyeClosed');
    let shown = false;
    togglePassword.addEventListener('click', function() {
      shown = !shown;
      passwordInput.setAttribute('type', shown ? 'text' : 'password');
      if (shown) {
        eyeOpen.style.display = 'none';
        eyeClosed.style.display = '';
        togglePassword.setAttribute('aria-label', 'Hide password');
      } else {
        eyeOpen.style.display = '';
        eyeClosed.style.display = 'none';
        togglePassword.setAttribute('aria-label', 'Show password');
      }
    });
  }

  // Check authentication
  fetch('/api/v1/whoami', { credentials: 'same-origin' }).then(async r => {
    if (r.ok) {
      const data = await r.json();
      if (data.user || data.role) {
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
    } else {
      document.getElementById('loginPanel').style.display = '';
      document.getElementById('app').style.display = 'none';
    }
  }).catch(() => {
    document.getElementById('loginPanel').style.display = '';
    document.getElementById('app').style.display = 'none';
  });

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
