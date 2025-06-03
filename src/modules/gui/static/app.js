console.log('DEBUG: app.js loaded at', new Date().toISOString(), 'VERSION: 2025-06-02-test');

const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
let ws;
let currentWsUrl = wsUrl;

function connectWS(customUrl) {
  if (ws) ws.close();
  currentWsUrl = customUrl || wsUrl;
  console.log('Opening WebSocket to', currentWsUrl);
  ws = new WebSocket(currentWsUrl);
  ws.onopen = () => {
    console.log('WebSocket connected');
    const status = document.getElementById('status');
    if (status) status.textContent = 'Connected.';
    const progressBarContainer = document.getElementById('progressBarContainer');
    if (progressBarContainer) progressBarContainer.style.display = 'none';
  };
  ws.onclose = () => {
    console.log('WebSocket closed, retrying...');
    const status = document.getElementById('status');
    if (status) status.textContent = 'Disconnected. Retrying...';
    setTimeout(() => connectWS(currentWsUrl), 2000);
  };
  ws.onerror = (e) => {
    console.log('WebSocket error', e);
    const status = document.getElementById('status');
    if (status) status.textContent = 'WebSocket error.';
  };
  ws.onmessage = (event) => {
    console.log('WebSocket message:', event.data);
    const msg = JSON.parse(event.data);
    if (msg.type === 'modules') {
      const sel = document.getElementById('modules');
      if (sel) {
        sel.innerHTML = '';
        msg.modules.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          sel.appendChild(opt);
        });
      }
      const moduleSelect = document.getElementById('moduleSelect');
      if (moduleSelect) moduleSelect.style.display = '';
    } else if (msg.type === 'progress') {
      const progress = document.getElementById('progress');
      const progressBarContainer = document.getElementById('progressBarContainer');
      if (progress) {
        progress.style.display = '';
        progress.textContent = msg.msg;
      }
      if (progressBarContainer) progressBarContainer.style.display = '';
      let percent = msg.percent || 0;
      const progressBar = document.getElementById('progressBar');
      if (progressBar) progressBar.style.width = percent + '%';
    } else if (msg.type === 'result') {
      const progress = document.getElementById('progress');
      const progressBarContainer = document.getElementById('progressBarContainer');
      const result = document.getElementById('result');
      const recordingIndicator = document.getElementById('recordingIndicator');
      if (progress) progress.style.display = 'none';
      if (progressBarContainer) progressBarContainer.style.display = 'none';
      if (result) {
        result.style.display = '';
        result.textContent = JSON.stringify(msg, null, 2);
      }
      if (recordingIndicator) recordingIndicator.style.display = 'none';
    } else if (msg.type === 'recording') {
      const recordingIndicator = document.getElementById('recordingIndicator');
      if (recordingIndicator) recordingIndicator.style.display = msg.active ? '' : 'none';
    } else if (msg.type === 'error') {
      alert(msg.msg);
      const progressBarContainer = document.getElementById('progressBarContainer');
      const recordingIndicator = document.getElementById('recordingIndicator');
      if (progressBarContainer) progressBarContainer.style.display = 'none';
      if (recordingIndicator) recordingIndicator.style.display = 'none';
    }
  };
}

document.addEventListener('DOMContentLoaded', () => {
  console.log('DOMContentLoaded fired');
  const appDiv = document.getElementById('app');
  if (appDiv) {
    appDiv.style.display = '';
    console.log('#app made visible');
  } else {
    console.log('No #app element found');
  }

  // Add Admin Login/Logout button
  const adminDiv = document.createElement('div');
  adminDiv.style.marginBottom = '1em';
  adminDiv.innerHTML = `
    <button id="adminLoginBtn">Admin Login</button>
    <button id="adminLogoutBtn" style="display:none;">Admin Logout</button>
    <span id="adminStatus"></span>
  `;
  if (appDiv) appDiv.prepend(adminDiv);

  async function refreshAdmin() {
    const resp = await fetch('/api/v1/whoami');
    const data = await resp.json();
    const adminPanel = document.getElementById('adminPanel');
    const adminLoginBtn = document.getElementById('adminLoginBtn');
    const adminLogoutBtn = document.getElementById('adminLogoutBtn');
    const adminStatus = document.getElementById('adminStatus');
    if (!adminPanel) console.warn('adminPanel not found');
    if (!adminLoginBtn) console.warn('adminLoginBtn not found');
    if (!adminLogoutBtn) console.warn('adminLogoutBtn not found');
    if (!adminStatus) console.warn('adminStatus not found');
    if (data.admin) {
      if (adminPanel) adminPanel.style.display = '';
      if (adminLoginBtn) adminLoginBtn.style.display = 'none';
      if (adminLogoutBtn) adminLogoutBtn.style.display = '';
      if (adminStatus) adminStatus.textContent = ' (Admin mode)';
    } else {
      if (adminPanel) adminPanel.style.display = 'none';
      if (adminLoginBtn) adminLoginBtn.style.display = '';
      if (adminLogoutBtn) adminLogoutBtn.style.display = 'none';
      if (adminStatus) adminStatus.textContent = '';
    }
  }

  const adminLoginBtn = document.getElementById('adminLoginBtn');
  if (adminLoginBtn) {
    adminLoginBtn.onclick = async () => {
      await fetch('/api/v1/admin_login', { method: 'POST' });
      await refreshAdmin();
    };
  }
  const adminLogoutBtn = document.getElementById('adminLogoutBtn');
  if (adminLogoutBtn) {
    adminLogoutBtn.onclick = async () => {
      await fetch('/api/v1/admin_logout', { method: 'POST' });
      await refreshAdmin();
    };
  }

  refreshAdmin();

  // Always connect WebSocket on load
  console.log('Connecting WebSocket...');
  connectWS();

  // Workflow button
  const startBtn = document.getElementById('startBtn');
  if (startBtn) {
    startBtn.onclick = () => {
      if (ws) ws.send(JSON.stringify({type: 'start_workflow'}));
    };
  }

  // Module run button (optional, only if present)
  const runBtn = document.getElementById('runBtn');
  if (runBtn) {
    runBtn.onclick = () => {
      const moduleSel = document.getElementById('modules');
      const module = moduleSel ? moduleSel.value : '';
      const defaultsCheckbox = document.getElementById('defaultsCheckbox');
      const useDefaults = defaultsCheckbox ? defaultsCheckbox.checked : false;
      if (ws) ws.send(JSON.stringify({type: 'run_module', module: module, input: {}, defaults: useDefaults}));
      const progress = document.getElementById('progress');
      const result = document.getElementById('result');
      if (progress) {
        progress.style.display = '';
        progress.textContent = 'Running...';
      }
      if (result) result.style.display = 'none';
    };
  }

  // Add recording button for demonstration (admin only)
  const adminPanel = document.getElementById('adminPanel');
  if (adminPanel) {
    const recBtn = document.createElement('button');
    recBtn.textContent = 'Toggle Recording';
    recBtn.style.marginTop = '1em';
    recBtn.onclick = () => {
      const recordingIndicator = document.getElementById('recordingIndicator');
      if (ws && recordingIndicator)
        ws.send(JSON.stringify({type: 'recording', active: recordingIndicator.style.display === 'none'}));
    };
    adminPanel.appendChild(recBtn);
  }

  // In the UI, update any file upload or accepted file type logic to allow PDF for both document analysis and transcription.
  // In the API docs, clarify that PDF is accepted for both workflows.
  // In the FastAPI docs, ensure /api/generate and any upload endpoints mention PDF as a valid input for transcription as well as analysis.
});
