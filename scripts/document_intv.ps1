# PowerShell version of run_and_info.sh for Intv_App
param(
    [string]$Action = "start"
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path "$ScriptDir/.."
Set-Location $ProjectRoot

# Activate venv if present
$venvActivate = ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) { . $venvActivate }

$API_HOST = $env:API_HOST
if (-not $API_HOST) { $API_HOST = "0.0.0.0" }
$API_PORT = 3773

function Free-Port($port) {
    $pids = netstat -ano | Select-String ":$port" | Select-String "LISTENING" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Sort-Object -Unique
    foreach ($pid in $pids) {
        if ($pid -match '^\d+$') {
            Write-Host "Killing PID $pid on port $port..."
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 1
    # Double-check
    $stillUsed = netstat -ano | Select-String ":$port" | Select-String "LISTENING"
    return -not $stillUsed
}

if ($Action -eq '--exit') {
    Write-Host "Stopping all FastAPI and cloudflared processes on ports 3773 and 3774..."
    foreach ($p in 3773, 3774) {
        $pids = netstat -ano | Select-String ":$p" | Select-String "LISTENING" | ForEach-Object {
            ($_ -split '\s+')[-1]
        } | Sort-Object -Unique
        foreach ($pid in $pids) {
            if ($pid -match '^\d+$') {
                Write-Host "Killing PID $pid on port $p..."
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Get-ChildItem -Path "$env:TEMP" -Filter 'intvapp_*.pid' | ForEach-Object {
        $pid = Get-Content $_.FullName
        try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {}
        Remove-Item $_.FullName -Force
    }
    Write-Host "All relevant processes stopped."
    exit 0
}

# Try to free 3773, else use 3774
if (Free-Port 3773) {
    $API_PORT = 3773
    Write-Host "Using port 3773."
} elseif (Free-Port 3774) {
    $API_PORT = 3774
    Write-Host "Using port 3774."
} else {
    Write-Host "Neither port 3773 nor 3774 is available. Exiting."
    exit 1
}

# Check for cloudflared.exe in PATH or download if missing
$cloudflared_bin = "cloudflared.exe"
$cloudflared_local = Join-Path $ScriptDir "cloudflared-windows-amd64.exe"
if (-not (Get-Command $cloudflared_bin -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path $cloudflared_local)) {
        Write-Host "cloudflared not found in PATH. Downloading latest Windows binary to $cloudflared_local..."
        Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cloudflared_local
    }
    $cloudflared_bin = $cloudflared_local
}

# Start FastAPI app in the background
$fastapi_log = "fastapi_$API_PORT.log"
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "src.modules.gui.app:app --host $API_HOST --port $API_PORT --workers 4" -RedirectStandardOutput $fastapi_log -RedirectStandardError $fastapi_log

# Start cloudflared in the background
$cloudflared_log = "cloudflared_$API_PORT.log"
Start-Process -NoNewWindow -FilePath $cloudflared_bin -ArgumentList "tunnel --url http://localhost:$API_PORT" -RedirectStandardOutput $cloudflared_log -RedirectStandardError $cloudflared_log

Write-Host "==============================="
Write-Host "FastAPI and cloudflared started in background."
Write-Host "Check $fastapi_log and $cloudflared_log for output."
Write-Host "==============================="
