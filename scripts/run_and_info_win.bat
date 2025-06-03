@echo off
setlocal enabledelayedexpansion

REM Move to script directory
cd /d %~dp0
cd ..

REM Activate venv if present
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Set default ports
set API_PORT=3773
set API_HOST=0.0.0.0

REM Function to check and free port (Windows)
:free_port
set PORT=%1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :!PORT! ^| findstr LISTENING') do (
    set PID=%%a
    if not "!PID!"=="" (
        echo Port !PORT! is in use by PID: !PID!. Stopping it...
        taskkill /F /PID !PID! >nul 2>&1
        timeout /t 1 >nul
    )
)
REM Double-check port is free
set PORT_FREE=1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :!PORT! ^| findstr LISTENING') do (
    set PORT_FREE=0
)
exit /b !PORT_FREE!

REM Parse CLI arguments
if "%1"=="--exit" (
    echo Stopping all FastAPI and cloudflared processes on ports 3773 and 3774...
    for %%P in (3773 3774) do (
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%P ^| findstr LISTENING') do (
            set PID=%%a
            if not "!PID!"=="" (
                echo Killing PID !PID! on port %%P...
                taskkill /F /PID !PID! >nul 2>&1
            )
        )
    )
    REM Also kill any background FastAPI or cloudflared started by this script (using PID files)
    for %%F in (%TEMP%\intvapp_fastapi_*.pid %TEMP%\intvapp_cloudflared_*.pid) do (
        if exist %%F (
            set /p PID=<%%F
            taskkill /F /PID !PID! >nul 2>&1
            del %%F
        )
    )
    echo All relevant processes stopped.
    exit /b 0
)

REM Try to free 3773, else use 3774
call :free_port 3773
if errorlevel 1 (
    set API_PORT=3773
    echo Using port 3773.
) else (
    echo Port 3773 unavailable, trying 3774...
    call :free_port 3774
    if errorlevel 1 (
        set API_PORT=3774
        echo Using port 3774.
    ) else (
        echo Neither port 3773 nor 3774 is available. Exiting.
        exit /b 1
    )
)

REM Check for cloudflared.exe in PATH or download if missing
set CLOUDFLARED_BIN=cloudflared.exe
set CLOUDFLARED_LOCAL=%~dp0cloudflared-windows-amd64.exe
where cloudflared.exe >nul 2>&1
if errorlevel 1 (
    if not exist "%CLOUDFLARED_LOCAL%" (
        echo cloudflared not found in PATH. Downloading latest Windows binary to %CLOUDFLARED_LOCAL%...
        powershell -Command "Invoke-WebRequest -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile '%CLOUDFLARED_LOCAL%'"
    )
    set CLOUDFLARED_BIN=%CLOUDFLARED_LOCAL%
)

REM Start FastAPI app in the background and save PID
start "FastAPI" cmd /c "uvicorn src.modules.gui.app:app --host %API_HOST% --port %API_PORT% --workers 4 > fastapi_%API_PORT%.log 2>&1" 
REM Save PID (not trivial in Windows batch, so we use a workaround)
REM Find the most recent uvicorn process and save its PID
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq FastAPI" /NH') do (
    echo %%a > %TEMP%\intvapp_fastapi_%API_PORT%.pid
    goto :after_fastapi_pid
)
:after_fastapi_pid

REM Start cloudflared in the background and save PID
start "cloudflared" cmd /c "%CLOUDFLARED_BIN% tunnel --url http://localhost:%API_PORT% > cloudflared_%API_PORT%.log 2>&1"
REM Find the most recent cloudflared process and save its PID
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq cloudflared.exe" /FI "WINDOWTITLE eq cloudflared" /NH') do (
    echo %%a > %TEMP%\intvapp_cloudflared_%API_PORT%.pid
    goto :after_cloudflared_pid
)
:after_cloudflared_pid

REM Print info and exit
REM Wait for cloudflared public URL (not trivial in batch, so just print log location)
echo ===============================
echo FastAPI and cloudflared started in background.
echo Check fastapi_%API_PORT%.log and cloudflared_%API_PORT%.log for output.
echo ===============================
exit /b 0
