@echo off
setlocal enabledelayedexpansion

REM =============================
REM Intv_App Startup Script (Windows)
REM Supports CLI and GUI (FastAPI) modes, cloudflared tunnel, and robust process management
REM Usage:
REM   intv_startup.bat [--cli|--gui] [--cloudflare] [--exit]
REM   --cli        Run CLI pipeline (python src/main.py)
REM   --gui        Run FastAPI GUI (default)
REM   --cloudflare Start cloudflared tunnel (default: off)
REM   --exit       Stop all FastAPI and cloudflared processes
REM =============================

REM Move to script directory
cd /d %~dp0
cd ..

REM Activate venv if present
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Set default ports and mode
set API_PORT=3773
set API_HOST=0.0.0.0
set APP_MODE=gui
set USE_CLOUDFLARE_TUNNEL=false

REM Parse CLI arguments
:parse_args
if "%1"=="--cli" set APP_MODE=cli
if "%1"=="--gui" set APP_MODE=gui
if "%1"=="--cloudflare" set USE_CLOUDFLARE_TUNNEL=true
if "%1"=="--exit" goto :shutdown
shift
if not "%1"=="" goto :parse_args

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

REM Start FastAPI or CLI pipeline in the background and save PID
if /i "%APP_MODE%"=="cli" (
    start "IntvAppCLI" cmd /c "python src/main.py > cli_%API_PORT%.log 2>&1"
    for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq IntvAppCLI" /NH') do (
        echo %%a > %TEMP%\intvapp_cli_%API_PORT%.pid
        goto :after_cli_pid
    )
    :after_cli_pid
) else (
    start "FastAPI" cmd /c "uvicorn src.modules.gui.app:app --host %API_HOST% --port %API_PORT% --workers 4 > fastapi_%API_PORT%.log 2>&1"
    for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq FastAPI" /NH') do (
        echo %%a > %TEMP%\intvapp_fastapi_%API_PORT%.pid
        goto :after_fastapi_pid
    )
    :after_fastapi_pid
)

REM Start cloudflared if requested
if /i "%USE_CLOUDFLARE_TUNNEL%"=="true" (
    start "cloudflared" cmd /c "%CLOUDFLARED_BIN% tunnel --url http://localhost:%API_PORT% > cloudflared_%API_PORT%.log 2>&1"
    for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq cloudflared.exe" /FI "WINDOWTITLE eq cloudflared" /NH') do (
        echo %%a > %TEMP%\intvapp_cloudflared_%API_PORT%.pid
        goto :after_cloudflared_pid
    )
    :after_cloudflared_pid
)

REM Print info and exit
REM Wait for cloudflared public URL (not trivial in batch, so just print log location)
echo ===============================
echo Intv_App started in background (mode: %APP_MODE%).
echo Check cli_%API_PORT%.log, fastapi_%API_PORT%.log, and cloudflared_%API_PORT%.log for output.
echo ===============================
exit /b 0

REM Shutdown logic
:shutdown
echo Stopping all Intv_App (FastAPI, CLI, and cloudflared) processes on ports 3773 and 3774...
for %%P in (3773 3774) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%P ^| findstr LISTENING') do (
        set PID=%%a
        if not "!PID!"=="" (
            echo Killing PID !PID! on port %%P...
            taskkill /F /PID !PID! >nul 2>&1
        )
    )
)
REM Also kill any background IntvApp, FastAPI, CLI, or cloudflared started by this script (using PID files)
for %%F in (%TEMP%\intvapp_*.pid) do (
    if exist %%F (
        set /p PID=<%%F
        taskkill /F /PID !PID! >nul 2>&1
        del %%F
    )
)
echo All relevant processes stopped.
exit /b 0
