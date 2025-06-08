@echo off
REM INTV Quick Installation Script for Windows
REM This script downloads and runs the Python-based installation script

setlocal enabledelayedexpansion

echo ============================================
echo INTV Quick Installation for Windows
echo ============================================

REM Check if we're in the INTV directory with install.py
if exist "install.py" (
    echo [INFO] Found install.py in current directory
    set INSTALL_SCRIPT=install.py
) else if exist "C:\intv\install.py" (
    echo [INFO] Using install.py from C:\intv\
    set INSTALL_SCRIPT=C:\intv\install.py
) else (
    echo [ERROR] install.py not found!
    echo [INFO] Please run this script from the INTV project directory
    echo [INFO] Or download the project first:
    echo          git clone https://github.com/Erebus-Nyx/Intv_App.git
    echo          cd Intv_App
    echo          quick-install.bat
    exit /b 1
)

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 is required but not installed
    echo [INFO] Please install Python 3.10 or later from https://python.org
    pause
    exit /b 1
)

REM Parse command line arguments
set ARGS=
set DRY_RUN=false
set HELP=false

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--dry-run" (
    set ARGS=%ARGS% --dry-run
    set DRY_RUN=true
    echo [INFO] Dry run mode enabled
) else if "%~1"=="--cpu-only" (
    set ARGS=%ARGS% --cpu-only
    echo [INFO] CPU-only installation mode
) else if "%~1"=="--gpu-only" (
    set ARGS=%ARGS% --gpu-only
    echo [INFO] GPU-only installation mode
) else if "%~1"=="--force-reinstall" (
    set ARGS=%ARGS% --force-reinstall
    echo [INFO] Force reinstall mode enabled
) else if "%~1"=="--skip-native" (
    set ARGS=%ARGS% --skip-native
    echo [INFO] Skipping native dependency installation
) else if "%~1"=="--skip-verification" (
    set ARGS=%ARGS% --skip-verification
    echo [INFO] Skipping installation verification
) else if "%~1"=="--help" (
    set HELP=true
) else if "%~1"=="-h" (
    set HELP=true
) else (
    echo [ERROR] Unknown option: %~1
    echo [INFO] Use --help for usage information
    exit /b 1
)
shift
goto :parse_args

:args_done

if "%HELP%"=="true" (
    echo INTV Quick Installation Script for Windows
    echo.
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --dry-run             Show what would be done without doing it
    echo   --cpu-only            Force CPU-only installation
    echo   --gpu-only            Only install if GPU is detected
    echo   --force-reinstall     Force reinstallation
    echo   --skip-native         Skip native dependency installation
    echo   --skip-verification   Skip installation verification
    echo   --help, -h            Show this help message
    echo.
    echo Examples:
    echo   %0                    # Standard installation
    echo   %0 --dry-run          # See what would be installed
    echo   %0 --cpu-only         # CPU-only installation
    echo   %0 --force-reinstall  # Reinstall even if already installed
    pause
    exit /b 0
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python %PYTHON_VERSION% detected

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% == 0 (
    echo [WARNING] Running as administrator. This may cause permission issues with pipx.
    echo [INFO] Consider running as a regular user for better isolation.
)

REM Check available disk space (rough estimate - 5GB)
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set AVAILABLE_SPACE=%%a
if !AVAILABLE_SPACE! LSS 5000000000 (
    echo [WARNING] Low disk space detected. Installation requires ~5GB for models and dependencies.
)

REM Run the Python installation script
echo [INFO] Starting Python installation script...
echo.

python "%INSTALL_SCRIPT%" %ARGS%
set EXIT_CODE=%errorlevel%

if %EXIT_CODE% == 0 (
    echo.
    echo [SUCCESS] Installation completed successfully!
    echo.
    echo [INFO] Next steps:
    echo   1. Restart your command prompt
    echo   2. Test the installation: intv --version
    echo   3. Check platform info: intv-platform
    echo   4. Get help: intv --help
    echo.
    echo [INFO] Example usage:
    echo   intv process document.pdf
    echo   intv module create --interactive
    echo   intv module list
) else (
    echo.
    echo [ERROR] Installation failed with exit code %EXIT_CODE%
    echo.
    echo [INFO] Troubleshooting:
    echo   1. Try running with --dry-run to see what would be installed
    echo   2. Check system requirements and dependencies
    echo   3. Run as administrator if needed
    echo   4. Check internet connectivity
    echo   5. Install Chocolatey for easier package management: https://chocolatey.org
)

echo.
pause
exit /b %EXIT_CODE%
