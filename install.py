#!/usr/bin/env python3
"""
INTV Automated Installation Script

This script handles both pipx package installation and native system dependencies
to avoid global install limitations with native packages.

Features:
- Automatic platform detection (Linux, macOS, Windows)
- System package manager detection (apt, yum, brew, chocolatey)
- Native dependency installation (tesseract-ocr, poppler-utils)
- Hardware-optimized pipx installation (GPU detection)
- Comprehensive error handling and fallback options
- Verification and testing of installed components

Usage:
    python install.py [--dry-run] [--gpu-only] [--cpu-only] [--force-reinstall]
"""

import argparse
import os
import platform
import subprocess
import sys
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_step(text: str):
    """Print a step indicator"""
    print(f"\n{Colors.OKBLUE}🔹 {text}{Colors.ENDC}")


def print_success(text: str):
    """Print a success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print a warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_error(text: str):
    """Print an error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print an info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def run_command(command: List[str], shell: bool = False, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command with proper error handling"""
    try:
        cmd_str = ' '.join(command) if isinstance(command, list) else command
        print(f"    Running: {cmd_str}")
        
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.stdout and capture_output:
            print(f"    Output: {result.stdout.strip()}")
        
        return result
    except subprocess.CalledProcessError as e:
        if capture_output:
            error_msg = e.stderr.strip() if e.stderr else "No error details"
            print_error(f"Command failed: {cmd_str}")
            print_error(f"Error: {error_msg}")
        raise
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out: {cmd_str}")
        raise


class SystemDetector:
    """Detect system platform, package managers, and hardware capabilities"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.machine = platform.machine().lower()
        self.release = platform.release()
        self.version = platform.version()
        
    def detect_platform(self) -> Dict:
        """Detect platform and available package managers"""
        info = {
            'platform': self.platform,
            'machine': self.machine,
            'release': self.release,
            'package_managers': [],
            'is_wsl': False,
            'is_raspberry_pi': False
        }
        
        # Check for WSL
        if 'microsoft' in self.version.lower() or 'wsl' in self.version.lower():
            info['is_wsl'] = True
        
        # Check for Raspberry Pi
        if 'arm' in self.machine or 'aarch64' in self.machine:
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'raspberry pi' in model:
                        info['is_raspberry_pi'] = True
            except:
                pass
        
        # Detect package managers
        if self.platform == 'linux':
            # Check for various Linux package managers
            managers = {
                'apt': ['apt', 'apt-get'],
                'yum': ['yum'],
                'dnf': ['dnf'],
                'pacman': ['pacman'],
                'zypper': ['zypper'],
                'emerge': ['emerge'],
                'snap': ['snap'],
                'flatpak': ['flatpak']
            }
            
            for manager, commands in managers.items():
                for cmd in commands:
                    if self._command_exists(cmd):
                        info['package_managers'].append(manager)
                        break
                        
        elif self.platform == 'darwin':
            # macOS package managers
            if self._command_exists('brew'):
                info['package_managers'].append('brew')
            if self._command_exists('port'):
                info['package_managers'].append('macports')
                
        elif self.platform == 'windows':
            # Windows package managers
            if self._command_exists('choco'):
                info['package_managers'].append('chocolatey')
            if self._command_exists('winget'):
                info['package_managers'].append('winget')
            if self._command_exists('scoop'):
                info['package_managers'].append('scoop')
        
        return info
    
    def detect_gpu(self) -> Dict:
        """Detect GPU capabilities"""
        gpu_info = {
            'has_nvidia': False,
            'has_amd': False,
            'has_intel': False,
            'nvidia_devices': [],
            'total_vram_gb': 0,
            'recommended_backend': 'cpu'
        }
        
        # Try nvidia-smi for NVIDIA GPUs
        try:
            result = run_command(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                               capture_output=True, check=False)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory_mb = int(parts[1].strip())
                            memory_gb = memory_mb / 1024
                            gpu_info['nvidia_devices'].append({
                                'name': name,
                                'memory_gb': memory_gb
                            })
                            gpu_info['total_vram_gb'] += memory_gb
                
                if gpu_info['nvidia_devices']:
                    gpu_info['has_nvidia'] = True
                    gpu_info['recommended_backend'] = 'cuda'
        except:
            pass
        
        # Check for AMD GPUs (ROCm)
        try:
            result = run_command(['rocm-smi', '--showproductname'], capture_output=True, check=False)
            if result.returncode == 0:
                gpu_info['has_amd'] = True
                if not gpu_info['has_nvidia']:  # Prefer CUDA over ROCm
                    gpu_info['recommended_backend'] = 'rocm'
        except:
            pass
        
        # Check for Intel GPUs
        try:
            result = run_command(['lspci'], capture_output=True, check=False)
            if result.returncode == 0 and 'intel' in result.stdout.lower():
                lines = result.stdout.lower().split('\n')
                for line in lines:
                    if 'vga' in line and 'intel' in line:
                        gpu_info['has_intel'] = True
                        break
        except:
            pass
        
        # Special handling for Apple Silicon
        if self.platform == 'darwin' and ('arm64' in self.machine or 'aarch64' in self.machine):
            gpu_info['recommended_backend'] = 'mps'
        
        return gpu_info
    
    def detect_python_environment(self) -> Dict:
        """Detect Python environment details"""
        return {
            'python_version': platform.python_version(),
            'python_executable': sys.executable,
            'pip_available': self._command_exists('pip'),
            'pipx_available': self._command_exists('pipx'),
            'virtual_env': os.environ.get('VIRTUAL_ENV') is not None,
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV') is not None
        }
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            run_command(['which', command] if self.platform != 'windows' else ['where', command], 
                       capture_output=True, check=True)
            return True
        except:
            return False


class DependencyInstaller:
    """Handle installation of native system dependencies"""
    
    def __init__(self, system_info: Dict, dry_run: bool = False):
        self.system_info = system_info
        self.dry_run = dry_run
        
    def install_native_dependencies(self) -> bool:
        """Install native system dependencies based on platform"""
        print_step("Installing native system dependencies...")
        
        platform_name = self.system_info['platform']
        package_managers = self.system_info['package_managers']
        
        if platform_name == 'linux':
            return self._install_linux_dependencies(package_managers)
        elif platform_name == 'darwin':
            return self._install_macos_dependencies(package_managers)
        elif platform_name == 'windows':
            return self._install_windows_dependencies(package_managers)
        else:
            print_warning(f"Unsupported platform: {platform_name}")
            return False
    
    def _install_linux_dependencies(self, package_managers: List[str]) -> bool:
        """Install dependencies on Linux"""
        success = True
        
        # Define packages for different package managers
        packages = {
            'apt': ['tesseract-ocr', 'tesseract-ocr-eng', 'poppler-utils', 'ffmpeg'],
            'yum': ['tesseract', 'poppler-utils', 'ffmpeg'],
            'dnf': ['tesseract', 'poppler-utils', 'ffmpeg'],
            'pacman': ['tesseract', 'poppler', 'ffmpeg'],
            'zypper': ['tesseract-ocr', 'poppler-tools', 'ffmpeg']
        }
        
        # Try package managers in order of preference
        preferred_order = ['apt', 'dnf', 'yum', 'pacman', 'zypper']
        
        for manager in preferred_order:
            if manager in package_managers:
                try:
                    pkg_list = packages.get(manager, [])
                    if not pkg_list:
                        continue
                    
                    print_info(f"Using {manager} package manager")
                    
                    if manager == 'apt':
                        # Update package list first
                        if not self.dry_run:
                            run_command(['sudo', 'apt', 'update'])
                        else:
                            print("    [DRY RUN] sudo apt update")
                        
                        # Install packages
                        cmd = ['sudo', 'apt', 'install', '-y'] + pkg_list
                        if not self.dry_run:
                            run_command(cmd)
                        else:
                            print(f"    [DRY RUN] {' '.join(cmd)}")
                            
                    elif manager in ['yum', 'dnf']:
                        cmd = ['sudo', manager, 'install', '-y'] + pkg_list
                        if not self.dry_run:
                            run_command(cmd)
                        else:
                            print(f"    [DRY RUN] {' '.join(cmd)}")
                            
                    elif manager == 'pacman':
                        cmd = ['sudo', 'pacman', '-S', '--noconfirm'] + pkg_list
                        if not self.dry_run:
                            run_command(cmd)
                        else:
                            print(f"    [DRY RUN] {' '.join(cmd)}")
                            
                    elif manager == 'zypper':
                        cmd = ['sudo', 'zypper', 'install', '-y'] + pkg_list
                        if not self.dry_run:
                            run_command(cmd)
                        else:
                            print(f"    [DRY RUN] {' '.join(cmd)}")
                    
                    print_success(f"Native dependencies installed via {manager}")
                    return True
                    
                except Exception as e:
                    print_warning(f"Failed to install via {manager}: {e}")
                    success = False
                    continue
        
        if not success:
            print_error("Failed to install native dependencies with any package manager")
            self._print_manual_linux_instructions()
        
        return success
    
    def _install_macos_dependencies(self, package_managers: List[str]) -> bool:
        """Install dependencies on macOS"""
        if 'brew' in package_managers:
            try:
                print_info("Using Homebrew package manager")
                packages = ['tesseract', 'poppler', 'ffmpeg']
                
                # Update brew
                if not self.dry_run:
                    run_command(['brew', 'update'])
                else:
                    print("    [DRY RUN] brew update")
                
                # Install packages
                for package in packages:
                    cmd = ['brew', 'install', package]
                    if not self.dry_run:
                        run_command(cmd, check=False)  # Don't fail if already installed
                    else:
                        print(f"    [DRY RUN] {' '.join(cmd)}")
                
                print_success("Native dependencies installed via Homebrew")
                return True
                
            except Exception as e:
                print_warning(f"Failed to install via Homebrew: {e}")
        
        print_error("Homebrew not available or failed")
        self._print_manual_macos_instructions()
        return False
    
    def _install_windows_dependencies(self, package_managers: List[str]) -> bool:
        """Install dependencies on Windows"""
        success = False
        
        # Try Chocolatey first
        if 'chocolatey' in package_managers:
            try:
                print_info("Using Chocolatey package manager")
                packages = ['tesseract', 'poppler', 'ffmpeg']
                
                for package in packages:
                    cmd = ['choco', 'install', package, '-y']
                    if not self.dry_run:
                        run_command(cmd, check=False)
                    else:
                        print(f"    [DRY RUN] {' '.join(cmd)}")
                
                print_success("Native dependencies installed via Chocolatey")
                success = True
                
            except Exception as e:
                print_warning(f"Failed to install via Chocolatey: {e}")
        
        # Try winget
        if not success and 'winget' in package_managers:
            try:
                print_info("Using winget package manager")
                packages = ['tesseract', 'poppler', 'ffmpeg']
                
                for package in packages:
                    cmd = ['winget', 'install', package]
                    if not self.dry_run:
                        run_command(cmd, check=False)
                    else:
                        print(f"    [DRY RUN] {' '.join(cmd)}")
                
                print_success("Native dependencies installed via winget")
                success = True
                
            except Exception as e:
                print_warning(f"Failed to install via winget: {e}")
        
        if not success:
            print_error("No suitable package manager found for Windows")
            self._print_manual_windows_instructions()
        
        return success
    
    def _print_manual_linux_instructions(self):
        """Print manual installation instructions for Linux"""
        print_info("Manual installation required:")
        print("  For Debian/Ubuntu:")
        print("    sudo apt update && sudo apt install -y tesseract-ocr poppler-utils ffmpeg")
        print("  For RHEL/CentOS/Fedora:")
        print("    sudo dnf install tesseract poppler-utils ffmpeg")
        print("  For Arch Linux:")
        print("    sudo pacman -S tesseract poppler ffmpeg")
    
    def _print_manual_macos_instructions(self):
        """Print manual installation instructions for macOS"""
        print_info("Manual installation required:")
        print("  Install Homebrew first: https://brew.sh")
        print("  Then run: brew install tesseract poppler ffmpeg")
    
    def _print_manual_windows_instructions(self):
        """Print manual installation instructions for Windows"""
        print_info("Manual installation required:")
        print("  Tesseract: https://github.com/tesseract-ocr/tesseract/wiki")
        print("  Poppler: http://blog.alivate.com.au/poppler-utils-windows/")
        print("  FFmpeg: https://ffmpeg.org/download.html")
        print("  Add all to your system PATH")


class PipxInstaller:
    """Handle pipx installation and dependency injection"""
    
    def __init__(self, python_info: Dict, gpu_info: Dict, dry_run: bool = False):
        self.python_info = python_info
        self.gpu_info = gpu_info
        self.dry_run = dry_run
        
    def ensure_pipx(self) -> bool:
        """Ensure pipx is installed and available"""
        print_step("Checking pipx installation...")
        
        if self.python_info['pipx_available']:
            print_success("pipx is already installed")
            return True
        
        print_info("Installing pipx...")
        try:
            # Install pipx
            cmd = [sys.executable, '-m', 'pip', 'install', '--user', 'pipx']
            if not self.dry_run:
                run_command(cmd)
            else:
                print(f"    [DRY RUN] {' '.join(cmd)}")
            
            # Ensure path
            cmd = [sys.executable, '-m', 'pipx', 'ensurepath']
            if not self.dry_run:
                run_command(cmd)
            else:
                print(f"    [DRY RUN] {' '.join(cmd)}")
            
            print_success("pipx installed successfully")
            print_warning("You may need to restart your terminal or run 'source ~/.bashrc' to use pipx")
            return True
            
        except Exception as e:
            print_error(f"Failed to install pipx: {e}")
            return False
    
    def install_intv(self, cpu_only: bool = False, force_reinstall: bool = False, local_install: bool = False) -> bool:
        """Install INTV with appropriate dependency groups"""
        print_step("Installing INTV package...")
        
        # Check if we should install locally (development mode)
        if local_install or os.path.exists('pyproject.toml'):
            print_info("Local pyproject.toml detected - installing from current directory")
            variant = "."
            if not cpu_only:
                backend = self.gpu_info['recommended_backend']
                if backend == 'cuda':
                    print_info("Local install with CUDA optimization will be handled by dependency injection")
                elif backend == 'rocm':
                    print_info("Local install with ROCm optimization will be handled by dependency injection")
                elif backend == 'mps':
                    print_info("Local install with Apple Silicon optimization will be handled by dependency injection")
                else:
                    print_info("Local install with CPU-only configuration")
            else:
                print_info("Local install with CPU-only variant")
        else:
            # Determine installation variant based on hardware
            if cpu_only:
                variant = "intv[full-cpu]"
                print_info("Installing CPU-only variant from PyPI")
            else:
                backend = self.gpu_info['recommended_backend']
                if backend == 'cuda':
                    variant = "intv[full-cuda]"
                    print_info("Installing CUDA-optimized variant from PyPI")
                elif backend == 'rocm':
                    variant = "intv[full-rocm]"
                    print_info("Installing ROCm-optimized variant from PyPI")
                elif backend == 'mps':
                    variant = "intv[full-mps]"
                    print_info("Installing Apple Silicon-optimized variant from PyPI")
                else:
                    variant = "intv[full-cpu]"
                    print_info("Installing CPU-only variant from PyPI (no GPU detected)")
        
        try:
            # Check if already installed
            if not force_reinstall:
                try:
                    result = run_command(['pipx', 'list'], capture_output=True, check=False)
                    if result.returncode == 0 and 'intv' in result.stdout:
                        print_info("INTV is already installed")
                        if not self._ask_reinstall():
                            return True
                        force_reinstall = True
                except:
                    pass
            
            # Uninstall if force reinstall
            if force_reinstall:
                print_info("Uninstalling existing INTV installation...")
                if not self.dry_run:
                    run_command(['pipx', 'uninstall', 'intv'], check=False)
                else:
                    print("    [DRY RUN] pipx uninstall intv")
            
            # Install INTV
            cmd = ['pipx', 'install', variant]
            if not self.dry_run:
                run_command(cmd)
            else:
                print(f"    [DRY RUN] {' '.join(cmd)}")
            
            print_success("INTV package installed successfully")
            return True
            
        except Exception as e:
            print_error(f"Failed to install INTV: {e}")
            return False
    
    def inject_additional_dependencies(self) -> bool:
        """Inject additional dependencies based on system capabilities"""
        print_step("Installing additional dependencies...")
        
        # GPU-specific dependencies
        if self.gpu_info['recommended_backend'] == 'cuda':
            try:
                print_info("Installing CUDA-specific PyTorch...")
                cmd = ['pipx', 'inject', 'intv', 'torch', 'torchvision', 'torchaudio', 
                       '--index-url', 'https://download.pytorch.org/whl/cu121']
                if not self.dry_run:
                    run_command(cmd)
                else:
                    print(f"    [DRY RUN] {' '.join(cmd)}")
                
                print_success("CUDA PyTorch installed")
            except Exception as e:
                print_warning(f"Failed to install CUDA PyTorch: {e}")
        
        # Optional enhanced dependencies
        optional_deps = [
            'pyannote.audio',  # Advanced audio processing
            'faiss-gpu' if self.gpu_info['has_nvidia'] else 'faiss-cpu',  # Vector search
            'chromadb',  # Vector database
        ]
        
        for dep in optional_deps:
            try:
                print_info(f"Installing {dep}...")
                cmd = ['pipx', 'inject', 'intv', dep]
                if not self.dry_run:
                    run_command(cmd, check=False)  # Don't fail on optional deps
                else:
                    print(f"    [DRY RUN] {' '.join(cmd)}")
            except Exception as e:
                print_warning(f"Failed to install optional dependency {dep}: {e}")
        
        return True
    
    def _ask_reinstall(self) -> bool:
        """Ask user if they want to reinstall"""
        try:
            response = input("INTV is already installed. Reinstall? [y/N]: ").strip().lower()
            return response in ['y', 'yes']
        except KeyboardInterrupt:
            print("\nInstallation cancelled by user")
            return False


class InstallationVerifier:
    """Verify that installation was successful"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
    
    def verify_installation(self) -> bool:
        """Run comprehensive verification of the installation"""
        print_step("Verifying installation...")
        
        success = True
        
        # Test INTV CLI
        success &= self._test_intv_cli()
        
        # Test native dependencies
        success &= self._test_native_dependencies()
        
        # Test Python dependencies
        success &= self._test_python_dependencies()
        
        if success:
            print_success("All installation verification tests passed!")
        else:
            print_error("Some verification tests failed")
        
        return success
    
    def _test_intv_cli(self) -> bool:
        """Test INTV CLI is working"""
        try:
            print_info("Testing INTV CLI...")
            if not self.dry_run:
                result = run_command(['intv', '--version'], capture_output=True)
                if result.returncode == 0:
                    print_success("INTV CLI is working")
                    return True
            else:
                print("    [DRY RUN] intv --version")
                return True
        except Exception as e:
            print_error(f"INTV CLI test failed: {e}")
        
        return False
    
    def _test_native_dependencies(self) -> bool:
        """Test native dependencies are available"""
        dependencies = {
            'tesseract': ['tesseract', '--version'],
            'poppler': ['pdftoppm', '-v'],  # Part of poppler-utils
            'ffmpeg': ['ffmpeg', '-version']
        }
        
        success = True
        for name, cmd in dependencies.items():
            try:
                print_info(f"Testing {name}...")
                if not self.dry_run:
                    result = run_command(cmd, capture_output=True, check=False)
                    if result.returncode == 0:
                        print_success(f"{name} is available")
                    else:
                        print_warning(f"{name} test failed")
                        success = False
                else:
                    print(f"    [DRY RUN] {' '.join(cmd)}")
            except Exception as e:
                print_warning(f"{name} test failed: {e}")
                success = False
        
        return success
    
    def _test_python_dependencies(self) -> bool:
        """Test key Python dependencies"""
        # Map package names to their import names
        dependencies = {
            'torch': 'torch',
            'transformers': 'transformers', 
            'sentence-transformers': 'sentence_transformers',
            'pytesseract': 'pytesseract',
            'pdf2image': 'pdf2image',
            'pillow': 'PIL',
            'faiss-cpu': 'faiss',
            'python-docx': 'docx',
            'pyyaml': 'yaml'
        }
        
        # Use the pipx virtual environment's Python interpreter
        pipx_python = os.path.expanduser('~/.local/share/pipx/venvs/intv/bin/python')
        if not os.path.exists(pipx_python):
            print_warning("pipx virtual environment not found, falling back to system Python")
            pipx_python = 'python3'
        
        success = True
        for package_name, import_name in dependencies.items():
            try:
                print_info(f"Testing {package_name}...")
                if not self.dry_run:
                    result = run_command([pipx_python, '-c', f'import {import_name}; print(f"{package_name}: OK")'], 
                                       capture_output=True, check=False)
                    if result.returncode == 0:
                        print_success(f"{package_name} is importable")
                    else:
                        print_warning(f"{package_name} import failed")
                        success = False
                else:
                    print(f"    [DRY RUN] {pipx_python} -c 'import {import_name}'")
            except Exception as e:
                print_warning(f"{package_name} test failed: {e}")
                success = False
        
        return success


def main():
    """Main installation script"""
    parser = argparse.ArgumentParser(description='INTV Automated Installation Script')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually doing it')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Only install GPU-optimized packages (skip if no GPU)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only installation even if GPU is detected')
    parser.add_argument('--force-reinstall', action='store_true',
                        help='Force reinstallation even if already installed')
    parser.add_argument('--skip-native', action='store_true',
                        help='Skip native dependency installation')
    parser.add_argument('--skip-verification', action='store_true',
                        help='Skip installation verification')
    parser.add_argument('--local-install', action='store_true',
                        help='Install from local directory (development mode)')
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("INTV Automated Installation Script")
    
    if args.dry_run:
        print_warning("DRY RUN MODE - No actual changes will be made")
    
    # Detect system
    print_step("Detecting system configuration...")
    detector = SystemDetector()
    
    system_info = detector.detect_platform()
    gpu_info = detector.detect_gpu()
    python_info = detector.detect_python_environment()
    
    # Print system information
    print_info(f"Platform: {system_info['platform']} ({system_info['machine']})")
    print_info(f"Package managers: {', '.join(system_info['package_managers']) if system_info['package_managers'] else 'None detected'}")
    print_info(f"Python: {python_info['python_version']}")
    print_info(f"GPU Backend: {gpu_info['recommended_backend']}")
    
    if gpu_info['nvidia_devices']:
        for gpu in gpu_info['nvidia_devices']:
            print_info(f"NVIDIA GPU: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    
    if system_info['is_raspberry_pi']:
        print_info("Raspberry Pi detected")
    if system_info['is_wsl']:
        print_info("WSL environment detected")
    
    # Check prerequisites
    if not python_info['pip_available']:
        print_error("pip is not available. Please install Python pip first.")
        return 1
    
    if python_info['virtual_env'] or python_info['conda_env']:
        print_warning("You are in a virtual environment. INTV will be installed globally via pipx.")
    
    # GPU-only mode check
    if args.gpu_only and gpu_info['recommended_backend'] == 'cpu':
        print_error("--gpu-only specified but no GPU detected. Aborting.")
        return 1
    
    success = True
    
    try:
        # Install native dependencies
        if not args.skip_native:
            installer = DependencyInstaller(system_info, args.dry_run)
            if not installer.install_native_dependencies():
                print_warning("Native dependency installation failed. Continuing anyway...")
                success = False
        
        # Install pipx and INTV
        pipx_installer = PipxInstaller(python_info, gpu_info, args.dry_run)
        
        if not pipx_installer.ensure_pipx():
            print_error("Failed to install pipx")
            return 1
        
        if not pipx_installer.install_intv(args.cpu_only, args.force_reinstall, args.local_install):
            print_error("Failed to install INTV")
            return 1
        
        # Install additional dependencies
        if not pipx_installer.inject_additional_dependencies():
            print_warning("Some additional dependencies failed to install")
            success = False
        
        # Verify installation
        if not args.skip_verification:
            verifier = InstallationVerifier(args.dry_run)
            if not verifier.verify_installation():
                print_warning("Installation verification failed")
                success = False
        
        # Print final status
        if success:
            print_header("Installation Complete!")
            print_success("INTV has been successfully installed!")
            print_info("Try these commands to get started:")
            print("    intv --help")
            print("    intv-platform")
            print("    intv process --help")
            print("    intv module --help")
        else:
            print_header("Installation Completed with Warnings")
            print_warning("INTV was installed but some components may not work correctly.")
            print_info("Check the warnings above and install missing dependencies manually.")
        
        return 0 if success else 2
        
    except KeyboardInterrupt:
        print_error("\nInstallation cancelled by user")
        return 1
    except Exception as e:
        print_error(f"Installation failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
