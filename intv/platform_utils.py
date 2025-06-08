#!/usr/bin/env python3
"""
Platform utilities for INTV package
"""

def detect_hardware_capabilities():
    """Detect hardware capabilities including GPU, memory, etc."""
    capabilities = {
        'gpu': False,
        'gpu_memory_gb': 0,
        'gpu_name': 'None',
        'cpu_cores': 1,
        'system_memory_gb': 0
    }
    
    try:
        # Check for NVIDIA GPU
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                gpu_info = lines[0].split(', ')
                capabilities['gpu'] = True
                capabilities['gpu_name'] = gpu_info[0] if len(gpu_info) > 0 else 'Unknown'
                capabilities['gpu_memory_gb'] = int(gpu_info[1]) // 1024 if len(gpu_info) > 1 else 0
    except:
        pass
    
    # Try PyTorch for GPU detection
    if not capabilities['gpu']:
        try:
            import torch
            if torch.cuda.is_available():
                capabilities['gpu'] = True
                capabilities['gpu_name'] = torch.cuda.get_device_name(0)
                capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        except:
            pass
    
    # Get CPU info
    try:
        import psutil
        capabilities['cpu_cores'] = psutil.cpu_count()
        capabilities['system_memory_gb'] = psutil.virtual_memory().total // (1024**3)
    except:
        import os
        capabilities['cpu_cores'] = os.cpu_count() or 1
    
    return capabilities

def get_hardware_tier():
    """Determine hardware tier based on capabilities."""
    capabilities = detect_hardware_capabilities()
    
    if capabilities['gpu']:
        gpu_memory = capabilities['gpu_memory_gb']
        if gpu_memory >= 12:
            return 'gpu_high'
        elif gpu_memory >= 6:
            return 'gpu_medium'
        else:
            return 'gpu_low'
    else:
        cpu_cores = capabilities['cpu_cores']
        memory_gb = capabilities['system_memory_gb']
        
        if cpu_cores >= 8 and memory_gb >= 16:
            return 'cpu_high'
        elif cpu_cores >= 4 and memory_gb >= 8:
            return 'cpu_medium'
        else:
            return 'cpu_low'

def print_install_recommendations():
    """Print installation recommendations for the current platform."""
    try:
        from .dependency_manager import get_dependency_manager
        dm = get_dependency_manager()
        dm.print_status()
    except ImportError:
        # Fallback to basic recommendations
        import platform
        import sys
        
        print("INTV Platform Installation Recommendations")
        print("=" * 50)
        print(f"Python Version: {sys.version}")
        print(f"Platform: {platform.platform()}")
        print(f"Architecture: {platform.architecture()}")
        print(f"Machine: {platform.machine()}")
        print()
        
        print("⚠️  Enhanced dependency management not available.")
        print("For full functionality, install core dependencies:")
        print()
        print("pipx inject intv PyPDF2 python-docx requests pyyaml psutil")
        print("pipx inject intv torch transformers sentence-transformers")
        print("pipx inject intv pytesseract Pillow pdf2image")
        print()
        
        print("Optional packages:")
        print("pipx inject intv faster-whisper sounddevice soundfile  # Audio")
        print("pipx inject intv faiss-cpu chromadb  # RAG/Vector search")
        print()
        
        if platform.system() == "Linux":
            print("Linux-specific recommendations:")
            print("- Install system packages: tesseract-ocr, ffmpeg")
            print("- For GPU support: nvidia-docker2")
        elif platform.system() == "Windows":
            print("Windows-specific recommendations:")
            print("- Install Tesseract OCR from UB-Mannheim")
            print("- Install ffmpeg")
        elif platform.system() == "Darwin":
            print("macOS-specific recommendations:")
            print("- Install via Homebrew: tesseract, ffmpeg")

def main():
    """Entry point for intv-platform command"""
    print_install_recommendations()

if __name__ == "__main__":
    main()
