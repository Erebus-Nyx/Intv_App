#!/usr/bin/env python3
"""
Platform utilities for INTV package
"""

def print_install_recommendations():
    """Print installation recommendations for the current platform."""
    import platform
    import sys
    
    print("INTV Platform Installation Recommendations")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine: {platform.machine()}")
    print()
    
    print("Recommended packages for optimal performance:")
    print("- torch: PyTorch for AI models")
    print("- transformers: HuggingFace transformers")
    print("- faster-whisper: Fast transcription")
    print("- sounddevice: Audio recording")
    print("- soundfile: Audio file handling")
    print("- streamlit: Web GUI")
    print("- opencv-python: Image processing")
    print("- pytesseract: OCR")
    print()
    
    print("Optional packages:")
    print("- cloudflared: For remote tunneling")
    print("- docker: For containerized deployment")
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
