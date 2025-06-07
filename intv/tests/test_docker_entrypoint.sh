[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "intv"
version = "0.2.0"
description = "INTV: Interview Automation & Document Analysis with embedded RAG/LLM and external API support"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "David Anderson", email = "david@erebusnyx.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: MIT License", 
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
    "Topic :: Office/Business",
]
requires-python = ">=3.10"
dependencies = [
    # Core dependencies
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "websockets>=11.0",
    "python-multipart>=0.0.6",
    
    # Document processing
    "PyPDF2>=3.0.0",
    "python-docx>=0.8.11",
    "pdf2image>=1.16.0",
    "pytesseract>=0.3.10",
    "Pillow>=9.0.0",
    
    # Audio processing
    "faster-whisper>=0.9.0",
    "sounddevice>=0.4.0",
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
    
    # LLM and ML
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "sentence-transformers>=2.2.0",
    "llama-cpp-python>=0.2.0",
    
    # RAG and search
    "faiss-cpu>=1.7.0",
    "chromadb>=0.4.0",
    
    # Utilities
    "requests>=2.28.0",
    "pyyaml>=6.0",
    "python-jose[cryptography]>=3.3.0",
    "psutil>=5.9.0",
    "rich>=13.0.0",
    "click>=8.0.0",
    "tqdm>=4.65.0",
    "packaging>=21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]
audio = [
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "speechrecognition>=3.10.0",
    # pyaudio is platform-specific, handle separately
]
ocr = [
    "pytesseract>=0.3.10",
    "Pillow>=9.0.0",
    "pdf2image>=1.16.0",
]
# Platform-specific GPU support
gpu-cuda = [
    "torch[cuda]>=2.0.0",
    "transformers[torch]>=4.30.0",
]
gpu-rocm = [
    "torch[rocm]>=2.0.0",  # AMD GPUs
    "transformers[torch]>=4.30.0",
]
gpu-mps = [
    "torch>=2.0.0",  # Apple Silicon (MPS backend built-in)
    "transformers[torch]>=4.30.0",
]
# Intel GPU support (when available)
gpu-intel = [
    "intel-extension-for-pytorch>=2.0.0",
    "torch>=2.0.0",
    "transformers[torch]>=4.30.0",
]
# Convenience groups
gpu = ["intv[gpu-cuda]"]  # Default to CUDA (explicit)
arm = ["intv[audio,ocr]"]  # ARM-friendly (no GPU)
raspberry-pi = ["intv[audio,ocr]"]  # Raspberry Pi
# Platform-specific full installs
full-cuda = ["intv[audio,ocr,gpu-cuda]"]
full-rocm = ["intv[audio,ocr,gpu-rocm]"] 
full-mps = ["intv[audio,ocr,gpu-mps]"]
full-intel = ["intv[audio,ocr,gpu-intel]"]
# Safe full install (no GPU conflicts)
full = ["intv[audio,ocr]"]

[project.scripts]
intv = "intv.cli_entry:main"
intv-pipeline = "intv.pipeline_cli:main"
intv-audio = "intv.audio_utils:main"
intv-ocr = "intv.ocr:main"
intv-gui = "intv.gui.app:main"
intv-platform = "intv.platform_utils:print_install_recommendations"

[tool.setuptools.packages.find]
where = ["."]
include = ["intv*"]

[tool.setuptools.package-dir]

[tool.setuptools.package-data]
intv = ["*.yaml", "*.json"]
