# INTV: Interview Automation & Document Analysis

> **Warning**
> 
> This project is in **alpha** status and is still under active development. Not all features are fully functional or stable. Expect breaking changes, incomplete modules, and evolving APIs. Use at your own risk and see the issues tracker for known limitations.

This project provides a robust, modular system for document analysis using Retrieval Augmented Generation (RAG) and LLMs. It is designed to process TXT, PDF, and DOCX files, extract structured variables, and generate narrative outputs for various interview and assessment modules.

---

## 📊 **SYSTEM STATUS** (Updated: June 8, 2025)

### 🚨 **KNOWN ISSUES**

1. **Model Path Organization**: Minor path handling inconsistencies in cache detection
2. **Memory Usage**: Large models require significant RAM (8GB+ recommended for GPU models)
3. **Audio Pipeline Gap**: Complete audio processing system needs implementation (major feature gap)
4. **DOCX Dependencies**: Some complex DOCX files may require additional system packages

### 📈 **DEVELOPMENT PRIORITIES**

1. **Short-term**: Complete RAG-to-LLM integration testing and audio pipeline implementation
2. **Medium-term**: Web UI overhaul and multi-user support
3. **Long-term**: API restructuring and advanced AI features
4. **Ongoing**: Performance optimization and cross-platform compatibility

### ✅ **WORKING FEATURES**

#### Core Infrastructure
- ✅ **Pipeline Orchestrator**: Unified document/image processing interface
- ✅ **RAG System**: Full implementation with 438MB embedding model
- ✅ **Hardware Detection**: Automatic GPU/CPU tier detection
- ✅ **Model Caching**: Intelligent cache system prevents unnecessary re-downloads
- ✅ **Dependency Management**: Isolated pipx environment with ML dependencies
- ✅ **Configuration System**: YAML-based configuration with runtime loading

#### Document Processing
- ✅ **Text Files**: TXT, DOCX, JSON, TOML, YAML processing
- ✅ **PDF Processing**: Text extraction with OCR fallback
- ✅ **Basic OCR**: Tesseract integration (verified working)
- ✅ **File Type Detection**: Automatic input type classification
- ✅ **Chunking**: Intelligent text chunking for large documents
- ✅ **Unified Interface**: Single `process_document_or_image()` method

#### RAG & Embeddings
- ✅ **Embedding Models**: sentence-transformers integration
- ✅ **Semantic Search**: GPU-accelerated similarity search (99.30 batches/second)
- ✅ **Model Auto-Selection**: Hardware-optimized model selection
- ✅ **Cache Management**: Efficient model storage and retrieval
- ✅ **Local Model Support**: GGUF, safetensors, custom models
- ✅ **Query Processing**: Sub-second query processing for document chunks

#### Development Environment
- ✅ **Testing Suite**: Comprehensive test coverage (4/4 tests passing)
- ✅ **Error Handling**: Graceful degradation when dependencies missing
- ✅ **Logging**: Detailed logging with progress indicators
- ✅ **Git Integration**: Proper .gitignore for model files
- ✅ **Documentation**: Comprehensive status tracking with detailed technical guides
  - 📄 `docs/RAG_SYSTEM_STATUS.md`: Complete RAG implementation details (528 lines)
  - 📄 `docs/PIPELINE_ENHANCEMENT_COMPLETE.md`: Pipeline development history
  - 📄 `docs/TESTING_GUIDE.md`: Testing procedures and validation
  - 📄 `TODO.md`: Current development status and priorities (389 lines)

#### Hybrid Module System
- ✅ **Adult Module**: Complete hybrid implementation (v2.0.0)
- ✅ **Casefile Module**: Complete hybrid implementation (v2.0.0)  
- ✅ **Affidavit Module**: Complete hybrid implementation (v2.0.0)
- 🔧 **Child Module**: Basic structure exists, hybrid upgrade in progress
- ❌ **Collateral Module**: Needs creation with hybrid approach
- ❌ **AR (Alternative Response) Module**: Needs creation with hybrid approach
- ✅ **Backward Compatibility**: Legacy method support with deprecation warnings

### 🔧 **PARTIALLY WORKING**

#### Document Processing
- 🔧 **DOCX Processing**: Basic support, needs enhanced dependency handling
- 🔧 **Image Processing**: Framework exists, needs ML dependency completion
- 🔧 **Large File Processing**: Works but needs optimization for memory usage

#### Module System
- 🔧 **Child Module**: Basic structure exists, needs hybrid upgrade completion
- 🔧 **Configuration Hot-Reload**: Framework exists, needs full implementation

#### LLM Integration
- 🔧 **RAG-to-LLM Pipeline**: RAG system complete, LLM tunnel needs integration testing
- 🔧 **Multiple LLM Providers**: Basic support, needs comprehensive testing

### ❌ **NOT WORKING / MISSING**

#### Critical Missing Features
- ❌ **Complete Audio Pipeline**: Major implementation needed
- ❌ **Advanced OCR**: Preprocessing, multi-language, quality enhancement
- ❌ **Dynamic Module Creation**: Needs creation with that can be implemented without code modification

#### Audio Processing (Major Gap)
- ❌ **Audio Transcription**: Pipeline not implemented
- ❌ **Speaker Diarization**: Not implemented
- ❌ **Real-time Audio Streaming**: Not implemented
- ❌ **Voice Activity Detection**: Not implemented
- ❌ **Audio Quality Enhancement**: Not implemented

#### User Interface
- ❌ **Web UI**: Partially functional, needs major fixes
- ❌ **Interactive CLI**: Basic CLI exists, needs enhancement
- ❌ **Real-time Progress**: Limited progress indicators
- ❌ **File Upload Interface**: Needs implementation

#### API & Integration
- ❌ **API Restructuring**: Endpoints need reorganization
- ❌ **External App Support**: Integration framework needs implementation
- ❌ **WebSocket Support**: Real-time communication needs work
- ❌ **Authentication System**: Security features missing

#### Performance & Scalability
- ❌ **Multi-user Support**: Single-user design currently
- ❌ **Concurrent Processing**: No job queuing system
- ❌ **Resource Management**: Basic resource handling only
- ❌ **Performance Optimization**: Large file streaming needed

### 🎯 **PERFORMANCE METRICS**

#### RAG System Performance by Hardware Tier

**GPU High-End (RTX 4070 Ti SUPER - gpu_high)** *(Tested)*
- **Model**: 438MB multi-qa-mpnet-base-dot-v1
- **Initialization**: ~26 seconds (cached model)
- **Query Speed**: 0.02-0.21 seconds for 3-5 chunks (GPU accelerated)
- **Processing Rate**: 6-10 chunks/second (GPU optimized)
- **System Memory**: 1.2GB RAM
- **GPU Memory**: 427MB VRAM

**CPU-Only Systems (Intel I9-14900K - cpu_medium)** *(Tested)*
- **Model**: Same 438MB model (no auto-downscaling detected)
- **Initialization**: ~26 seconds (cached model, CPU mode)
- **Query Speed**: 0.3-2.5 seconds for 3-5 chunks
- **Processing Rate**: 6-15 chunks/second
- **Memory Usage**: 2.8GB RAM (CPU processing)

**Raspberry Pi / ARM Systems (cpu_low)**
- **Model**: Minimal embedding models (50-150MB)
- **Initialization**: 2-5 minutes (first run)
- **Query Speed**: 5-15 seconds for 5 chunks
- **Embedding Speed**: 1-3 batches/second
- **Memory Usage**: 1-2GB RAM
- **Storage**: 500MB+ for minimal models

#### System Requirements by Deployment Type

**High-Performance (Desktop/Server)**
- **RAM**: 8GB+ recommended for GPU models
- **Storage**: 2GB+ for models and cache
- **GPU**: NVIDIA RTX series recommended
- **CPU**: Multi-core Intel/AMD for CPU fallback

**Minimal (Raspberry Pi/ARM)**
- **RAM**: 2GB+ (4GB+ recommended for Pi 4)
- **Storage**: 500MB+ for minimal models
- **CPU**: ARM Cortex-A72+ (Pi 4 or newer)
- **GPU**: Not required, CPU-only processing
- **Note**: Automatic model downscaling for ARM compatibility


---

## 📦 Installation

INTV uses a comprehensive dependency system with optional feature groups for different deployment scenarios. The recommended installation method is using **pipx** for isolated global CLI access.

### Quick Install (Recommended)

Install pipx if you haven't already:
```sh
# Linux/macOS
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Windows
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

**Base Installation** (CPU-only, no GPU acceleration):
```sh
pipx install intv
```

**GPU-Accelerated Installation** (NVIDIA CUDA):
```sh
pipx install "intv[full-cuda]"
```

**Platform-Specific Installations**:
```sh
# AMD GPUs (ROCm)
pipx install "intv[full-rocm]"

# Apple Silicon (MPS)
pipx install "intv[full-mps]"

# Intel GPUs
pipx install "intv[full-intel]"

# Raspberry Pi / ARM systems
pipx install "intv[raspberry-pi]"

# CPU-only with all features (no GPU conflicts)
pipx install "intv[full]"
```

### System Dependencies

These are **native system libraries** that provide binary executables and can't be installed through pipx/pip. Python packages depend on these but can't bundle them due to platform differences, licensing, and size constraints.

**Linux (Debian/Ubuntu)**:
```sh
sudo apt update && sudo apt install -y tesseract-ocr poppler-utils
```

**macOS (Homebrew)**:
```sh
brew install tesseract poppler
```

**Windows**:
- **Tesseract-OCR**: [Download installer](https://github.com/tesseract-ocr/tesseract/wiki)
- **Poppler**: [Download binaries](http://blog.alivate.com.au/poppler-utils-windows/), add `bin/` to your PATH

**Why these can't be in pipx:**
- **`tesseract-ocr`**: Native C++ OCR engine with system libraries
- **`poppler-utils`**: Native C++ PDF tools required by `pdf2image` Python package
- **`portaudio`**: Native C audio library for real-time audio processing

### Optional Dependencies

**Audio Processing** (if using audio features):
```sh
# Linux
sudo apt install -y portaudio19-dev

# macOS
brew install portaudio

# Windows: Usually works out of the box
```
*Note: `portaudio` provides native C libraries for real-time audio - Python's `sounddevice` depends on it*

**Cloudflare Tunnel** (handled automatically):
- INTV includes a `cloudflared` binary (`scripts/cloudflared-linux-amd64`)
- No separate installation required - works out of the box
- Automatically detects system-installed `cloudflared` if available
- Falls back to included binary if not found in PATH

### How pipx Handles Dependencies

pipx automatically installs the **Python packages** that interface with these system libraries:

```
System Library → Python Package (installed by pipx)
===============================================
tesseract-ocr  → pytesseract>=0.3.10
poppler-utils  → pdf2image>=1.16.0  
portaudio      → sounddevice>=0.4.0
```

**Note:** `cloudflared` is included as a bundled binary (`scripts/cloudflared-linux-amd64`) - no separate installation needed.

The Python packages are just **wrappers** that call the native system binaries, which is why both layers are required.

### Verify Installation

Check that INTV is properly installed:
```sh
intv --version
intv-platform  # Shows recommended installation for your system
```

---

## 🚀 Quick Start

### Command Line Usage

After installation with pipx, INTV commands are available globally:

```sh
# Process a document
intv --file document.pdf --type pdf

# Use CPU-only mode (disables GPU acceleration)
intv --file document.pdf --type pdf --cpu

# Process with custom config
intv --file document.pdf --config custom-config.yaml

# Run the web interface
intv --gui

# Process audio files
intv-audio --file recording.wav --output transcript.txt

# OCR processing
intv-ocr --file scanned-document.pdf --output text-output.txt

# Check system compatibility
intv-platform
```

### Web Interface

Start the web server:
```sh
# Local access only
intv --gui

# With Cloudflare tunnel (public access)
intv --gui --cloudflare
```

- **Local access**: [http://localhost:3773](http://localhost:3773)
- **Cloudflare tunnel**: Public URL shown in terminal output

### Docker Deployment

For containerized deployment, use the provided Docker Compose setup:

```sh
# Clone repository for Docker files
git clone <repository-url>
cd intv

# GPU deployment
docker-compose up --build -d

# CPU-only deployment
docker-compose -f docker-compose.cpu.yml up --build -d
```
  ```
- For CPU-only, change the Dockerfile in `docker-compose.yml` to `docker/Dockerfile.cpu`.
- The app will be available at [http://localhost:3773](http://localhost:3773)
- To enable Cloudflare tunnel in Docker, set `USE_CLOUDFLARE_TUNNEL=true` in `.env`.

---

## License
See `LICENSE` for details.

---

## Credits
- Inspired by best practices from open-source LLM, RAG, and document automation projects.
- Cloudflare integration modeled after KoboldCpp and similar projects.

---
