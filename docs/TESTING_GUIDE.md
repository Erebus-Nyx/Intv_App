# INTV Testing Guide

## Overview

This guide documents the INTV test suite, organized by functionality and complexity. All tests are located in the `/tests/` directory and should be run using the pipx environment to ensure proper dependencies.

## Running Tests

### Prerequisites
```bash
# Ensure INTV is installed with all dependencies
pipx install -e .
pipx inject intv pyannote.audio pyannote.core pytesseract Pillow pdf2image
```

### Basic Test Execution
```bash
# Run from the pipx environment
pipx run --spec . python -m pytest tests/

# Or run specific tests
pipx run --spec . python tests/test_comprehensive_workflow.py
```

## Test Categories

### 🚀 **Comprehensive Workflow Tests**

#### `test_comprehensive_workflow.py` ⭐ **PRIMARY TEST**
**Purpose**: Complete end-to-end validation of the entire INTV pipeline
**Coverage**:
- PDF processing with adult model (`sample_typed_adult.pdf`)
- Audio transcription and analysis (`sample_audio_child.m4a`)
- Video audio extraction (`sample_video_child.mp4`)
- Word document processing (`sample_withfields_adult.docx`)
- RAG-enhanced context retrieval
- LLM summary generation
- Policy-adherent analysis
- Output formatting and caching

**How to Run**:
```bash
cd /home/nyx/intv
pipx run --spec . python tests/test_comprehensive_workflow.py
```

**Expected Results**: 7/7 tests passing with full pipeline validation

---

### 🎵 **Audio Processing Tests**

#### `test_audio_pipeline_complete.py`
**Purpose**: Verify complete audio processing workflow
**Coverage**:
- Audio file processing and RAG integration
- Transcription storage in output directory
- Complete pipeline: Audio → Transcription → VAD → Diarization → RAG → LLM

**How to Run**:
```bash
pipx run --spec . python tests/test_audio_pipeline_complete.py
```

#### `test_audio_verification.py`
**Purpose**: Validate audio processing components individually
**Coverage**:
- FastWhisper transcription
- VAD (Voice Activity Detection)
- Speaker diarization
- Audio format conversion (M4A → WAV)

#### `test_audio_focused.py`
**Purpose**: Focused audio pipeline testing with real files
**Coverage**:
- Sample audio file processing
- Audio system capabilities detection
- Enhanced audio processing features

#### `test_audio_utils.py`
**Purpose**: Unit tests for audio utility functions
**Coverage**:
- Audio normalization
- File loading and saving
- Audio feature computation
- Silence detection

---

### 📊 **Performance Tests**

#### `test_performance_complete.py`
**Purpose**: Compare GPU vs CPU performance across the system
**Coverage**:
- RAG system performance benchmarking
- GPU vs CPU mode comparison
- Memory usage analysis
- Processing time measurements

#### `test_performance_cpu.py`
**Purpose**: CPU-only performance validation
**Coverage**:
- CPU-optimized model performance
- Memory efficiency on CPU systems
- Processing benchmarks for CPU-only deployments

#### `test_performance_gpu.py`
**Purpose**: GPU acceleration validation
**Coverage**:
- CUDA/GPU utilization testing
- GPU memory optimization
- Accelerated model performance

---

### 🔧 **Component Tests**

#### `test_config.py`
**Purpose**: Configuration system validation
**Coverage**:
- YAML configuration loading
- Environment variable handling
- Model selection logic
- Hardware detection configuration

#### `test_rag_llm_integration.py`
**Purpose**: RAG and LLM integration testing
**Coverage**:
- Embedded vs external RAG modes
- LLM integration with context
- Vector search functionality
- Hybrid processing workflows

#### `test_unified_pipeline.py`
**Purpose**: Unified processing interface validation
**Coverage**:
- Document type detection
- Processing workflow orchestration
- Output standardization

#### `test_cli_and_pipeline.py`
**Purpose**: CLI interface and pipeline integration
**Coverage**:
- Command-line argument processing
- Pipeline execution via CLI
- Error handling and user feedback

---

### 🧪 **Specialized Tests**

#### `test_model_detection.py`
**Purpose**: Hardware and model detection validation
**Coverage**:
- GPU detection and configuration
- Optimal model selection
- Hardware capability assessment

#### `test_auto_context_detection.py`
**Purpose**: Automatic context and processing detection
**Coverage**:
- Content type inference
- Processing strategy selection
- Context-aware optimization

#### `test_server_utils.py`
**Purpose**: Server and utility functions testing
**Coverage**:
- API endpoint functionality
- Server utility functions
- Network and communication features

---

## Test Execution Strategies

### Quick Validation
```bash
# Run core functionality tests
pipx run --spec . python tests/test_comprehensive_workflow.py
pipx run --spec . python tests/test_audio_pipeline_complete.py
```

### Full Test Suite
```bash
# Run all tests with pytest
cd /home/nyx/intv
pipx run --spec . python -m pytest tests/ -v
```

### Performance Benchmarking
```bash
# Run performance tests
pipx run --spec . python tests/test_performance_complete.py
pipx run --spec . python tests/test_performance_cpu.py
pipx run --spec . python tests/test_performance_gpu.py
```

### Audio System Validation
```bash
# Comprehensive audio testing
pipx run --spec . python tests/test_audio_verification.py
pipx run --spec . python tests/test_audio_pipeline_complete.py
pipx run --spec . python tests/test_audio_focused.py
```

## Expected Test Results

### ✅ **Success Indicators**
- All tests pass without errors
- Audio dependencies properly detected
- Document processing successful
- RAG system operational
- LLM integration functional

### ⚠️ **Common Issues**
- **Missing Dependencies**: Install with `pipx inject intv <package>`
- **Environment Issues**: Always run tests via `pipx run --spec .`
- **File Permissions**: Ensure sample files are accessible
- **GPU Memory**: Reduce batch sizes if GPU memory insufficient

## Sample Files Required

Ensure these sample files exist in `/home/nyx/intv/sample-sources/`:
- `sample_typed_adult.pdf` - PDF document for OCR testing
- `sample_audio_child.m4a` - Audio file for transcription testing
- `sample_video_child.mp4` - Video file for audio extraction
- `sample_withfields_adult.docx` - Word document with form fields

## Debugging Failed Tests

### Check Dependencies
```bash
pipx run --spec . python -c "import torch, numpy, pyannote; print('All deps OK')"
```

### Verify Installation
```bash
pipx list | grep intv
pipx show intv
```

### Run with Verbose Output
```bash
pipx run --spec . python tests/test_comprehensive_workflow.py --verbose
```

## Test Development Guidelines

### Adding New Tests
1. Place in appropriate category subdirectory under `/tests/`
2. Follow naming convention: `test_<functionality>_<type>.py`
3. Include docstring with purpose and coverage
4. Ensure compatibility with pipx environment
5. Add entry to this testing guide

### Test Structure
```python
#!/usr/bin/env python3
"""
Test Description and Purpose
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_functionality():
    """Test specific functionality"""
    # Test implementation
    pass

if __name__ == "__main__":
    # Direct execution logic
    pass
```

---

**Last Updated**: June 9, 2025  
**INTV Version**: 0.2.5
