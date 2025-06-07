# INTV Pipeline System Documentation

## Overview

The INTV Pipeline System provides a unified interface for processing different types of input through Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) pipelines. It supports:

- **Documents** (PDF, DOCX, TXT, Markdown) → RAG processing
- **Images** (PNG, JPG, etc.) → OCR → RAG processing  
- **Audio Files** (WAV, MP3, etc.) → Transcription → RAG processing
- **Microphone Input** → Real-time transcription → RAG processing

## Architecture

### Core Components

1. **Pipeline Orchestrator** (`pipeline_orchestrator.py`)
   - Central routing and processing logic
   - Input type detection
   - Pipeline coordination

2. **Pipeline API** (`pipeline_api.py`)
   - High-level Python API
   - Convenient wrapper functions
   - Batch processing support

3. **Pipeline CLI** (`pipeline_cli.py`)
   - Command-line interface
   - Support for single files and batch processing
   - Integration with existing INTV modules

### Processing Flow

```
Input File/Stream → Type Detection → Appropriate Pipeline → RAG/LLM → Results
```

#### Supported Pipelines

1. **Document Pipeline**: File → Text Extraction → Chunking → RAG → LLM
2. **Image Pipeline**: File → OCR → Text Extraction → Chunking → RAG → LLM  
3. **Audio Pipeline**: File → Transcription → (Optional Diarization) → Chunking → RAG → LLM
4. **Microphone Pipeline**: Stream → Real-time Transcription → Chunking → RAG → LLM

## Installation

### Basic Installation
```bash
pip install -e .
```

### With Audio Support
```bash
pip install -e ".[audio]"
```

### With OCR Support
```bash
pip install -e ".[ocr]"  
```

### Full Installation (all features)
```bash
pip install -e ".[full]"
```

### Platform-specific GPU support
```bash
# NVIDIA CUDA
pip install -e ".[full-cuda]"

# AMD ROCm  
pip install -e ".[full-rocm]"

# Apple Silicon MPS
pip install -e ".[full-mps]"

# Intel GPU
pip install -e ".[full-intel]"
```

## Usage

### Command Line Interface

#### Process a single file
```bash
intv-pipeline document.pdf --module adult --query "Analyze this content"
intv-pipeline image.png --module child --apply-llm
intv-pipeline audio.wav --module ar --diarization
```

#### Record from microphone
```bash
intv-pipeline --microphone --duration 30 --module adult --save-audio recording.wav
```

#### Batch processing
```bash
intv-pipeline file1.pdf file2.png file3.wav --module adult --output results.json --format json
```

#### List available modules
```bash
intv-pipeline --list-modules
```

### Python API

#### Simple file processing
```python
from intv.pipeline_api import process_file

# Process any supported file type
result = process_file("document.pdf", module_key="adult")
print(f"Success: {result['success']}")
print(f"Text length: {len(result['extracted_text'])}")
```

#### Microphone recording
```python
from intv.pipeline_api import record_and_process

# Record for 10 seconds and process
result = record_and_process(duration=10, module_key="adult")
print(f"Transcript: {result['transcript']}")
```

#### Advanced usage with INTVPipeline class
```python
from intv.pipeline_api import INTVPipeline

# Initialize pipeline
pipeline = INTVPipeline(config_path="config/config.yaml")

# Process document
doc_result = pipeline.process_document(
    "report.pdf", 
    module_key="adult",
    query="Summarize key findings",
    apply_llm=True
)

# Process image with OCR
img_result = pipeline.process_image(
    "screenshot.png",
    module_key="child", 
    apply_llm=True
)

# Process audio with diarization
audio_result = pipeline.process_audio(
    "interview.wav",
    module_key="ar",
    enable_diarization=True,
    apply_llm=True
)

# Record from microphone
mic_result = pipeline.process_microphone(
    duration=30,
    module_key="adult",
    save_audio="recording.wav",
    enable_diarization=True
)

# Batch processing
files = ["doc1.pdf", "image1.png", "audio1.wav"]
batch_results = pipeline.batch_process(files, module_key="adult")
```

#### Validation and file checking
```python
# Check if file is supported
validation = pipeline.validate_file("unknown_file.xyz")
if validation['valid']:
    result = pipeline.process_document(file_path)
else:
    print(f"Error: {validation['error']}")

# Get supported formats
formats = pipeline.get_supported_formats()
print(f"Supported documents: {formats['documents']}")

# Detect input type
input_type = pipeline.detect_input_type("mystery_file.pdf")
print(f"Detected type: {input_type}")
```

### Integration with Existing CLI

The pipeline system integrates with the existing INTV CLI:

```bash
# Traditional CLI (still works)
python src/main.py --file document.pdf --module adult

# New pipeline CLI (enhanced features)
intv-pipeline document.pdf --module adult --query "Custom query"
```

## Configuration

### Audio Processing Settings

```yaml
# config/config.yaml
enable_diarization: true
whisper_model: "faster-whisper/large-v2"  
audio_sample_rate: 16000
enable_vad: true
vad_window_size: 512
vad_speech_pad_ms: 50
vad_min_segment_ms: 500
```

### OCR Settings

```yaml
# OCR configuration
tesseract_config: "--psm 6"
ocr_languages: ["eng"]
image_preprocessing: true
```

### RAG/LLM Settings

```yaml
# Chunking settings
chunk_size: 1000
chunk_overlap: 100

# LLM settings  
max_file_size_mb: 50
enable_retrieval: true
retrieval_top_k: 5
```

## Output Formats

### Text Format (default)
Human-readable output with sections for:
- Processing status
- Input type and metadata  
- Extracted text preview
- Transcript segments (for audio)
- RAG and LLM results

### JSON Format
Structured output suitable for programmatic use:

```json
{
  "success": true,
  "input_type": "audio_file", 
  "extracted_text": "Full transcript...",
  "transcript": "Full transcript...",
  "chunks": ["chunk1", "chunk2"],
  "segments": [
    {"text": "Hello", "start": 0.0, "end": 1.2},
    {"text": "World", "start": 1.2, "end": 2.5}
  ],
  "rag_result": {...},
  "llm_output": {...},
  "metadata": {
    "num_segments": 2,
    "num_chunks": 5,
    "transcript_length": 1234,
    "has_diarization": true
  }
}
```

## Error Handling

The pipeline system provides comprehensive error handling:

### Common Errors

1. **Missing Dependencies**
   ```
   Error: Microphone recording requires additional dependencies: sounddevice
   Solution: pip install sounddevice
   ```

2. **Unsupported File Format**
   ```
   Error: Unsupported input type: unknown
   Solution: Use supported formats (.pdf, .wav, .png, etc.)
   ```

3. **File Too Large**
   ```
   Error: File too large: 75.5 MB (max: 50 MB)
   Solution: Reduce file size or increase max_file_size_mb in config
   ```

4. **OCR Failed**
   ```
   Error: No text could be extracted from image
   Solution: Check image quality, ensure tesseract is installed
   ```

### Graceful Degradation

- If diarization fails, continues with transcription only
- If LLM processing fails, returns extracted text and chunks
- Missing optional dependencies result in clear error messages

## Advanced Features

### Real-time Processing

For microphone input, the system supports:
- Real-time audio capture
- Streaming transcription
- Optional speaker diarization
- Automatic saving of audio and transcripts

### Platform Compatibility

The pipeline system includes platform detection and optimization:

```bash
# Check platform compatibility
intv-platform

# Install platform-specific optimizations
pip install -e ".[full-cuda]"  # NVIDIA
pip install -e ".[full-rocm]"  # AMD
pip install -e ".[full-mps]"   # Apple Silicon
```

### Extensibility

The system is designed for easy extension:

1. **Custom Input Types**: Add new `InputType` enum values and processing methods
2. **Custom Pipelines**: Implement new processing pipelines in `PipelineOrchestrator`
3. **Custom Modules**: Add new interview modules in `src/modules/`
4. **Custom Output Formats**: Extend the CLI or API to support new output formats

## Troubleshooting

### Audio Issues

1. **No microphone detected**
   - Check system audio permissions
   - Verify microphone hardware
   - Install platform-specific audio drivers

2. **Poor transcription quality**
   - Use higher quality audio (16kHz minimum)
   - Enable VAD (Voice Activity Detection)
   - Try different Whisper models

### OCR Issues

1. **Poor OCR results**
   - Ensure high-resolution images
   - Check image format compatibility
   - Verify tesseract installation

2. **Missing text from images**
   - Try different tesseract PSM modes
   - Enable image preprocessing
   - Check language settings

### Performance Optimization

1. **Slow processing**
   - Use GPU acceleration when available
   - Reduce chunk sizes for faster processing
   - Use smaller Whisper models for audio

2. **Memory usage**
   - Process files individually instead of batch
   - Reduce max file size limits
   - Use streaming for long audio files

## Examples

See `example_api_usage.py` for complete examples of:
- Processing different file types
- Batch processing workflows
- Error handling patterns
- Configuration management
- Integration with existing systems

## API Reference

For detailed API documentation, see the docstrings in:
- `pipeline_orchestrator.py` - Core processing logic
- `pipeline_api.py` - High-level Python API  
- `pipeline_cli.py` - Command-line interface

## Migration Guide

### From Existing INTV CLI

Old usage:
```bash
python src/main.py --file document.pdf --module adult
python src/main.py --audio recording.wav --module child
python src/main.py --mic --module ar
```

New usage:
```bash
intv-pipeline document.pdf --module adult
intv-pipeline recording.wav --module child  
intv-pipeline --microphone --module ar
```

### From Direct Module Usage

Old usage:
```python
from intv.rag import chunk_document
from intv.llm import rag_llm_pipeline

chunks = chunk_document("doc.pdf")
result = rag_llm_pipeline("doc.pdf", "adult")
```

New usage:
```python
from intv.pipeline_api import process_file

result = process_file("doc.pdf", module_key="adult")
```

The new API provides the same functionality with better error handling, input type detection, and unified processing across all file types.
