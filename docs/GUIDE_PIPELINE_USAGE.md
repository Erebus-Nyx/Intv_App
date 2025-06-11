# INTV Pipeline System - Usage Guide

## Quick Start

The INTV application now includes a comprehensive pipeline system that can process multiple input types through unified RAG and LLM workflows.

### Installation

```bash
# Basic installation
pip install -e .

# With audio support
pip install -e ".[audio]"

# With OCR support  
pip install -e ".[ocr]"

# Full installation (all features)
pip install -e ".[full]"
```

### Available Commands

After installation, you'll have these new commands available:

- `intv-pipeline` - Unified pipeline for all input types
- `intv-audio` - Audio processing utilities  
- `intv-ocr` - OCR processing utilities
- `intv-platform` - Platform compatibility checker

## Processing Different Input Types

### Documents (PDF, DOCX, TXT)

```bash
# Command line
intv-pipeline document.pdf --module adult --query "Analyze this content"

# Python API
from intv.pipeline_api import process_file
result = process_file("document.pdf", module_key="adult")
```

### Images (PNG, JPG, etc.)

```bash
# Command line with OCR
intv-pipeline image.png --module child --apply-llm

# Python API
from intv.pipeline_api import INTVPipeline
pipeline = INTVPipeline()
result = pipeline.process_image("image.png", module_key="child")
```

### Audio Files (WAV, MP3, etc.)

```bash
# Command line with transcription
intv-pipeline audio.wav --module ar --diarization

# Python API with speaker diarization
pipeline = INTVPipeline()
result = pipeline.process_audio(
    "audio.wav", 
    module_key="ar",
    enable_diarization=True
)
```

### Microphone Input

```bash
# Command line - record for 30 seconds
intv-pipeline --microphone --duration 30 --module adult --save-audio recording.wav

# Python API
from intv.pipeline_api import record_and_process
result = record_and_process(duration=30, module_key="adult")
```

## Batch Processing

```bash
# Process multiple files
intv-pipeline file1.pdf file2.png file3.wav --module adult --output results.json --format json

# Python API
pipeline = INTVPipeline()
files = ["doc1.pdf", "image1.png", "audio1.wav"]
results = pipeline.batch_process(files, module_key="adult")
```

## Output Formats

### Text Format (Human Readable)
```bash
intv-pipeline document.pdf --module adult --format text
```

### JSON Format (Machine Readable)
```bash
intv-pipeline document.pdf --module adult --format json --output results.json
```

Example JSON output:
```json
{
  "success": true,
  "input_type": "document",
  "extracted_text": "Full document text...",
  "chunks": ["chunk1", "chunk2", "chunk3"],
  "rag_result": {...},
  "llm_output": {...},
  "metadata": {
    "num_chunks": 3,
    "total_length": 1234
  }
}
```

## Available Interview Modules

List available modules:
```bash
intv-pipeline --list-modules
```

Common modules:
- `adult` - Adult interview module
- `child` - Child interview module  
- `ar` - Arkansas-specific module
- `collateral` - Collateral interview module

## Configuration

The pipeline system uses the same configuration as the main INTV application:

```yaml
# config/config.yaml

# Audio processing
enable_diarization: true
whisper_model: "faster-whisper/large-v2"
audio_sample_rate: 16000

# OCR processing  
tesseract_config: "--psm 6"
ocr_languages: ["eng"]

# RAG/LLM processing
chunk_size: 1000
chunk_overlap: 100
max_file_size_mb: 50
```

## Platform Compatibility

Check your platform's capabilities:
```bash
intv-platform
```

Install platform-specific optimizations:
```bash
# NVIDIA GPU
pip install -e ".[full-cuda]"

# AMD GPU
pip install -e ".[full-rocm]"

# Apple Silicon
pip install -e ".[full-mps]"

# CPU only (safe for all platforms)
pip install -e ".[full]"
```

## Error Handling

The pipeline system provides clear error messages for common issues:

### Missing Dependencies
```
Error: Microphone recording requires additional dependencies: sounddevice
Solution: pip install sounddevice
```

### Unsupported Files
```
Error: Unsupported input type: unknown
Solution: Use supported formats (.pdf, .wav, .png, etc.)
```

### File Size Limits
```
Error: File too large: 75.5 MB (max: 50 MB)
Solution: Increase max_file_size_mb in config or reduce file size
```

## Migration from Old CLI

### Old Usage
```bash
python src/main.py --file document.pdf --module adult
python src/main.py --audio recording.wav --module child
python src/main.py --mic --module ar
```

### New Usage
```bash
intv-pipeline document.pdf --module adult
intv-pipeline recording.wav --module child
intv-pipeline --microphone --module ar
```

## Advanced Features

### Custom Queries
```bash
intv-pipeline document.pdf --module adult --query "What are the key risk factors mentioned?"
```

### Audio with Speaker Diarization
```bash
intv-pipeline interview.wav --module adult --diarization --apply-llm
```

### Save Microphone Recordings
```bash
intv-pipeline --microphone --duration 60 --save-audio session.wav --module adult
```

### Batch Processing with Custom Output
```bash
intv-pipeline *.pdf --module adult --output batch_results.json --format json
```

## Troubleshooting

### Audio Issues
1. **No microphone detected**: Check system permissions and hardware
2. **Poor transcription**: Use higher quality audio, enable VAD
3. **Missing audio packages**: `pip install sounddevice librosa`

### OCR Issues  
1. **Poor OCR results**: Use high-resolution images
2. **Missing tesseract**: Install tesseract-ocr system package
3. **Wrong language**: Configure OCR languages in config

### Performance Issues
1. **Slow processing**: Use GPU acceleration, smaller models
2. **Memory usage**: Process files individually, reduce chunk sizes
3. **File size limits**: Increase limits in configuration

## Examples

See the `/home/nyx/intv/PIPELINE_DOCUMENTATION.md` file for comprehensive examples and detailed API documentation.

## Integration with Existing Workflows

The pipeline system is designed to work alongside existing INTV functionality:

- Use `intv` for traditional CLI workflows
- Use `intv-pipeline` for enhanced multi-modal processing
- Use the Python API for programmatic integration
- All systems share the same configuration and modules

This provides a smooth migration path while adding powerful new capabilities for processing diverse input types through unified RAG and LLM pipelines.
