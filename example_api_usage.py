#!/usr/bin/env python3
"""
Example showing how the modular entry points work for API usage
"""

def process_document_pipeline():
    """Example: Full document processing pipeline"""
    import subprocess
    import sys
    
    # Check what's available
    try:
        # OCR stage
        result = subprocess.run([
            'intv-ocr', 'document.pdf', '--config', 'config.yaml'
        ], capture_output=True, text=True, check=True)
        ocr_text = result.stdout
        
        # Audio stage (if document has audio)
        result = subprocess.run([
            'intv-audio', 'audio.wav', '--config', 'config.yaml'  
        ], capture_output=True, text=True, check=True)
        audio_text = result.stdout
        
        # Main processing
        result = subprocess.run([
            'intv', '--text', ocr_text, '--audio', audio_text
        ], capture_output=True, text=True, check=True)
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        if "dependencies not installed" in e.stdout:
            print(f"Missing dependencies: {e.stdout}")
            print("Install with: pipx install intv[full]")
        else:
            print(f"Processing failed: {e}")
        return None

def programmatic_usage():
    """Example: Direct module usage (when installed with dependencies)"""
    try:
        from intv.ocr import ocr_file
        from intv.audio_utils import process_audio
        from intv.config import load_config
        
        config = load_config('config.yaml')
        
        # Direct function calls
        ocr_result = ocr_file('document.pdf', config)
        audio_result = process_audio('audio.wav', config) 
        
        return {"ocr": ocr_result, "audio": audio_result}
        
    except ImportError as e:
        print(f"Required dependencies not available: {e}")
        return None

if __name__ == "__main__":
    print("Pipeline usage:")
    pipeline_result = process_document_pipeline()
    
    print("\nProgrammatic usage:")
    programmatic_result = programmatic_usage()
