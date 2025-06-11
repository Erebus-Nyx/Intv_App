#!/usr/bin/env python3
"""
Debug file type classification to understand PDF/audio misclassification issue
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_file_classification():
    """Test file type classification for sample files"""
    print("🔍 Testing File Type Classification")
    print("=" * 50)
    
    # Test files
    test_files = [
        'sample-sources/sample_typed_adult.pdf',
        'sample-sources/sample_audio_child.m4a',
        'sample-sources/sample_video_child.mp4',
        'sample-sources/sample_withfields_adult.docx'
    ]
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator, InputType
        
        orchestrator = PipelineOrchestrator()
        
        for file_path in test_files:
            full_path = Path(file_path)
            if full_path.exists():
                detected_type = orchestrator.detect_input_type(full_path)
                print(f"📄 {file_path}")
                print(f"   Extension: {full_path.suffix}")
                print(f"   Detected Type: {detected_type}")
                print(f"   Expected: {'DOCUMENT' if full_path.suffix in ['.pdf', '.docx'] else 'AUDIO' if full_path.suffix in ['.m4a'] else 'VIDEO' if full_path.suffix in ['.mp4'] else 'OTHER'}")
                print()
            else:
                print(f"❌ File not found: {file_path}")
                
    except Exception as e:
        print(f"❌ Error during classification test: {e}")
        import traceback
        traceback.print_exc()

def test_pipeline_routing():
    """Test how files are routed through the pipeline"""
    print("\n🔀 Testing Pipeline Routing")
    print("=" * 50)
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator()
        
        # Test PDF file processing
        pdf_path = Path('sample-sources/sample_typed_adult.pdf')
        if pdf_path.exists():
            print(f"📄 Testing PDF routing: {pdf_path}")
            
            # Test input type detection
            input_type = orchestrator.detect_input_type(pdf_path)
            print(f"   Detected Type: {input_type}")
            
            # Test which method would be called
            if input_type.value == 'document':
                print("   ✅ Would route to: process_document_or_image()")
            elif input_type.value == 'audio':
                print("   ❌ INCORRECT: Would route to: process_audio_file()")
            elif input_type.value == 'video':
                print("   ❌ INCORRECT: Would route to: process_video_file()")
            else:
                print(f"   ⚠️  Would route to: unsupported ({input_type})")
        else:
            print("❌ PDF test file not found")
            
    except Exception as e:
        print(f"❌ Error during routing test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_file_classification()
    test_pipeline_routing()
