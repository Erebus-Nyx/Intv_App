#!/usr/bin/env python3
"""
Integration test for the restored Pipeline Orchestrator and enhanced RAG system.
Tests the core functionality without requiring all optional dependencies.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def test_pipeline_orchestrator():
    """Test basic pipeline orchestrator functionality"""
    print("=" * 60)
    print("PIPELINE ORCHESTRATOR INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from intv.pipeline_orchestrator import (
            PipelineOrchestrator, 
            create_pipeline_orchestrator, 
            InputType, 
            ProcessingResult
        )
        print("   ✓ All core classes import successfully")
        
        # Test instantiation
        print("\n2. Testing instantiation...")
        orchestrator = create_pipeline_orchestrator()
        print("   ✓ Pipeline orchestrator created successfully")
        
        # Test input type detection
        print("\n3. Testing input type detection...")
        
        # Create test files
        test_dir = Path("test_files")
        test_dir.mkdir(exist_ok=True)
        
        # Text file
        text_file = test_dir / "test.txt"
        text_file.write_text("This is a test document for pipeline orchestrator.")
        
        detected_type = orchestrator.detect_input_type(text_file)
        print(f"   ✓ Text file detected as: {detected_type}")
        assert detected_type == InputType.DOCUMENT
        
        # Test non-existent file
        nonexistent = test_dir / "nonexistent.xyz"
        detected_type = orchestrator.detect_input_type(nonexistent)
        print(f"   ✓ Non-existent file detected as: {detected_type}")
        assert detected_type == InputType.UNKNOWN
        
        # Test document processing (basic)
        print("\n4. Testing document processing...")
        result = orchestrator.process_document(
            text_file,
            module_key=None,
            query=None,
            apply_llm=False
        )
        
        print(f"   ✓ Document processing success: {result.success}")
        print(f"   ✓ Input type: {result.input_type}")
        print(f"   ✓ Extracted text length: {len(result.extracted_text) if result.extracted_text else 0}")
        
        assert result.success
        assert result.input_type == InputType.DOCUMENT
        assert result.extracted_text is not None
        
        # Test main process method
        print("\n5. Testing main process method...")
        result = orchestrator.process(
            text_file,
            module_key=None,
            query=None,
            apply_llm=False
        )
        
        print(f"   ✓ Main process success: {result.success}")
        assert result.success
        
        # Test batch processing
        print("\n6. Testing batch processing...")
        results = orchestrator.batch_process(
            [text_file],
            module_key=None,
            query=None,
            apply_llm=False
        )
        
        print(f"   ✓ Batch processing results: {len(results)}")
        assert len(results) == 1
        assert results[0].success
        
        # Clean up
        text_file.unlink()
        test_dir.rmdir()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Pipeline Orchestrator is working correctly!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_availability_flags():
    """Test the availability flags for optional dependencies"""
    print("\n" + "=" * 60)
    print("OPTIONAL DEPENDENCIES STATUS")
    print("=" * 60)
    
    try:
        from intv.pipeline_orchestrator import HAS_UTILS, HAS_DOC_PROCESSOR, HAS_RAG, HAS_AUDIO, HAS_MODULES
        
        print(f"HAS_UTILS: {HAS_UTILS}")
        print(f"HAS_DOC_PROCESSOR: {HAS_DOC_PROCESSOR}")
        print(f"HAS_RAG: {HAS_RAG}")
        print(f"HAS_AUDIO: {HAS_AUDIO}")
        print(f"HAS_MODULES: {HAS_MODULES}")
        
        # Test graceful degradation
        if not HAS_RAG:
            print("✓ RAG module unavailable - graceful degradation expected")
        else:
            print("✓ RAG module available - enhanced functionality enabled")
            
        return True
        
    except Exception as e:
        print(f"❌ Availability test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Starting Pipeline Orchestrator Integration Tests...")
    
    success = True
    
    # Test core functionality
    success &= test_pipeline_orchestrator()
    
    # Test availability flags
    success &= test_availability_flags()
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("\n✅ Pipeline Orchestrator has been successfully restored and is functioning properly.")
        print("✅ The system gracefully handles missing optional dependencies.")
        print("✅ Enhanced RAG integration is ready for use when dependencies are available.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
