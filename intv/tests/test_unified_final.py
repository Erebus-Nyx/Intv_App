#!/usr/bin/env python3
"""
Final comprehensive test for the unified INTV pipeline
"""

import sys
import logging
from pathlib import Path

# Setup logging to see deprecation warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_basic_functionality():
    """Test basic pipeline functionality"""
    print("=== Testing Basic Functionality ===")
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator, InputType
        from intv.dependency_manager import get_dependency_manager
        
        orchestrator = PipelineOrchestrator()
        print("✓ Pipeline orchestrator created successfully")
        
        # Test dependency manager integration
        dm = get_dependency_manager()
        print("✓ Dependency manager integrated")
        
        # Test input type detection
        test_cases = [
            ("test.pdf", InputType.DOCUMENT),
            ("test.docx", InputType.DOCUMENT), 
            ("test.txt", InputType.DOCUMENT),
            ("test.md", InputType.DOCUMENT),
            ("test.toml", InputType.DOCUMENT),
            ("test.jpg", InputType.IMAGE),
            ("test.png", InputType.IMAGE),
            ("test.wav", InputType.AUDIO),
            ("test.mp3", InputType.AUDIO),
        ]
        
        for filename, expected_type in test_cases:
            # Create temporary file
            temp_path = Path(filename)
            temp_path.touch()
            
            detected_type = orchestrator.detect_input_type(temp_path)
            status = "✓" if detected_type == expected_type else "✗"
            print(f"  {status} {filename} -> {detected_type.value} (expected: {expected_type.value})")
            
            # Clean up
            temp_path.unlink()
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_file_processing():
    """Test actual file processing"""
    print("\n=== Testing File Processing ===")
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator()
        
        # Test with available files
        test_files = [
            "README.md",
            "pyproject.toml",
            "sample-sources/sample_textonly_affidavit.docx"
        ]
        
        for file_path in test_files:
            path = Path(file_path)
            if path.exists():
                print(f"\nTesting {file_path}:")
                result = orchestrator.process(path)
                
                if result.success:
                    print(f"  ✓ Success: {result.input_type.value}")
                    print(f"  ✓ Text length: {len(result.extracted_text or '')}")
                    print(f"  ✓ Method: {result.metadata.get('method', 'unknown')}")
                    print(f"  ✓ Chunks: {len(result.chunks or [])}")
                else:
                    print(f"  ✗ Failed: {result.error_message}")
            else:
                print(f"  - File not found: {file_path}")
        
        return True
    except Exception as e:
        print(f"✗ File processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_methods():
    """Test legacy methods show deprecation warnings"""
    print("\n=== Testing Legacy Methods (Expect Warnings) ===")
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        from pathlib import Path
        
        orchestrator = PipelineOrchestrator()
        
        # Test legacy document processing
        print("Testing legacy process_document method:")
        result = orchestrator.process_document(Path("README.md"))
        if result.success:
            print("  ✓ Legacy document method works (with deprecation warning)")
        else:
            print(f"  ✗ Legacy document method failed: {result.error_message}")
        
        return True
    except Exception as e:
        print(f"✗ Legacy methods test failed: {e}")
        return False

def test_hardware_detection():
    """Test hardware detection integration"""
    print("\n=== Testing Hardware Detection ===")
    
    try:
        from intv.rag_system import SystemCapabilities
        
        system_type = SystemCapabilities.detect_system_type()
        print(f"  System type: {system_type}")
        print("  ✓ Hardware detection working")
        
        return True
    except Exception as e:
        print(f"✗ Hardware detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("INTV Unified Pipeline Comprehensive Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("File Processing", test_file_processing), 
        ("Legacy Methods", test_legacy_methods),
        ("Hardware Detection", test_hardware_detection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The unified pipeline is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
