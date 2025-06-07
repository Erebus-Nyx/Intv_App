#!/usr/bin/env python3
"""
Test document and image processing pathway comprehensively
"""

import os
import tempfile
import subprocess
import sys
import json
from pathlib import Path

def test_document_processing():
    """Test document processing with different formats"""
    print("\n=== Testing Document Processing ===")
    
    # Create test files
    test_files = {}
    
    # Create text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a comprehensive test document.\n")
        f.write("It contains multiple paragraphs for testing.\n")
        f.write("The content should be chunked and processed through RAG.\n")
        f.write("We want to verify that JSON output works correctly.\n")
        test_files['txt'] = f.name
    
    success_count = 0
    total_tests = 0
    
    for file_type, file_path in test_files.items():
        print(f"\n--- Testing {file_type.upper()} file processing ---")
        
        # Test 1: Basic processing without module
        total_tests += 1
        print(f"Test 1: Basic processing ({file_type})")
        if test_file_processing(file_path, None, "text"):
            success_count += 1
            print("✅ Basic processing works")
        else:
            print("❌ Basic processing failed")
        
        # Test 2: JSON output format
        total_tests += 1
        print(f"Test 2: JSON output format ({file_type})")
        if test_file_processing(file_path, None, "json"):
            success_count += 1
            print("✅ JSON output works")
        else:
            print("❌ JSON output failed")
        
        # Test 3: With module (if available)
        total_tests += 1
        print(f"Test 3: With module ({file_type})")
        if test_file_processing(file_path, "adult", "json"):
            success_count += 1
            print("✅ Module processing works")
        else:
            print("❌ Module processing failed")
        
        # Clean up
        try:
            os.unlink(file_path)
        except:
            pass
    
    print(f"\n=== Document Processing Results: {success_count}/{total_tests} tests passed ===")
    return success_count == total_tests

def test_file_processing(file_path, module, output_format):
    """Test processing a single file"""
    try:
        cmd = [
            sys.executable, "-m", "intv.pipeline_cli",
            "--files", file_path,
            "--format", output_format,
            "--verbose"
        ]
        
        if module:
            cmd.extend(["--module", module])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{output_format}', delete=False) as f:
            output_file = f.name
        
        cmd.extend(["--output", output_file])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Command failed with code {result.returncode}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False
        
        # Check if output file was created
        if not os.path.exists(output_file):
            print("Output file was not created")
            return False
        
        # Validate output content
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print("Output file is empty")
            return False
        
        # For JSON format, try to parse it
        if output_format == "json":
            try:
                data = json.loads(content)
                print(f"JSON output contains {len(data)} items" if isinstance(data, list) else "JSON output is valid")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON output: {e}")
                print(f"Content: {content[:200]}...")
                return False
        
        # Clean up
        try:
            os.unlink(output_file)
        except:
            pass
        
        return True
        
    except subprocess.TimeoutExpired:
        print("Test timed out")
        return False
    except Exception as e:
        print(f"Test error: {e}")
        return False

def test_ocr_functionality():
    """Test OCR functionality with a simple image"""
    print("\n=== Testing OCR Functionality ===")
    
    # For now, just check if OCR modules are available
    try:
        import pytesseract
        import pdf2image
        print("✅ OCR dependencies available")
        
        # Test if tesseract is actually working
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
            return True
        except Exception as e:
            print(f"❌ Tesseract not properly configured: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ OCR dependencies missing: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid files"""
    print("\n=== Testing Error Handling ===")
    
    # Test with non-existent file
    cmd = [
        sys.executable, "-m", "intv.pipeline_cli",
        "--files", "nonexistent_file.txt"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("✅ Properly handles non-existent files")
            return True
        else:
            print("❌ Should have failed for non-existent file")
            return False
    except Exception as e:
        print(f"❌ Error testing error handling: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Document/Image Processing Pipeline")
    print("=" * 50)
    
    tests = [
        ("Document Processing", test_document_processing),
        ("OCR Functionality", test_ocr_functionality),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"Overall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
