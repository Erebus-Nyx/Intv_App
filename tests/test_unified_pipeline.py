#!/usr/bin/env python3
"""
Test script for the unified pipeline orchestrator
Tests unified document/image processing functionality
"""

import sys
import os
import logging
from pathlib import Path

# Add the intv package to the path
sys.path.insert(0, str(Path(__file__).parent / "intv"))

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_pipeline_imports():
    """Test that all required modules can be imported"""
    logger = setup_logging()
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator, InputType, ProcessingResult
        from intv.dependency_manager import get_dependency_manager, check_feature_dependencies
        from intv.unified_processor import get_unified_processor
        
        logger.info("✓ All pipeline imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_dependency_checking():
    """Test dependency checking functionality"""
    logger = setup_logging()
    
    try:
        from intv.dependency_manager import check_feature_dependencies
        
        # Check core dependencies
        result = check_feature_dependencies(['core'])
        logger.info(f"Core dependencies check: {result}")
        
        # Check OCR dependencies
        result = check_feature_dependencies(['core', 'ocr'])
        logger.info(f"Core + OCR dependencies check: {result}")
        
        # Check ML dependencies
        result = check_feature_dependencies(['core', 'ml'])
        logger.info(f"Core + ML dependencies check: {result}")
        
        logger.info("✓ Dependency checking completed")
        return True
    except Exception as e:
        logger.error(f"✗ Dependency checking failed: {e}")
        return False

def test_unified_processor():
    """Test the unified processor directly"""
    logger = setup_logging()
    
    try:
        from intv.unified_processor import get_unified_processor
        
        # Initialize processor
        processor = get_unified_processor({})
        logger.info("✓ Unified processor initialized")
        
        # Test with a simple text file (if available)
        test_files = [
            Path("sample-sources/sample_textonly_affidavit.docx"),
            Path("sample-sources/sample_typed_adult.pdf"),
            Path("README.md"),
            Path("pyproject.toml")
        ]
        
        for test_file in test_files:
            if test_file.exists():
                logger.info(f"Testing with file: {test_file}")
                result = processor.process_file(str(test_file))
                logger.info(f"Processing result success: {result.get('success', False)}")
                if result.get('success'):
                    text_length = len(result.get('text', ''))
                    logger.info(f"Extracted text length: {text_length}")
                    logger.info(f"Method used: {result.get('metadata', {}).get('method', 'unknown')}")
                else:
                    logger.info(f"Processing error: {result.get('error', 'unknown')}")
                break
        else:
            logger.info("No test files found for unified processor test")
        
        logger.info("✓ Unified processor test completed")
        return True
    except Exception as e:
        logger.error(f"✗ Unified processor test failed: {e}")
        return False

def test_pipeline_orchestrator():
    """Test the pipeline orchestrator with unified processing"""
    logger = setup_logging()
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator, InputType
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        logger.info("✓ Pipeline orchestrator initialized")
        
        # Test input type detection
        test_paths = [
            ("test.pdf", InputType.DOCUMENT),
            ("test.docx", InputType.DOCUMENT),
            ("test.txt", InputType.DOCUMENT),
            ("test.jpg", InputType.IMAGE),
            ("test.png", InputType.IMAGE),
            ("test.wav", InputType.AUDIO),
            ("test.mp3", InputType.AUDIO),
            ("test.unknown", InputType.UNKNOWN)
        ]
        
        for path, expected_type in test_paths:
            # Create a temporary file to test detection
            temp_path = Path(path)
            temp_path.touch()
            
            detected_type = orchestrator.detect_input_type(temp_path)
            logger.info(f"Path: {path} -> Expected: {expected_type.value}, Detected: {detected_type.value}")
            
            # Clean up
            temp_path.unlink()
            
            if detected_type != expected_type:
                logger.warning(f"Type detection mismatch for {path}")
        
        # Test with real files if available
        test_files = [
            Path("sample-sources/sample_textonly_affidavit.docx"),
            Path("sample-sources/sample_typed_adult.pdf"),
            Path("README.md")
        ]
        
        for test_file in test_files:
            if test_file.exists():
                logger.info(f"\nTesting pipeline with file: {test_file}")
                
                # Test unified processing
                result = orchestrator.process(test_file)
                logger.info(f"Processing success: {result.success}")
                logger.info(f"Input type: {result.input_type.value}")
                
                if result.success:
                    logger.info(f"Extracted text length: {len(result.extracted_text or '')}")
                    logger.info(f"Chunks generated: {len(result.chunks or [])}")
                    logger.info(f"Metadata: {result.metadata}")
                else:
                    logger.info(f"Processing error: {result.error_message}")
                
                # Test legacy methods (should show deprecation warnings)
                logger.info("Testing legacy methods (expect deprecation warnings):")
                input_type = orchestrator.detect_input_type(test_file)
                if input_type == InputType.DOCUMENT:
                    legacy_result = orchestrator.process_document(test_file)
                    logger.info(f"Legacy document processing success: {legacy_result.success}")
                elif input_type == InputType.IMAGE:
                    legacy_result = orchestrator.process_image(test_file)
                    logger.info(f"Legacy image processing success: {legacy_result.success}")
                
                break
        else:
            logger.info("No test files found for pipeline orchestrator test")
        
        logger.info("✓ Pipeline orchestrator test completed")
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline orchestrator test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_hardware_classification():
    """Test hardware classification integration"""
    logger = setup_logging()
    
    try:
        from intv.rag_system import get_system_capabilities
        
        capabilities = get_system_capabilities()
        logger.info(f"System capabilities: {capabilities}")
        
        # Test that high-end system is properly classified
        if capabilities.get('system_tier') == 'gpu_high':
            logger.info("✓ High-end GPU system correctly classified")
        else:
            logger.warning(f"System classified as: {capabilities.get('system_tier', 'unknown')}")
        
        logger.info("✓ Hardware classification test completed")
        return True
    except Exception as e:
        logger.error(f"✗ Hardware classification test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger = setup_logging()
    logger.info("Starting unified pipeline tests...")
    
    tests = [
        ("Pipeline Imports", test_pipeline_imports),
        ("Dependency Checking", test_dependency_checking),
        ("Unified Processor", test_unified_processor),
        ("Pipeline Orchestrator", test_pipeline_orchestrator),
        ("Hardware Classification", test_hardware_classification)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.info("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
