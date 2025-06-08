#!/usr/bin/env python3
"""
Final verification test for INTV Pipeline Orchestrator enhancements.
Tests all completed functionality including unified processing, hardware detection,
dependency management, and backward compatibility.
"""

import sys
import os
from pathlib import Path

# Add intv to path
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_unified_processing():
    """Test the unified document/image processing method"""
    print("🔄 Testing unified processing...")
    
    from intv.pipeline_orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator()
    
    # Test with README.md (should work without dependencies)
    result = orchestrator.process_document_or_image('README.md')
    
    assert result.success, f"Processing failed: {result.error_message}"
    assert result.input_type.name == 'DOCUMENT'
    assert len(result.extracted_text) > 10000  # README is substantial
    assert len(result.chunks) > 10  # Should be chunked
    assert result.metadata['method'] == 'simple_text_read'
    
    print(f"  ✅ README.md processed: {len(result.extracted_text)} chars, {len(result.chunks)} chunks")
    
    # Test with pyproject.toml (new format support)
    result = orchestrator.process_document_or_image('pyproject.toml')
    
    assert result.success, f"TOML processing failed: {result.error_message}"
    assert len(result.extracted_text) > 1000  # pyproject.toml has content
    
    print(f"  ✅ pyproject.toml processed: {len(result.extracted_text)} chars")
    
    return True

def test_hardware_detection():
    """Test hardware detection and classification"""
    print("🔄 Testing hardware detection...")
    
    from intv.platform_utils import get_hardware_tier, detect_hardware_capabilities
    
    tier = get_hardware_tier()
    capabilities = detect_hardware_capabilities()
    
    print(f"  ✅ Hardware tier: {tier}")
    print(f"  ✅ GPU detected: {capabilities.get('gpu', False)}")
    print(f"  ✅ GPU memory: {capabilities.get('gpu_memory_gb', 'N/A')} GB")
    
    # Should detect high-end GPU correctly
    assert tier in ['gpu_high', 'gpu_medium', 'cpu_high'], f"Unexpected tier: {tier}"
    
    return True

def test_dependency_management():
    """Test dependency management functionality"""
    print("🔄 Testing dependency management...")
    
    from intv.dependency_manager import DependencyManager
    
    dep_manager = DependencyManager()
    
    # Test dependency checking
    status = dep_manager.check_dependencies()
    print(f"  ✅ Dependency status: {len(status)} groups checked")
    
    # Test injection command generation
    commands = dep_manager.get_pipx_injection_commands()
    assert len(commands) > 0, "No injection commands generated"
    print(f"  ✅ Generated {len(commands)} pipx injection commands")
    
    # Test system-specific recommendations
    recommendations = dep_manager.get_system_specific_recommendations()
    print(f"  ✅ System recommendations: {len(recommendations)} items")
    
    return True

def test_legacy_compatibility():
    """Test backward compatibility with legacy methods"""
    print("🔄 Testing legacy compatibility...")
    
    from intv.pipeline_orchestrator import PipelineOrchestrator
    import warnings
    
    orchestrator = PipelineOrchestrator()
    
    # Capture deprecation warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test legacy process_document method
        result = orchestrator.process_document('README.md')
        
        # Should have deprecation warning
        assert len(w) > 0, "No deprecation warning issued"
        assert "deprecated" in str(w[0].message).lower()
        
        # Should still work
        assert result.success, "Legacy method failed"
        
    print(f"  ✅ Legacy methods work with deprecation warnings")
    
    return True

def test_input_type_detection():
    """Test input type detection for various file formats"""
    print("🔄 Testing input type detection...")
    
    from intv.pipeline_orchestrator import PipelineOrchestrator, InputType
    
    orchestrator = PipelineOrchestrator()
    
    test_cases = [
        ('README.md', InputType.DOCUMENT),
        ('pyproject.toml', InputType.DOCUMENT),
        ('config/config.yaml', InputType.DOCUMENT),
        ('settings.json', InputType.DOCUMENT),
    ]
    
    for file_path, expected_type in test_cases:
        if os.path.exists(file_path):
            detected_type = orchestrator._detect_input_type(Path(file_path))
            assert detected_type == expected_type, f"Wrong type for {file_path}: got {detected_type}, expected {expected_type}"
            print(f"  ✅ {file_path} → {detected_type.name}")
    
    return True

def test_error_handling():
    """Test error handling for missing files and invalid inputs"""
    print("🔄 Testing error handling...")
    
    from intv.pipeline_orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator()
    
    # Test with non-existent file
    result = orchestrator.process_document_or_image('nonexistent_file.txt')
    
    assert not result.success, "Should fail for non-existent file"
    assert result.error_message is not None
    print(f"  ✅ Non-existent file handled gracefully")
    
    return True

def main():
    """Run all verification tests"""
    print("🚀 INTV Pipeline Orchestrator - Final Verification Tests")
    print("=" * 60)
    
    tests = [
        ("Unified Processing", test_unified_processing),
        ("Hardware Detection", test_hardware_detection),
        ("Dependency Management", test_dependency_management),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Input Type Detection", test_input_type_detection),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 {test_name}")
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏆 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        return True
    else:
        print("⚠️  Some tests failed - review above output")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
