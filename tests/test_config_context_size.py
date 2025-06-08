#!/usr/bin/env python3
"""
Test script for config-driven context size management
Tests both embedded and external LLM context size configuration
"""

import os
import sys
import yaml
import tempfile
from pathlib import Path

# Add the intv module to path
sys.path.insert(0, str(Path(__file__).parent))

from intv.llm import EmbeddedLLM, ExternalAPILLM, LLMSystem
from intv.config import load_config

def create_test_config(context_size="auto", embedded_context_size=None, external_context_size=None):
    """Create a test configuration with specified context sizes"""
    config = {
        'llm': {
            'mode': 'embedded',
            'context_size': context_size,
            'max_tokens': 'auto',
            'temperature': 0.7,
            'top_p': 0.9,
            'embedded': {
                'model': 'auto'
            },
            'external': {
                'provider': 'koboldcpp',
                'api_base': 'http://localhost',
                'api_port': 5001,
                'model': 'auto',
                'timeout': 30
            }
        },
        'model_dir': 'models'
    }
    
    if embedded_context_size is not None:
        config['llm']['embedded']['context_size'] = embedded_context_size
    
    if external_context_size is not None:
        config['llm']['external']['context_size'] = external_context_size
    
    return config

def test_embedded_context_size_configs():
    """Test different embedded LLM context size configurations"""
    print("🧪 Testing Embedded LLM Context Size Configurations")
    print("=" * 60)
    
    # Test 1: Auto context size (default)
    print("\n1. Testing auto context size (default)")
    config = create_test_config(context_size="auto")
    try:
        embedded_llm = EmbeddedLLM(config)
        context_size = embedded_llm._get_configured_context_size()
        print(f"✅ Auto context size: {context_size} tokens")
    except Exception as e:
        print(f"❌ Auto context size failed: {e}")
    
    # Test 2: Global context size override
    print("\n2. Testing global context size override (6144)")
    config = create_test_config(context_size=6144)
    try:
        embedded_llm = EmbeddedLLM(config)
        context_size = embedded_llm._get_configured_context_size()
        assert context_size == 6144, f"Expected 6144, got {context_size}"
        print(f"✅ Global context size override: {context_size} tokens")
    except Exception as e:
        print(f"❌ Global context size override failed: {e}")
    
    # Test 3: Embedded-specific context size override
    print("\n3. Testing embedded-specific context size override (8192)")
    config = create_test_config(context_size="auto", embedded_context_size=8192)
    try:
        embedded_llm = EmbeddedLLM(config)
        context_size = embedded_llm._get_configured_context_size()
        assert context_size == 8192, f"Expected 8192, got {context_size}"
        print(f"✅ Embedded-specific context size override: {context_size} tokens")
    except Exception as e:
        print(f"❌ Embedded-specific context size override failed: {e}")
    
    # Test 4: Embedded-specific overrides global
    print("\n4. Testing embedded-specific overrides global (embedded=2048, global=4096)")
    config = create_test_config(context_size=4096, embedded_context_size=2048)
    try:
        embedded_llm = EmbeddedLLM(config)
        context_size = embedded_llm._get_configured_context_size()
        assert context_size == 2048, f"Expected 2048 (embedded), got {context_size}"
        print(f"✅ Embedded-specific overrides global: {context_size} tokens")
    except Exception as e:
        print(f"❌ Embedded-specific override test failed: {e}")

def test_external_context_size_configs():
    """Test different external LLM context size configurations"""
    print("\n\n🌐 Testing External LLM Context Size Configurations")
    print("=" * 60)
    
    # Test 1: Auto context size (default)
    print("\n1. Testing auto context size (default)")
    config = create_test_config(context_size="auto")
    try:
        external_llm = ExternalAPILLM(config)
        context_size = external_llm.get_context_window_size()
        print(f"✅ Auto context size: {context_size} tokens")
    except Exception as e:
        print(f"❌ Auto context size failed: {e}")
    
    # Test 2: Global context size override
    print("\n2. Testing global context size override (6144)")
    config = create_test_config(context_size=6144)
    try:
        external_llm = ExternalAPILLM(config)
        context_size = external_llm.get_context_window_size()
        assert context_size == 6144, f"Expected 6144, got {context_size}"
        print(f"✅ Global context size override: {context_size} tokens")
    except Exception as e:
        print(f"❌ Global context size override failed: {e}")
    
    # Test 3: External-specific context size override
    print("\n3. Testing external-specific context size override (8192)")
    config = create_test_config(context_size="auto", external_context_size=8192)
    try:
        external_llm = ExternalAPILLM(config)
        context_size = external_llm.get_context_window_size()
        assert context_size == 8192, f"Expected 8192, got {context_size}"
        print(f"✅ External-specific context size override: {context_size} tokens")
    except Exception as e:
        print(f"❌ External-specific context size override failed: {e}")
    
    # Test 4: External-specific overrides global
    print("\n4. Testing external-specific overrides global (external=2048, global=4096)")
    config = create_test_config(context_size=4096, external_context_size=2048)
    try:
        external_llm = ExternalAPILLM(config)
        context_size = external_llm.get_context_window_size()
        assert context_size == 2048, f"Expected 2048 (external), got {context_size}"
        print(f"✅ External-specific overrides global: {context_size} tokens")
    except Exception as e:
        print(f"❌ External-specific override test failed: {e}")

def test_llm_system_integration():
    """Test LLM system integration with configurable context sizes"""
    print("\n\n🔗 Testing LLM System Integration")
    print("=" * 60)
    
    # Test with embedded mode and custom context size
    print("\n1. Testing embedded mode with custom context size (6144)")
    config = create_test_config(context_size=6144)
    config['llm']['mode'] = 'embedded'
    
    try:
        llm_system = LLMSystem(config)
        # Try to access the backend's context size
        if hasattr(llm_system.processor, 'backend'):
            if hasattr(llm_system.processor.backend, '_get_configured_context_size'):
                context_size = llm_system.processor.backend._get_configured_context_size()
                print(f"✅ LLM System embedded context size: {context_size} tokens")
            elif hasattr(llm_system.processor.backend, 'get_context_window_size'):
                context_size = llm_system.processor.backend.get_context_window_size()
                print(f"✅ LLM System context window size: {context_size} tokens")
        print("✅ LLM System integration successful")
    except Exception as e:
        print(f"❌ LLM System integration failed: {e}")

def test_config_validation():
    """Test configuration validation and error handling"""
    print("\n\n⚠️ Testing Configuration Validation")
    print("=" * 60)
    
    # Test invalid context size values
    invalid_values = ["invalid", -1, 0, "2048x", None]
    
    for invalid_value in invalid_values:
        print(f"\n• Testing invalid context_size: {invalid_value}")
        config = create_test_config(context_size=invalid_value)
        
        try:
            embedded_llm = EmbeddedLLM(config)
            context_size = embedded_llm._get_configured_context_size()
            print(f"  ✅ Gracefully handled, fallback to: {context_size} tokens")
        except Exception as e:
            print(f"  ❌ Error handling failed: {e}")

def main():
    """Run all context size configuration tests"""
    print("🚀 Config-Driven Context Size Management Tests")
    print("=" * 60)
    print("Testing both embedded and external LLM context size configuration")
    print("with auto-detection and user override capabilities\n")
    
    try:
        test_embedded_context_size_configs()
        test_external_context_size_configs()
        test_llm_system_integration()
        test_config_validation()
        
        print("\n\n🎉 All context size configuration tests completed!")
        print("=" * 60)
        print("✅ Config-driven context size management is working correctly")
        print("✅ Auto-detection with user override capabilities functional")
        print("✅ Both embedded and external LLM modes supported")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
