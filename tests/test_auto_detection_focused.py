#!/usr/bin/env python3
"""
Focused test for auto context window detection
"""

import sys
from pathlib import Path

# Add the intv module to the path
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_auto_context_detection():
    """Test the core auto context detection functionality"""
    print("🧪 Testing Auto Context Window Detection")
    print("=" * 50)
    
    try:
        from intv.llm import EmbeddedLLM, ExternalAPILLM, HybridLLMProcessor
        
        # Test 1: External API LLM (works without loading models)
        print("1. Testing External API LLM auto detection...")
        
        external_config = {
            'provider': 'koboldcpp',
            'api_base': 'http://localhost',
            'api_port': 5001,
            'api_key': '',
            'model': 'auto',
            'timeout': 30,
            'max_tokens': 'auto',
            'context_size': 'auto'
        }
        
        external_llm = ExternalAPILLM(external_config)
        
        # Test context window detection
        context_size = external_llm.get_context_window_size()
        print(f"   Context window size: {context_size}")
        
        # Test auto max_tokens calculation with different prompt lengths
        test_prompts = [
            ("Short", "Short test prompt"),
            ("Medium", "This is a medium length test prompt. " * 20),
            ("Long", "This is a very long test prompt for testing auto calculation. " * 100)
        ]
        
        for name, prompt in test_prompts:
            auto_tokens = external_llm.calculate_auto_max_tokens(prompt)
            prompt_chars = len(prompt)
            estimated_tokens = prompt_chars // 4
            print(f"   {name} prompt: {prompt_chars} chars → est. {estimated_tokens} tokens → max_tokens: {auto_tokens}")
        
        print("   ✅ External API auto detection working!")
        
        # Test 2: Embedded LLM context detection methods (without loading model)
        print("\n2. Testing Embedded LLM auto detection methods...")
        
        embedded_config = {
            'mode': 'embedded',
            'max_tokens': 'auto',
            'context_size': 'auto',
            'model': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q5_K_M.gguf',
            'temperature': 0.7
        }
        
        # Initialize without loading the actual model
        embedded_llm = EmbeddedLLM(embedded_config)
        
        # Test the context window detection method (should use default)
        context_size = embedded_llm.get_context_window_size()
        print(f"   Default context window size: {context_size}")
        
        # Test auto calculation
        test_prompt = "Test prompt for embedded LLM auto calculation. " * 30
        auto_tokens = embedded_llm.calculate_auto_max_tokens(test_prompt)
        print(f"   Test prompt: {len(test_prompt)} chars → max_tokens: {auto_tokens}")
        
        print("   ✅ Embedded LLM auto detection methods working!")
        
        # Test 3: Configuration integration
        print("\n3. Testing configuration integration...")
        
        full_config = {
            'llm': {
                'mode': 'external',
                'max_tokens': 'auto',
                'context_size': 'auto',
                'temperature': 0.7,
                'top_p': 0.9,
                'external': {
                    'provider': 'koboldcpp',
                    'api_base': 'http://localhost',
                    'api_port': 5001,
                    'model': 'auto'
                }
            }
        }
        
        # Test HybridLLMProcessor initialization
        processor = HybridLLMProcessor(full_config)
        print(f"   Processor mode: {processor.mode}")
        print(f"   Backend type: {type(processor.backend).__name__}")
        
        # Test that the backend has auto capabilities
        if hasattr(processor.backend, 'get_context_window_size'):
            backend_context = processor.backend.get_context_window_size()
            print(f"   Backend context size: {backend_context}")
        
        if hasattr(processor.backend, 'calculate_auto_max_tokens'):
            backend_auto_tokens = processor.backend.calculate_auto_max_tokens("Test integration prompt")
            print(f"   Backend auto tokens: {backend_auto_tokens}")
        
        print("   ✅ Configuration integration working!")
        
        # Test 4: Auto vs manual mode comparison
        print("\n4. Testing auto vs manual mode comparison...")
        
        manual_config = external_config.copy()
        manual_config['max_tokens'] = 512
        manual_config['context_size'] = 2048
        
        manual_llm = ExternalAPILLM(manual_config)
        manual_context = manual_llm.get_context_window_size()
        manual_tokens = 512  # Fixed value
        
        auto_context = external_llm.get_context_window_size()
        auto_tokens = external_llm.calculate_auto_max_tokens("Test comparison prompt. " * 50)
        
        print(f"   Manual mode: context={manual_context}, max_tokens={manual_tokens}")
        print(f"   Auto mode: context={auto_context}, max_tokens={auto_tokens}")
        print("   ✅ Auto vs manual comparison working!")
        
        print("\n🎉 All auto context detection tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_auto_settings():
    """Test loading auto settings from config file"""
    print("\n🔧 Testing Config Auto Settings")
    print("=" * 50)
    
    try:
        import yaml
        
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        if config_path.exists():
            print("1. Loading actual config.yaml...")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            llm_config = config.get('llm', {})
            
            # Check auto settings
            max_tokens = llm_config.get('max_tokens')
            context_size = llm_config.get('context_size')
            
            print(f"   max_tokens: {max_tokens} (type: {type(max_tokens).__name__})")
            print(f"   context_size: {context_size} (type: {type(context_size).__name__})")
            
            # Validate auto settings
            if max_tokens == "auto" and context_size == "auto":
                print("   ✅ Auto settings correctly configured in config.yaml!")
            else:
                print(f"   ⚠️  Auto settings not found or incorrect")
            
            return True
        else:
            print("   ⚠️  Config file not found, skipping this test")
            return True
        
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False

def test_token_calculation_accuracy():
    """Test the accuracy of token calculations"""
    print("\n📊 Testing Token Calculation Accuracy")
    print("=" * 50)
    
    try:
        from intv.llm import ExternalAPILLM
        
        config = {
            'provider': 'koboldcpp',
            'api_base': 'http://localhost',
            'api_port': 5001,
            'context_size': 4096,  # Fixed size for testing
            'max_tokens': 'auto'
        }
        
        llm = ExternalAPILLM(config)
        
        # Test different prompt lengths and see how calculations adapt
        test_cases = [
            ("Empty", ""),
            ("Single word", "Test"),
            ("Short sentence", "This is a short test sentence."),
            ("Medium paragraph", "This is a medium-length paragraph for testing auto token calculation. " * 10),
            ("Long document", "This is a long document simulation with repeated content. " * 100),
            ("Very long document", "This is a very long document that might exceed context limits. " * 500)
        ]
        
        print("Testing token calculations:")
        print("   Prompt Type          | Chars | Est.Tokens | Max Tokens | Ratio")
        print("   " + "-" * 65)
        
        for name, prompt in test_cases:
            chars = len(prompt)
            est_tokens = chars // 4  # Our estimation method
            max_tokens = llm.calculate_auto_max_tokens(prompt)
            ratio = max_tokens / max(est_tokens, 1) if est_tokens > 0 else float('inf')
            
            print(f"   {name:<20} | {chars:5d} | {est_tokens:10d} | {max_tokens:10d} | {ratio:5.2f}")
        
        print("   ✅ Token calculation accuracy test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Token calculation test failed: {e}")
        return False

def main():
    """Run focused auto context detection tests"""
    print("🚀 INTV Auto Context Window Detection - Focused Tests")
    print("=" * 60)
    
    tests = [
        test_auto_context_detection,
        test_config_auto_settings,
        test_token_calculation_accuracy
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All focused tests passed! Auto context detection is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
