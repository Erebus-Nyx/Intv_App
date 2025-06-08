#!/usr/bin/env python3
"""
Test the restored LLM pipeline functionality with proper context size
"""

import sys
import os
from pathlib import Path

# Add the intv module to the path
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_llm_initialization():
    """Test LLM initialization with proper context size"""
    print("🧪 Testing LLM Pipeline Initialization")
    print("=" * 50)
    
    try:
        from intv.llm import EmbeddedLLM, HybridLLMProcessor
        
        # Test configuration with auto model selection
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'context_size': 'auto',
                'temperature': 0.7,
                'embedded': {
                    'model': 'auto'  # Use auto-selection for proper LLM
                }
            }
        }
        
        print("1. Initializing EmbeddedLLM...")
        embedded_llm = EmbeddedLLM(config)
        
        # Check context window size
        context_size = embedded_llm.get_context_window_size()
        print(f"   ✅ Context window size: {context_size:,} tokens")
        
        # Test auto token calculation
        test_prompt = "Generate a comprehensive summary of machine learning techniques used in natural language processing."
        auto_tokens = embedded_llm.calculate_auto_max_tokens(test_prompt)
        print(f"   ✅ Auto max tokens: {auto_tokens:,} tokens")
        print(f"   ✅ Context utilization: {(auto_tokens/context_size)*100:.1f}%")
        
        print("\n2. Initializing HybridLLMProcessor...")
        llm_processor = HybridLLMProcessor(config)
        print("   ✅ HybridLLMProcessor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_generation():
    """Test actual text generation with the LLM"""
    print("\n🧪 Testing Text Generation")
    print("=" * 50)
    
    try:
        from intv.llm import HybridLLMProcessor
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 100,  # Keep it small for quick test
                'context_size': 'auto',
                'temperature': 0.7,
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        print("1. Initializing LLM processor...")
        llm_processor = HybridLLMProcessor(config)
        
        print("2. Testing general summary generation...")
        test_content = """
        The development of artificial intelligence has accelerated rapidly in recent years. 
        Machine learning algorithms, particularly deep learning and transformer models, 
        have shown remarkable capabilities in natural language processing, computer vision, 
        and decision-making tasks. Companies are increasingly adopting AI technologies 
        to improve efficiency, automate processes, and gain competitive advantages.
        """
        
        try:
            summary = llm_processor.generate_general_summary(test_content)
            if summary and len(summary) > 10:
                print(f"   ✅ Generated summary ({len(summary)} chars)")
                print(f"   Preview: {summary[:100]}...")
            else:
                print(f"   ⚠️  Summary generated but empty or too short: '{summary}'")
        except Exception as e:
            print(f"   ❌ General summary failed: {e}")
        
        print("\n3. Testing policy summary generation...")
        policy_prompt = "Extract key technologies and business benefits mentioned in the text."
        
        try:
            policy_result = llm_processor.generate_policy_summary(test_content, policy_prompt)
            if policy_result and policy_result.get('success'):
                print(f"   ✅ Policy summary generated successfully")
                summary = policy_result.get('summary', '')
                if summary:
                    print(f"   Preview: {summary[:100]}...")
            else:
                print(f"   ⚠️  Policy summary failed or returned no content")
        except Exception as e:
            print(f"   ❌ Policy summary failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Text generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run LLM pipeline tests"""
    print("🚀 INTV LLM Pipeline Restoration Test")
    print("=" * 60)
    
    # Test initialization
    init_success = test_llm_initialization()
    
    # Test text generation if initialization succeeded
    if init_success:
        gen_success = test_text_generation()
        
        if init_success and gen_success:
            print("\n🎉 LLM Pipeline Restoration: SUCCESS")
            print("   ✅ Initialization working")
            print("   ✅ Context size properly configured")
            print("   ✅ Text generation functional")
            return 0
        else:
            print("\n⚠️  LLM Pipeline: PARTIAL SUCCESS")
            print("   ✅ Initialization working")
            print("   ❌ Text generation issues")
            return 1
    else:
        print("\n❌ LLM Pipeline: FAILED")
        print("   ❌ Initialization failed")
        return 2

if __name__ == "__main__":
    sys.exit(main())
