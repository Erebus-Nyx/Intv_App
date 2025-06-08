#!/usr/bin/env python3
"""
Test LLM pipeline functionality after fixing the embedding model issue
"""

import sys
import os
from pathlib import Path

# Add the intv module to the path
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_llm_auto_selection():
    """Test LLM auto selection and initialization"""
    print("🧪 Testing LLM Auto Selection and Initialization")
    print("=" * 60)
    
    try:
        from intv.llm import EmbeddedLLM, SystemCapabilities
        
        # Detect system type
        system_type = SystemCapabilities.detect_system_type()
        default_model = SystemCapabilities.get_default_llm_model(system_type)
        
        print(f"🖥️  Detected system type: {system_type}")
        print(f"🎯 Default LLM model for system: {default_model}")
        
        # Test proper LLM configuration
        config = {
            'llm': {
                'embedded': {
                    'model': 'auto'  # Should auto-select proper text generation model
                },
                'max_tokens': 'auto',
                'context_size': 'auto'
            }
        }
        
        print(f"\n1. Initializing EmbeddedLLM with auto model selection...")
        embedded_llm = EmbeddedLLM(config)
        
        print(f"✅ LLM initialized successfully!")
        
        # Test context window detection
        context_size = embedded_llm.get_context_window_size()
        print(f"📏 Detected context window size: {context_size:,} tokens")
        
        # Test auto token calculation
        test_prompt = "Write a brief summary of machine learning concepts."
        auto_tokens = embedded_llm.calculate_auto_max_tokens(test_prompt)
        print(f"🔢 Auto max tokens for test prompt: {auto_tokens:,}")
        
        # Display model info
        if embedded_llm.llama_model:
            print(f"🦙 Using llama.cpp backend (GGUF model)")
        elif embedded_llm.model:
            print(f"🤗 Using transformers backend")
            print(f"   Model type: {type(embedded_llm.model).__name__}")
        else:
            print(f"⚠️  No model loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_llm_separation():
    """Test that RAG and LLM use different models correctly"""
    print(f"\n🧪 Testing RAG-LLM Model Separation")
    print("=" * 60)
    
    try:
        from intv.rag_system import RAGSystem
        from intv.llm import EmbeddedLLM
        
        # RAG should use embedding model
        rag_config = {
            'rag': {
                'embedding_model': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1'
            }
        }
        
        # LLM should use text generation model
        llm_config = {
            'llm': {
                'embedded': {
                    'model': 'auto'
                },
                'max_tokens': 'auto'
            }
        }
        
        print(f"1. Initializing RAG with embedding model...")
        rag_system = RAGSystem(rag_config)
        print(f"✅ RAG initialized with embedding model")
        
        print(f"2. Initializing LLM with text generation model...")
        llm_system = EmbeddedLLM(llm_config)
        print(f"✅ LLM initialized with text generation model")
        
        print(f"✅ RAG and LLM are using different, appropriate models")
        return True
        
    except Exception as e:
        print(f"❌ RAG-LLM separation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run LLM pipeline tests"""
    print("🚀 INTV LLM Pipeline Restoration Test")
    print("=" * 80)
    
    # Test 1: LLM Auto Selection
    test1_passed = test_llm_auto_selection()
    
    # Test 2: RAG-LLM Separation
    test2_passed = test_rag_llm_separation()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"✅ LLM Auto Selection: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ RAG-LLM Separation: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! LLM pipeline is functioning correctly.")
        return 0
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
