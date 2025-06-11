#!/usr/bin/env python3
"""
Debug LLM Policy Summary Generation

This test debugs the specific issue with policy summary generation
that's causing "Policy analysis failed" errors.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_embedded_llm():
    """Debug the embedded LLM directly"""
    print("=== Debug 1: Embedded LLM Direct ===")
    
    try:
        from intv.llm import EmbeddedLLM
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        print("1. Initializing EmbeddedLLM directly...")
        embedded_llm = EmbeddedLLM(config)
        
        print(f"LLM model loaded: {embedded_llm.llama_model is not None}")
        
        if embedded_llm.llama_model:
            print("2. Testing direct text generation...")
            simple_prompt = "Summarize: John is 35 years old and works as an engineer."
            
            try:
                result = embedded_llm.generate_text(simple_prompt, max_tokens=50)
                print(f"Direct generation result: {result}")
                return True
            except Exception as e:
                print(f"Direct generation failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ No LLM model loaded")
            return False
            
    except Exception as e:
        print(f"❌ Embedded LLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_policy_summary_step_by_step():
    """Debug policy summary generation step by step"""
    print("\n=== Debug 2: Policy Summary Step-by-Step ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'temperature': 0.7,
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        print("1. Initializing HybridLLMProcessor...")
        llm_processor = HybridLLMProcessor(config)
        
        print(f"Processor mode: {llm_processor.mode}")
        print(f"Backend type: {type(llm_processor.backend)}")
        
        # Test the backend directly first
        print("2. Testing backend generate_text directly...")
        test_text = "John Smith is 35 years old."
        simple_prompt = f"Extract age from: {test_text}"
        
        try:
            backend_result = llm_processor.backend.generate_text(simple_prompt, max_tokens=20)
            print(f"Backend result: {backend_result}")
        except Exception as e:
            print(f"Backend generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Now test policy summary method
        print("3. Testing generate_policy_summary method...")
        policy_prompt = "Extract the age from the text"
        variables = ['age']
        
        try:
            # Manually recreate the logic from generate_policy_summary
            llm_config = config.get('llm', {})
            configured_max_tokens = llm_config.get('max_tokens', 100)
            
            if configured_max_tokens == "auto":
                truncated_text = test_text
                max_tokens = "auto"
            else:
                truncated_text = test_text[:1000] if len(test_text) > 1000 else test_text
                max_tokens = 100
            
            if variables:
                var_instruction = f"Extract the following variables: {', '.join(variables[:10])}"
                prompt = f"{policy_prompt}\n\n{var_instruction}\n\nText: {truncated_text}\n\nJSON:"
            else:
                prompt = f"{policy_prompt}\n\nText: {truncated_text}\n\nAnalysis:"
            
            print(f"Generated prompt: {prompt}")
            print(f"Max tokens: {max_tokens}")
            
            # Test the generation
            manual_result = llm_processor.backend.generate_text(prompt, max_tokens=max_tokens)
            print(f"Manual generation result: {manual_result}")
            
            # Now test the actual method
            policy_result = llm_processor.generate_policy_summary(test_text, policy_prompt, variables)
            print(f"Policy method result: {policy_result}")
            
            return True
            
        except Exception as e:
            print(f"Policy generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Policy summary debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_with_error_capture():
    """Debug with detailed error capture"""
    print("\n=== Debug 3: Error Capture ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        import logging
        
        # Enable detailed logging
        logging.basicConfig(level=logging.DEBUG)
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 50,  # Use fixed tokens to avoid auto issues
                'temperature': 0.7,
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        print("1. Testing with fixed max_tokens...")
        llm_processor = HybridLLMProcessor(config)
        
        test_text = "Name: Alice, Age: 30"
        policy_prompt = "Extract name and age"
        variables = ['name', 'age']
        
        # Test with detailed error handling
        try:
            result = llm_processor.generate_policy_summary(test_text, policy_prompt, variables)
            print(f"Fixed tokens result: {result}")
            
            if result.get('success'):
                print("✅ Policy summary succeeded with fixed tokens!")
                return True
            else:
                print(f"❌ Policy summary failed: {result.get('output')}")
                
        except Exception as e:
            print(f"Exception in policy summary: {e}")
            import traceback
            traceback.print_exc()
        
        return False
        
    except Exception as e:
        print(f"❌ Error capture debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_fix():
    """Test a simple fix for the policy summary issue"""
    print("\n=== Debug 4: Simple Fix Test ===")
    
    try:
        # Test by directly calling the embedded LLM with a simple policy extraction
        from intv.llm import EmbeddedLLM
        
        config = {
            'llm': {
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        embedded_llm = EmbeddedLLM(config)
        
        # Simple extraction prompt
        text = "Patient: John Smith, Age: 35, Occupation: Engineer"
        prompt = """Extract information from the text and format as JSON:
        
Text: Patient: John Smith, Age: 35, Occupation: Engineer

JSON:"""
        
        print("Testing simple JSON extraction...")
        result = embedded_llm.generate_text(prompt, max_tokens=100, temperature=0.3)
        print(f"Simple extraction result: {result}")
        
        # Check if it contains JSON-like content
        if '{' in result and '}' in result:
            print("✅ Result contains JSON structure!")
            
            # Try to extract and parse
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_part = result[start:end]
                print(f"Extracted JSON: {json_part}")
                
                try:
                    import json
                    parsed = json.loads(json_part)
                    print(f"✅ Successfully parsed JSON: {parsed}")
                    return True
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing failed: {e}")
        else:
            print("❌ No JSON structure in result")
        
        return False
        
    except Exception as e:
        print(f"❌ Simple fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🔧 LLM Policy Summary Debug Suite")
    print("=" * 50)
    
    tests = [
        ("Embedded LLM Direct", debug_embedded_llm),
        ("Policy Summary Step-by-Step", debug_policy_summary_step_by_step),
        ("Error Capture", debug_with_error_capture),
        ("Simple Fix Test", test_simple_fix)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Debug Results:")
    for test_name, result in results:
        print(f"{'✅' if result else '❌'} {test_name}")
    
    passed = sum(1 for _, result in results if result)
    print(f"\nDebug success rate: {passed}/{len(results)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
