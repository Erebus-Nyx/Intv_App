#!/usr/bin/env python3
"""
Test LLM Output Formatting Issues

This test investigates and resolves the LLM output formatting problem
to ensure structured, well-formatted output from the LLM system.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_llm_output_basic():
    """Test basic LLM output generation"""
    print("=== Test 1: Basic LLM Output ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'context_size': 'auto',
                'temperature': 0.7,
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        print("1. Initializing LLM processor...")
        llm_processor = HybridLLMProcessor(config)
        
        # Test basic text generation
        test_text = """
        John Smith is a 35-year-old software engineer living at 123 Main Street.
        He is married to Jane Smith and has two children: Tommy (8) and Sarah (5).
        He works at Tech Corp and volunteers at the local food bank.
        """
        
        print("2. Testing general summary generation...")
        summary = llm_processor.generate_general_summary(test_text)
        print(f"Summary generated: {type(summary)}")
        print(f"Summary length: {len(summary) if summary else 0}")
        print(f"Summary content: {summary[:200] if summary else 'None'}...")
        
        print("3. Testing policy-driven extraction...")
        policy_prompt = "Extract the following information: name, age, occupation, family_members"
        variables = ['name', 'age', 'occupation', 'family_members']
        
        policy_result = llm_processor.generate_policy_summary(test_text, policy_prompt, variables)
        print(f"Policy result type: {type(policy_result)}")
        print(f"Policy result success: {policy_result.get('success', False)}")
        print(f"Policy result output: {policy_result.get('output', 'None')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_output_formatting():
    """Test JSON output formatting specifically"""
    print("\n=== Test 2: JSON Output Formatting ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'temperature': 0.3,  # Lower temperature for more structured output
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        llm_processor = HybridLLMProcessor(config)
        
        # Test with structured data request
        test_text = """
        Case File: John Smith (ID: 2025-001)
        Personal Information:
        - Full Name: John Michael Smith
        - Age: 35 years old
        - Date of Birth: March 15, 1990
        - Address: 123 Main Street, Anytown, State 12345
        - Phone: (555) 123-4567
        - Employment: Senior Software Engineer at Tech Corp
        - Salary: $95,000 annually
        
        Family Information:
        - Spouse: Jane Elizabeth Smith (32 years old)
        - Children: Tommy Smith (8), Sarah Smith (5)
        - Emergency Contact: Mary Smith (mother) - (555) 987-6543
        
        Assessment Notes:
        - Family appears stable and well-functioning
        - No safety concerns identified
        - Strong community involvement
        - Recommended for continued monitoring
        """
        
        print("1. Testing structured JSON extraction...")
        policy_prompt = """
        Extract the following information from the case file and format as valid JSON:
        {
            "personal_info": {
                "full_name": "extracted name",
                "age": "extracted age", 
                "address": "extracted address",
                "employment": "extracted job",
                "phone": "extracted phone"
            },
            "family_info": {
                "spouse": "spouse information",
                "children": "children information",
                "emergency_contact": "emergency contact"
            },
            "assessment": {
                "status": "assessment status",
                "recommendations": "recommendations"
            }
        }
        """
        
        variables = ['personal_info', 'family_info', 'assessment']
        
        json_result = llm_processor.generate_policy_summary(test_text, policy_prompt, variables)
        
        print(f"JSON result success: {json_result.get('success', False)}")
        raw_output = json_result.get('output', '')
        print(f"Raw output length: {len(raw_output)}")
        print(f"Raw output preview: {raw_output[:300]}...")
        
        # Try to parse as JSON
        try:
            if raw_output:
                parsed_json = json.loads(raw_output)
                print("✅ Successfully parsed as JSON!")
                print("Extracted structure:")
                for key, value in parsed_json.items():
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, dict):
                        for subkey in value.keys():
                            print(f"    - {subkey}")
                return True
        except json.JSONDecodeError:
            print("⚠️ Output not valid JSON, checking for JSON-like content...")
            # Look for JSON patterns in the output
            if '{' in raw_output and '}' in raw_output:
                # Try to extract JSON from the response
                start = raw_output.find('{')
                end = raw_output.rfind('}') + 1
                if start >= 0 and end > start:
                    json_part = raw_output[start:end]
                    try:
                        parsed_json = json.loads(json_part)
                        print("✅ Successfully extracted and parsed JSON from response!")
                        return True
                    except json.JSONDecodeError:
                        print("❌ Could not parse extracted JSON")
            else:
                print("❌ No JSON structure found in output")
        
        return False
        
    except Exception as e:
        print(f"❌ JSON formatting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_formatting_improvements():
    """Test and implement formatting improvements"""
    print("\n=== Test 3: Formatting Improvements ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        
        # Try different prompt strategies
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'temperature': 0.1,  # Very low temperature for consistency
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        llm_processor = HybridLLMProcessor(config)
        
        test_text = """
        Jane Doe, age 28, works as a Marketing Manager at Global Corp.
        She lives at 456 Oak Avenue, Springfield, IL 62701.
        She is single with no children and earns $75,000 per year.
        """
        
        # Test 1: Simple structured prompt
        print("1. Testing simple structured prompt...")
        simple_prompt = """
        Extract information and respond ONLY with valid JSON in this exact format:
        {"name": "value", "age": "value", "job": "value", "address": "value", "status": "value"}
        
        Do not include any explanation or additional text. Only return the JSON.
        """
        
        result1 = llm_processor.generate_policy_summary(test_text, simple_prompt)
        output1 = result1.get('output', '')
        print(f"Simple prompt result: {output1}")
        
        # Test 2: More explicit formatting
        print("\n2. Testing explicit formatting prompt...")
        explicit_prompt = """
        You must respond with ONLY a JSON object. No explanations, no additional text.
        
        Format: {"field": "value", "field2": "value2"}
        
        Extract: name, age, occupation, location, marital_status
        """
        
        result2 = llm_processor.generate_policy_summary(test_text, explicit_prompt)
        output2 = result2.get('output', '')
        print(f"Explicit prompt result: {output2}")
        
        # Test 3: Template-based approach
        print("\n3. Testing template-based approach...")
        template_prompt = """
        Fill in this template with extracted information:
        
        {
            "name": "[EXTRACT_NAME]",
            "age": "[EXTRACT_AGE]", 
            "occupation": "[EXTRACT_JOB]",
            "address": "[EXTRACT_ADDRESS]",
            "marital_status": "[EXTRACT_STATUS]"
        }
        
        Replace [EXTRACT_X] with actual values. Return only the completed JSON.
        """
        
        result3 = llm_processor.generate_policy_summary(test_text, template_prompt)
        output3 = result3.get('output', '')
        print(f"Template prompt result: {output3}")
        
        # Evaluate which works best
        outputs = [output1, output2, output3]
        names = ["Simple", "Explicit", "Template"]
        
        for i, (output, name) in enumerate(zip(outputs, names)):
            try:
                if output and '{' in output:
                    # Extract JSON part
                    start = output.find('{')
                    end = output.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_part = output[start:end]
                        parsed = json.loads(json_part)
                        print(f"✅ {name} approach: Valid JSON with {len(parsed)} fields")
                    else:
                        print(f"❌ {name} approach: No valid JSON structure")
                else:
                    print(f"❌ {name} approach: No output or no JSON markers")
            except Exception as e:
                print(f"❌ {name} approach failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Formatting improvements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_system_integration():
    """Test LLM system integration with proper formatting"""
    print("\n=== Test 4: LLM System Integration ===")
    
    try:
        from intv.llm import LLMSystem
        
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'temperature': 0.2,
                'embedded': {
                    'model': 'auto'
                }
            }
        }
        
        print("1. Initializing LLM system...")
        llm_system = LLMSystem(config)
        
        # Test document processing
        chunks = [
            """Personal Information: Michael Johnson, 42 years old, employed as Project Manager at BuildCorp.""",
            """Family: Married to Lisa Johnson (39), two children: Emma (12) and Alex (9).""",
            """Address: 789 Pine Street, Riverside, CA 92501. Phone: (951) 555-0123.""",
            """Assessment: Stable family environment, good communication, no concerns identified."""
        ]
        
        print("2. Testing document processing with structured output...")
        
        policy_prompt = """
        Extract information and format as JSON with these exact fields:
        - participant_name: full name of main participant
        - participant_age: age as number
        - employment: job title and company
        - family_structure: spouse and children information
        - contact_info: address and phone
        - assessment_summary: brief assessment status
        
        Return only valid JSON without any additional text.
        """
        
        variables = ['participant_name', 'participant_age', 'employment', 'family_structure', 'contact_info', 'assessment_summary']
        
        result = llm_system.process_document(
            chunks=chunks,
            policy_prompt=policy_prompt,
            variables=variables
        )
        
        print(f"Processing success: {result.get('success', False)}")
        print(f"Processing mode: {result.get('mode', 'unknown')}")
        
        # Check policy summary
        policy_summary = result.get('policy_summary', {})
        print(f"Policy summary type: {type(policy_summary)}")
        
        if isinstance(policy_summary, dict):
            policy_output = policy_summary.get('output', '')
            print(f"Policy output length: {len(policy_output)}")
            print(f"Policy output preview: {policy_output[:300]}...")
            
            # Try parsing the output
            try:
                if '{' in policy_output:
                    start = policy_output.find('{')
                    end = policy_output.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_part = policy_output[start:end]
                        parsed_json = json.loads(json_part)
                        print("✅ Successfully parsed structured output!")
                        print("Extracted variables:")
                        for key, value in parsed_json.items():
                            print(f"  {key}: {value}")
                        return True
            except Exception as e:
                print(f"❌ JSON parsing failed: {e}")
        
        # Check general summary
        general_summary = result.get('general_summary', '')
        print(f"General summary length: {len(general_summary)}")
        print(f"General summary: {general_summary[:200]}...")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ LLM system integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all LLM output formatting tests"""
    print("🔧 INTV LLM Output Formatting Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic LLM Output", test_llm_output_basic),
        ("JSON Output Formatting", test_json_output_formatting),
        ("Formatting Improvements", test_formatting_improvements),
        ("LLM System Integration", test_llm_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 40)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        print(f"{'✅' if result else '❌'} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All LLM output formatting tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
