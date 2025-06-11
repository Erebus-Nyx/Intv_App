#!/usr/bin/env python3
"""
Test adult sample files through the complete RAG/LLM pipeline
to verify output formatting and policy compliance
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_adult_pdf_processing():
    """Test adult PDF processing through complete pipeline"""
    print("📄 Testing Adult PDF Processing")
    print("=" * 50)
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        
        # Initialize orchestrator
        config = {
            'audio': {
                'enable_vad': True,
                'enable_diarization': True,
                'whisper_model': 'base'
            },
            'llm': {
                'mode': 'embedded',  # Use embedded mode for consistent testing
                'provider': 'mock'
            }
        }
        
        orchestrator = PipelineOrchestrator(config)
        
        # Test with adult PDF file
        pdf_path = Path('sample-sources/sample_typed_adult.pdf')
        if not pdf_path.exists():
            print("❌ Adult PDF sample file not found")
            return
        
        print(f"📄 Processing: {pdf_path}")
        print(f"📏 File size: {pdf_path.stat().st_size:,} bytes")
        
        # Process through complete pipeline
        result = orchestrator.process(
            input_path=pdf_path,
            module_key='adult',  # Use adult module
            query="Analyze this document for policy compliance",
            apply_llm=True
        )
        
        # Analyze results
        print(f"\n📊 Processing Results:")
        print(f"   Success: {result.success}")
        print(f"   Input Type: {result.input_type}")
        print(f"   Error: {result.error_message}")
        
        if result.success and result.extracted_text:
            print(f"   Text Length: {len(result.extracted_text):,} characters")
            
            # Show text preview
            preview = result.extracted_text[:200] + "..." if len(result.extracted_text) > 200 else result.extracted_text
            print(f"   Text Preview: {preview}")
            
        if result.chunks:
            print(f"   RAG Chunks: {len(result.chunks)} chunks")
            
        if result.llm_output:
            print(f"\n🤖 LLM Output:")
            print(f"   Type: {type(result.llm_output)}")
            
            # Try to parse as JSON if it's a string
            if isinstance(result.llm_output, str):
                try:
                    parsed = json.loads(result.llm_output)
                    print(f"   ✅ Valid JSON Output:")
                    print(json.dumps(parsed, indent=2))
                except json.JSONDecodeError:
                    print(f"   📝 Text Output: {result.llm_output}")
            else:
                print(f"   📦 Object Output: {result.llm_output}")
                
        # Test policy summary generation specifically
        print(f"\n📋 Testing Policy Summary Generation:")
        try:
            if hasattr(orchestrator, 'llm_processor') and orchestrator.llm_processor:
                summary_result = orchestrator.llm_processor.generate_policy_summary(
                    text=result.extracted_text[:1000] if result.extracted_text else "Test content",
                    policy_prompt="Extract key information in JSON format with fields: client_name, age, assessment_summary",
                    variables=['client_name', 'age', 'assessment_summary']
                )
                
                print(f"   Summary Success: {summary_result.get('success', False)}")
                print(f"   Summary Mode: {summary_result.get('mode', 'unknown')}")
                
                if summary_result.get('success'):
                    output = summary_result.get('output', '')
                    print(f"   Summary Output Type: {type(output)}")
                    
                    # Check if output is JSON formatted
                    if isinstance(output, str) and output.strip().startswith('{'):
                        try:
                            parsed_summary = json.loads(output)
                            print(f"   ✅ Valid JSON Policy Summary:")
                            for key, value in parsed_summary.items():
                                print(f"      {key}: {value}")
                        except json.JSONDecodeError as e:
                            print(f"   ❌ JSON Parse Error: {e}")
                            print(f"   Raw Output: {output[:200]}...")
                    else:
                        print(f"   📝 Text Summary: {output[:200]}...")
                else:
                    print(f"   ❌ Summary Error: {summary_result.get('error', 'unknown')}")
            else:
                print("   ❌ LLM Processor not available")
                
        except Exception as e:
            print(f"   ❌ Policy Summary Error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Adult PDF processing error: {e}")
        import traceback
        traceback.print_exc()

def test_output_formatting_compliance():
    """Test output formatting compliance with policy requirements"""
    print("\n📝 Testing Output Formatting Compliance")
    print("=" * 50)
    
    try:
        from intv.llm import HybridLLMProcessor
        
        # Initialize LLM processor in embedded mode
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto'
            }
        }
        processor = HybridLLMProcessor(config)
        
        # Test sample content
        test_content = """
        Adult Assessment Report
        
        Client Name: John Smith
        Age: 35
        Occupation: Software Engineer
        Address: 123 Main Street, City, State
        
        Assessment Summary:
        The client presents as cooperative and willing to engage in services.
        No immediate safety concerns identified at this time.
        Recommendation for ongoing support services.
        """
        
        print("🤖 Testing Policy Summary Generation...")
        
        # Test policy summary with different configurations
        test_configs = [
            {'max_tokens': 200, 'format': 'json'},
            {'max_tokens': 'auto', 'format': 'json'},
            {'max_tokens': 150, 'format': 'structured'}
        ]
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n📋 Test {i}: max_tokens={config['max_tokens']}")
            
            try:
                result = processor.generate_policy_summary(
                    text=test_content,
                    policy_prompt="Extract key information in JSON format",
                    variables=['client_name', 'age', 'assessment_summary']
                )
                
                print(f"   Success: {result.get('success', False)}")
                print(f"   Mode: {result.get('mode', 'unknown')}")
                
                if result.get('success'):
                    output = result.get('output', '')
                    
                    # Validate JSON formatting
                    if isinstance(output, str):
                        try:
                            json_data = json.loads(output)
                            print(f"   ✅ Valid JSON format")
                            print(f"   📊 Keys: {list(json_data.keys())}")
                            
                            # Check for required fields in adult policy
                            required_fields = ['client_name', 'age', 'assessment_summary']
                            found_fields = [field for field in required_fields if field in json_data]
                            print(f"   📋 Required fields found: {found_fields}")
                            
                        except json.JSONDecodeError as e:
                            print(f"   ❌ JSON format error: {e}")
                            print(f"   Raw output: {output[:100]}...")
                    else:
                        print(f"   📝 Non-string output: {type(output)}")
                        
                else:
                    print(f"   ❌ Processing failed: {result.get('error', 'unknown')}")
                    
            except Exception as e:
                print(f"   ❌ Test {i} error: {e}")
                
    except Exception as e:
        print(f"❌ Output formatting test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_adult_pdf_processing()
    test_output_formatting_compliance()
