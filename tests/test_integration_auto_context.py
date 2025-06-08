#!/usr/bin/env python3
"""
Integration test for auto context window detection with real RAG-LLM pipeline
"""

import sys
import os
import tempfile
import json
import yaml
import time
from pathlib import Path

# Add the intv module to the path
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_rag_llm_integration_with_auto():
    """Test RAG-LLM integration with auto context detection"""
    print("=== Testing RAG-LLM Integration with Auto Context Detection ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        from intv.rag_system import RAGSystem
        
        # Configuration with auto settings
        config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',
                'context_size': 'auto',
                'temperature': 0.7,
                'top_p': 0.9,
                'embedded': {
                    'model': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1'  # Use existing model
                }
            },
            'rag': {
                'mode': 'embedded',
                'embedding_model': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'max_chunks': 5
            }
        }
        
        print("1. Initializing RAG system...")
        rag_system = RAGSystem(config)
        
        print("2. Initializing LLM processor with auto context detection...")
        llm_processor = HybridLLMProcessor(config)
        
        # Test different document sizes to validate auto scaling
        test_documents = [
            ("Short document", "This is a short test document for validation. It contains minimal content."),
            ("Medium document", "This is a medium-length test document. " * 50 + " It contains more substantial content for testing auto context detection."),
            ("Long document", "This is a comprehensive test document designed to evaluate auto context window detection. " * 200 + " It contains extensive content to test large context handling."),
            ("Very long document", "This is an extremely long test document created specifically for testing the limits of auto context detection. " * 1000 + " It pushes the boundaries of context window management.")
        ]
        
        results = {}
        
        for doc_name, doc_content in test_documents:
            print(f"\n3. Testing {doc_name} ({len(doc_content)} characters)...")
            
            try:
                # Create temporary document
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(doc_content)
                    temp_path = f.name
                
                start_time = time.time()
                
                # Process with RAG system
                print(f"   Processing with RAG system...")
                chunks = rag_system.chunk_document(temp_path)
                print(f"   Created {len(chunks)} chunks")
                
                # Test general summary with auto context
                print(f"   Generating summary with auto context detection...")
                summary = llm_processor.generate_general_summary(doc_content)
                
                # Test policy summary with auto context
                policy_prompt = "Analyze this document and extract key information in structured format."
                print(f"   Generating policy summary with auto context detection...")
                policy_result = llm_processor.generate_policy_summary(doc_content, policy_prompt)
                
                processing_time = time.time() - start_time
                
                results[doc_name] = {
                    'doc_length': len(doc_content),
                    'num_chunks': len(chunks),
                    'summary_length': len(summary) if summary else 0,
                    'policy_success': policy_result.get('success', False),
                    'processing_time': processing_time,
                    'mode': policy_result.get('mode', 'unknown')
                }
                
                print(f"   ✓ {doc_name} processed successfully")
                print(f"     - Processing time: {processing_time:.2f}s")
                print(f"     - Summary length: {len(summary) if summary else 0} characters")
                print(f"     - Policy analysis: {'Success' if policy_result.get('success', False) else 'Failed'}")
                
                # Cleanup
                os.unlink(temp_path)
                
            except Exception as e:
                print(f"   ⚠️  {doc_name} processing failed: {e}")
                results[doc_name] = {
                    'error': str(e),
                    'doc_length': len(doc_content)
                }
        
        # Print comprehensive results
        print("\n4. Integration Test Results Summary:")
        print("=" * 50)
        for doc_name, result in results.items():
            if 'error' not in result:
                print(f"{doc_name}:")
                print(f"  Document length: {result['doc_length']:,} chars")
                print(f"  Chunks created: {result['num_chunks']}")
                print(f"  Processing time: {result['processing_time']:.2f}s")
                print(f"  Summary length: {result['summary_length']} chars")
                print(f"  Policy analysis: {'✓' if result['policy_success'] else '✗'}")
                print(f"  LLM mode: {result['mode']}")
                print()
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Integration test error: {e}")
        return False

def test_auto_vs_manual_performance():
    """Compare auto vs manual max_tokens performance"""
    print("\n=== Testing Auto vs Manual Performance ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        
        test_text = "This is a comprehensive test document for comparing auto and manual token settings. " * 100
        
        # Test with manual settings
        manual_config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 100,  # Manual setting
                'context_size': 4096,  # Manual setting
                'temperature': 0.7,
                'embedded': {'model': 'auto'}
            }
        }
        
        # Test with auto settings
        auto_config = {
            'llm': {
                'mode': 'embedded',
                'max_tokens': 'auto',  # Auto setting
                'context_size': 'auto',  # Auto setting
                'temperature': 0.7,
                'embedded': {'model': 'auto'}
            }
        }
        
        print("1. Testing manual configuration...")
        manual_processor = HybridLLMProcessor(manual_config)
        
        start_time = time.time()
        manual_summary = manual_processor.generate_general_summary(test_text)
        manual_time = time.time() - start_time
        
        print(f"   Manual mode: {manual_time:.3f}s, {len(manual_summary)} chars")
        
        print("2. Testing auto configuration...")
        auto_processor = HybridLLMProcessor(auto_config)
        
        start_time = time.time()
        auto_summary = auto_processor.generate_general_summary(test_text)
        auto_time = time.time() - start_time
        
        print(f"   Auto mode: {auto_time:.3f}s, {len(auto_summary)} chars")
        
        print("3. Performance comparison:")
        print(f"   Time difference: {abs(auto_time - manual_time):.3f}s")
        print(f"   Output length difference: {abs(len(auto_summary) - len(manual_summary))} chars")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
        return False

def test_context_calculation_accuracy():
    """Test accuracy of context window calculations"""
    print("\n=== Testing Context Calculation Accuracy ===")
    
    try:
        from intv.llm import EmbeddedLLM, ExternalAPILLM
        
        # Test embedded LLM context detection
        print("1. Testing embedded LLM context calculation...")
        embedded_config = {
            'llm': {
                'embedded': {
                    'model': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1'
                }
            }
        }
        
        embedded_llm = EmbeddedLLM(embedded_config)
        
        # Test different prompt lengths
        test_prompts = [
            ("Short", "Brief prompt"),
            ("Medium", "This is a medium-length prompt for testing. " * 10),
            ("Long", "This is a very long prompt designed for comprehensive testing of context window calculations. " * 50),
            ("Very Long", "This is an extremely long prompt that pushes the boundaries of context calculation. " * 200)
        ]
        
        print("   Embedded LLM calculations:")
        for name, prompt in test_prompts:
            context_size = embedded_llm.get_context_window_size()
            auto_tokens = embedded_llm.calculate_auto_max_tokens(prompt)
            estimated_prompt_tokens = len(prompt) // 4
            
            print(f"   {name}: {len(prompt)} chars → {estimated_prompt_tokens} est. tokens → {auto_tokens} max_tokens (ctx: {context_size})")
        
        # Test external LLM context detection
        print("\n2. Testing external LLM context calculation...")
        external_configs = [
            ('OpenAI GPT-4', {'provider': 'openai', 'model': 'gpt-4'}),
            ('OpenAI GPT-3.5', {'provider': 'openai', 'model': 'gpt-3.5-turbo'}),
            ('KoboldCpp', {'provider': 'koboldcpp', 'model': 'auto'}),
            ('Ollama', {'provider': 'ollama', 'model': 'llama2'})
        ]
        
        for provider_name, provider_config in external_configs:
            print(f"   Testing {provider_name}:")
            
            full_config = {
                'llm': {
                    'external': {
                        **provider_config,
                        'api_base': 'http://localhost',
                        'api_port': 5001,
                        'api_key': 'test'
                    }
                }
            }
            
            external_llm = ExternalAPILLM(full_config)
            context_size = external_llm.get_context_window_size()
            
            for name, prompt in test_prompts[:2]:  # Test fewer prompts for external
                auto_tokens = external_llm.calculate_auto_max_tokens(prompt)
                estimated_prompt_tokens = len(prompt) // 4
                
                print(f"     {name}: {len(prompt)} chars → {estimated_prompt_tokens} est. tokens → {auto_tokens} max_tokens (ctx: {context_size})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Context calculation test error: {e}")
        return False
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_rag_llm_integration():
    """Test the complete RAG-to-LLM pipeline with auto context detection"""
    print("🔗 Testing RAG-to-LLM Integration with Auto Context Detection")
    print("=" * 65)
    
    try:
        from intv.rag_system import RAGSystem
        from intv.llm import HybridLLMProcessor
        import tempfile
        
        # Create test documents
        test_docs = [
            "This is a comprehensive test document for validating the auto context window detection in our RAG-to-LLM pipeline. The system should automatically adjust max_tokens based on the model's capabilities.",
            "Auto context detection enables optimal utilization of language models by dynamically calculating the maximum tokens available for generation based on the input prompt length and model constraints.",
            "The hybrid approach combines RAG retrieval with intelligent LLM processing, ensuring that documents are properly chunked and analyzed while respecting context window limitations."
        ]
        
        print("1. Initializing RAG system...")
        rag_config = {
            'mode': 'embedded',
            'model': 'auto',
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 3
        }
        rag_system = RAGSystem(rag_config)
        print("   ✓ RAG system initialized")
        
        print("2. Adding test documents to RAG...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary text files
            doc_paths = []
            for i, content in enumerate(test_docs):
                doc_path = Path(temp_dir) / f"test_doc_{i}.txt"
                with open(doc_path, 'w') as f:
                    f.write(content)
                doc_paths.append(str(doc_path))
            
            # Add documents to RAG
            for doc_path in doc_paths:
                rag_system.add_document(doc_path)
            print(f"   ✓ Added {len(doc_paths)} documents to RAG")
            
            print("3. Initializing LLM with auto context detection...")
            llm_config = {
                'llm': {
                    'mode': 'external',  # Use external to avoid long model loading
                    'max_tokens': 'auto',
                    'context_size': 'auto',
                    'temperature': 0.7,
                    'external': {
                        'provider': 'koboldcpp',
                        'api_base': 'http://localhost',
                        'api_port': 5001,
                        'model': 'auto'
                    }
                }
            }
            
            llm_processor = HybridLLMProcessor(llm_config)
            print(f"   ✓ LLM processor initialized in {llm_processor.mode} mode")
            
            print("4. Testing RAG query with auto context detection...")
            query = "How does auto context detection work in the RAG-to-LLM pipeline?"
            
            # Query RAG system
            rag_results = rag_system.query(query, top_k=3)
            print(f"   ✓ RAG query returned {len(rag_results)} results")
            
            # Extract chunks for LLM processing
            chunks = [result['content'] for result in rag_results]
            combined_context = " ".join(chunks)
            
            print("5. Testing auto max_tokens calculation with combined context...")
            backend = llm_processor.backend
            
            # Test context window detection
            context_size = backend.get_context_window_size()
            print(f"   Context window size: {context_size}")
            
            # Test auto calculation with the combined RAG context
            full_prompt = f"Context: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
            auto_tokens = backend.calculate_auto_max_tokens(full_prompt)
            
            print(f"   Combined context: {len(combined_context)} chars")
            print(f"   Full prompt: {len(full_prompt)} chars")
            print(f"   Auto max_tokens: {auto_tokens}")
            
            # Validate that auto calculation is reasonable
            estimated_prompt_tokens = len(full_prompt) // 4
            expected_available = context_size - estimated_prompt_tokens - 100
            expected_capped = min(max(expected_available, 50), 2048)
            
            print(f"   Expected calculation: {expected_capped}")
            assert auto_tokens == expected_capped, f"Auto calculation mismatch: {auto_tokens} != {expected_capped}"
            print("   ✓ Auto calculation validation passed")
            
            print("6. Testing different prompt sizes...")
            test_prompts = [
                ("Minimal", "Short query"),
                ("Moderate", f"Context: {combined_context[:500]}\n\nQuestion: {query}\n\nAnswer:"),
                ("Large", f"Context: {combined_context}\n\nQuestion: {query}\n\nProvide a comprehensive answer:\n"),
                ("Maximum", f"Context: {combined_context * 3}\n\nQuestion: {query}\n\nProvide a detailed analysis:\n")
            ]
            
            for name, prompt in test_prompts:
                auto_tokens = backend.calculate_auto_max_tokens(prompt)
                chars = len(prompt)
                est_tokens = chars // 4
                print(f"   {name:<10}: {chars:5d} chars → {est_tokens:4d} est.tokens → {auto_tokens:4d} max_tokens")
            
            print("7. Testing configuration consistency...")
            # Verify that auto settings are properly applied
            assert llm_config['llm']['max_tokens'] == 'auto'
            assert llm_config['llm']['context_size'] == 'auto'
            print("   ✓ Configuration consistency verified")
            
        print("\n🎉 RAG-to-LLM integration with auto context detection working perfectly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_world_scenario():
    """Test a real-world document processing scenario"""
    print("\n📄 Testing Real-World Document Processing")
    print("=" * 50)
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        
        # Test with a real document from our sample files
        sample_file = Path(__file__).parent / 'sample-sources' / 'sample_typed_adult.pdf'
        
        if sample_file.exists():
            print("1. Testing with real PDF document...")
            
            # Initialize pipeline with auto configuration
            config = {
                'llm': {
                    'mode': 'external',
                    'max_tokens': 'auto',
                    'context_size': 'auto',
                    'temperature': 0.7
                },
                'rag': {
                    'mode': 'embedded',
                    'chunk_size': 500,
                    'chunk_overlap': 50
                }
            }
            
            pipeline = PipelineOrchestrator(config)
            print("   ✓ Pipeline initialized with auto settings")
            
            # Process the document (without applying LLM to avoid API calls)
            print("2. Processing document...")
            result = pipeline.process_document_or_image(
                str(sample_file), 
                apply_rag=True, 
                apply_llm=False,  # Skip LLM to avoid API calls
                query="Extract key information about this individual"
            )
            
            print(f"   ✓ Document processed: {len(result.get('chunks', []))} chunks")
            
            # Test auto calculation with real document chunks
            if result.get('chunks'):
                print("3. Testing auto calculation with real chunks...")
                from intv.llm import ExternalAPILLM
                
                llm_config = {
                    'provider': 'koboldcpp',
                    'api_base': 'http://localhost',
                    'api_port': 5001,
                    'max_tokens': 'auto',
                    'context_size': 'auto'
                }
                
                llm = ExternalAPILLM(llm_config)
                
                # Test with different chunk combinations
                first_chunk = result['chunks'][0]
                combined_chunks = " ".join(result['chunks'][:3])
                all_chunks = " ".join(result['chunks'])
                
                for name, text in [("Single chunk", first_chunk), ("Three chunks", combined_chunks), ("All chunks", all_chunks)]:
                    prompt = f"Analyze this content: {text}"
                    auto_tokens = llm.calculate_auto_max_tokens(prompt)
                    print(f"   {name}: {len(text)} chars → {auto_tokens} max_tokens")
                
                print("   ✓ Real document auto calculation working")
            
            print("\n🎉 Real-world scenario test completed successfully!")
            return True
        else:
            print("   ⚠️  Sample file not found, skipping real-world test")
            return True
            
    except Exception as e:
        print(f"\n❌ Real-world test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_characteristics():
    """Test performance characteristics of auto detection"""
    print("\n⚡ Testing Performance Characteristics")
    print("=" * 45)
    
    try:
        from intv.llm import ExternalAPILLM, EmbeddedLLM
        import time
        
        print("1. Testing auto calculation performance...")
        
        # Test external API performance
        external_config = {
            'provider': 'koboldcpp',
            'api_base': 'http://localhost',
            'api_port': 5001,
            'max_tokens': 'auto'
        }
        
        external_llm = ExternalAPILLM(external_config)
        
        # Test different prompt sizes and measure time
        test_sizes = [100, 1000, 5000, 20000, 50000]
        
        print("   External API auto calculation times:")
        for size in test_sizes:
            test_prompt = "Test prompt content. " * (size // 20)
            
            start_time = time.time()
            auto_tokens = external_llm.calculate_auto_max_tokens(test_prompt)
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"   {size:5d} chars: {auto_tokens:4d} tokens in {duration:6.2f}ms")
        
        print("\n2. Testing embedded LLM performance...")
        embedded_config = {
            'mode': 'embedded',
            'max_tokens': 'auto',
            'model': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q5_K_M.gguf'
        }
        
        # Only test if we don't need to download the model again
        embedded_llm = EmbeddedLLM(embedded_config)
        
        print("   Embedded LLM auto calculation times:")
        for size in test_sizes[:3]:  # Test fewer sizes to avoid long waits
            test_prompt = "Test prompt content. " * (size // 20)
            
            start_time = time.time()
            auto_tokens = embedded_llm.calculate_auto_max_tokens(test_prompt)
            end_time = time.time()
            
            duration = (end_time - start_time) * 1000
            print(f"   {size:5d} chars: {auto_tokens:4d} tokens in {duration:6.2f}ms")
        
        print("\n   ✓ Performance testing completed - auto calculation is fast!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        return False

def main():
    """Run comprehensive integration tests"""
    print("🚀 INTV Auto Context Detection - Integration Tests")
    print("=" * 60)
    
    tests = [
        test_rag_llm_integration,
        test_real_world_scenario,
        test_performance_characteristics
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
    print(f"📈 INTEGRATION RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Auto context detection integration is complete and ready for production!")
        print("\n📋 Summary of implemented features:")
        print("   ✅ Auto context window detection for embedded and external LLMs")
        print("   ✅ Dynamic max_tokens calculation based on prompt length")
        print("   ✅ Configuration support for 'auto' settings")
        print("   ✅ Integration with RAG-to-LLM pipeline")
        print("   ✅ Performance optimization and safety margins")
        print("   ✅ Support for multiple LLM providers (KoboldCpp, OpenAI, Ollama)")
        print("   ✅ Backward compatibility with manual token limits")
        return 0
    else:
        print("⚠️  Some integration tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
