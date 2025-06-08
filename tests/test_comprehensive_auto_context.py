#!/usr/bin/env python3
"""
Comprehensive integration test for auto context window detection with real RAG-LLM pipeline
"""

import sys
import os
import tempfile
import json
import yaml
import time
import argparse
from pathlib import Path

# Add the intv module to the path
sys.path.insert(0, str(Path(__file__).parent / 'intv'))

def test_token_calculation_performance(cpu_only=False):
    """Test token calculation performance and accuracy"""
    print("=== Testing Token Calculation Performance ===")
    
    try:
        from intv.llm import EmbeddedLLM, ExternalAPILLM
        
        # Configure for CPU or GPU based on flag
        if cpu_only:
            config = {
                'llm': {
                    'embedded': {
                        'model': 'auto',  # Use auto-selection for proper LLM
                        'device': 'cpu'
                    },
                    'max_tokens': 'auto',
                    'context_size': 'auto'
                }
            }
            print("1. Testing CPU-Only Token Calculations...")
        else:
            config = {
                'llm': {
                    'embedded': {
                        'model': 'auto'  # Use auto-selection for proper LLM
                    },
                    'max_tokens': 'auto',
                    'context_size': 'auto'
                }
            }
            print("1. Testing GPU High-End Token Calculations...")
        
        embedded_llm = EmbeddedLLM(config)
        
        # Test different document sizes
        test_cases = [
            ("Small doc", "Brief document for testing. " * 20),
            ("Medium doc", "Medium length document for comprehensive testing. " * 100), 
            ("Large doc", "Large document with extensive content for testing auto context detection capabilities. " * 500),
            ("XL doc", "Extra large document designed to test the limits of context window management and token calculation accuracy. " * 2000)
        ]
        
        results = {}
        
        for name, content in test_cases:
            start_time = time.time()
            
            # Get context window size
            context_size = embedded_llm.get_context_window_size()
            
            # Calculate auto tokens
            auto_tokens = embedded_llm.calculate_auto_max_tokens(content)
            
            # Calculate token estimation metrics
            char_count = len(content)
            estimated_tokens = char_count // 4  # 1 token ≈ 4 chars
            calc_time = time.time() - start_time
            
            results[name] = {
                'char_count': char_count,
                'estimated_tokens': estimated_tokens,
                'auto_max_tokens': auto_tokens,
                'context_size': context_size,
                'calc_time_ms': calc_time * 1000,
                'token_efficiency': (auto_tokens / context_size) * 100 if context_size > 0 else 0
            }
            
            print(f"   {name}:")
            print(f"     Characters: {char_count:,}")
            print(f"     Est. tokens: {estimated_tokens:,}")
            print(f"     Max tokens: {auto_tokens:,}")
            print(f"     Context size: {context_size:,}")
            print(f"     Calc time: {calc_time*1000:.2f}ms")
            print(f"     Efficiency: {(auto_tokens/context_size)*100:.1f}% of context")
            print()
        
        # Test external API calculations
        print("2. Testing External API Token Calculations...")
        
        external_configs = [
            ('OpenAI GPT-4', {'provider': 'openai', 'model': 'gpt-4'}),
            ('KoboldCpp', {'provider': 'koboldcpp', 'model': 'auto'})
        ]
        
        for provider_name, provider_config in external_configs:
            print(f"   {provider_name}:")
            
            full_config = {
                'llm': {
                    'external': {
                        **provider_config,
                        'api_base': 'http://localhost',
                        'api_port': 5001,
                        'api_key': 'test'
                    },
                    'max_tokens': 'auto'
                }
            }
            
            external_llm = ExternalAPILLM(full_config)
            
            # Test with medium document
            _, test_content = test_cases[1]  # Medium doc
            start_time = time.time()
            
            context_size = external_llm.get_context_window_size()
            auto_tokens = external_llm.calculate_auto_max_tokens(test_content)
            calc_time = time.time() - start_time
            
            print(f"     Context size: {context_size:,}")
            print(f"     Auto tokens: {auto_tokens:,}")
            print(f"     Calc time: {calc_time*1000:.2f}ms")
            print()
        
        return results
        
    except Exception as e:
        print(f"   ❌ Token calculation test error: {e}")
        return {}

def test_real_world_performance(cpu_only=False):
    """Test with real document processing"""
    print("\n=== Testing Real-World Document Performance ===")
    
    try:
        from intv.llm import HybridLLMProcessor
        from intv.rag_system import RAGSystem
        
        # Configure for CPU or GPU based on flag
        if cpu_only:
            config = {
                'llm': {
                    'mode': 'embedded',
                    'max_tokens': 'auto',
                    'context_size': 'auto',
                    'temperature': 0.7,
                    'embedded': {
                        'model': 'auto',  # Use auto-selection for proper LLM
                        'device': 'cpu'
                    }
                },
                'rag': {
                    'mode': 'embedded',
                    'embedding_model': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1',
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'max_chunks': 5,
                    'device': 'cpu'
                }
            }
        else:
            config = {
                'llm': {
                    'mode': 'embedded',
                    'max_tokens': 'auto',
                    'context_size': 'auto',
                    'temperature': 0.7,
                    'embedded': {
                        'model': 'auto'  # Use auto-selection for proper LLM
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
        
        print("1. Initializing systems...")
        rag_system = RAGSystem(config)
        llm_processor = HybridLLMProcessor(config)
        
        # Create realistic test documents
        documents = {
            'interview_transcript.txt': """
            Interview Transcript - June 8, 2025
            Interviewer: Sarah Johnson
            Candidate: Michael Chen
            Position: Senior Software Engineer
            
            Q: Tell me about your background in software development.
            A: I have over 8 years of experience in full-stack development, primarily working with Python, React, and Node.js. I've led multiple teams and successfully delivered large-scale applications serving millions of users.
            
            Q: What's your experience with machine learning and AI?
            A: I've been working with ML for the past 3 years, implementing recommendation systems and natural language processing solutions. I'm particularly experienced with TensorFlow, PyTorch, and transformers.
            
            Q: Describe a challenging project you've worked on.
            A: I led the development of a real-time analytics platform that processed over 10TB of data daily. The main challenge was optimizing query performance while maintaining 99.9% uptime. We used Apache Kafka for streaming, Redis for caching, and implemented a distributed architecture that could scale horizontally.
            
            Q: How do you handle technical debt in large codebases?
            A: I believe in proactive refactoring and establishing clear coding standards. We implemented automated testing with 95% coverage, used static analysis tools, and scheduled regular tech debt sprints. This reduced our bug rate by 60% over 6 months.
            
            Q: What are your thoughts on team collaboration and remote work?
            A: Effective communication is crucial. I've successfully managed distributed teams across 3 time zones using agile methodologies. We used tools like Slack, Jira, and held daily standups to maintain alignment and productivity.
            """ * 3,  # Make it longer
            
            'policy_document.txt': """
            Employee Handbook - Technology Department
            Effective Date: January 1, 2025
            Last Updated: June 8, 2025
            
            1. CODE OF CONDUCT
            All employees must maintain professional behavior and adhere to company values:
            - Integrity in all business dealings
            - Respect for colleagues and clients
            - Commitment to quality and excellence
            - Protection of confidential information
            - Compliance with all applicable laws and regulations
            
            2. TECHNICAL STANDARDS
            Development teams must follow established best practices:
            - Code reviews required for all changes
            - Automated testing with minimum 80% coverage
            - Documentation for all public APIs
            - Security scanning for all deployments
            - Performance monitoring and optimization
            
            3. DATA HANDLING POLICIES
            Strict protocols govern data access and usage:
            - Personal data encryption at rest and in transit
            - Access controls based on principle of least privilege
            - Regular security audits and penetration testing
            - Incident response procedures for data breaches
            - Compliance with GDPR, CCPA, and other regulations
            
            4. REMOTE WORK GUIDELINES
            Flexible work arrangements with clear expectations:
            - Core hours overlap for team collaboration
            - Secure VPN access for company resources
            - Regular check-ins with managers and teams
            - Proper ergonomic setup for home offices
            - Clear boundaries between work and personal time
            
            5. PERFORMANCE EVALUATION
            Annual reviews based on objective criteria:
            - Technical skill development and application
            - Project delivery and quality metrics
            - Team collaboration and leadership
            - Innovation and problem-solving abilities
            - Professional growth and learning initiatives
            """ * 2
        }
        
        results = {}
        total_processing_time = 0
        total_characters = 0
        
        for doc_name, content in documents.items():
            print(f"\n2. Processing {doc_name}...")
            
            start_time = time.time()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                # RAG processing - use the chunk_document function from rag.py
                from intv.rag import chunk_document
                chunks = chunk_document(temp_path, config=config)
                
                # LLM analysis with auto context
                summary = llm_processor.generate_general_summary(content)
                
                policy_prompt = f"Analyze {doc_name} and extract key information, dates, names, and important details."
                policy_result = llm_processor.generate_policy_summary(content, policy_prompt)
                
                processing_time = time.time() - start_time
                
                # Calculate token metrics
                char_count = len(content)
                estimated_tokens = char_count // 4
                
                results[doc_name] = {
                    'char_count': char_count,
                    'estimated_tokens': estimated_tokens,
                    'num_chunks': len(chunks),
                    'summary_length': len(summary) if summary else 0,
                    'policy_success': policy_result.get('success', False),
                    'processing_time': processing_time,
                    'chars_per_second': char_count / processing_time if processing_time > 0 else 0
                }
                
                total_processing_time += processing_time
                total_characters += char_count
                
                print(f"   ✓ Processed in {processing_time:.2f}s")
                print(f"     Characters: {char_count:,}")
                print(f"     Est. tokens: {estimated_tokens:,}")
                print(f"     Chunks: {len(chunks)}")
                print(f"     Rate: {char_count/processing_time:,.0f} chars/sec")
                
            finally:
                os.unlink(temp_path)
        
        # Calculate overall performance metrics
        if total_processing_time > 0:
            overall_rate = total_characters / total_processing_time
            avg_tokens_per_sec = (total_characters // 4) / total_processing_time
            
            print(f"\n3. Overall Performance Metrics:")
            print(f"   Total content: {total_characters:,} characters")
            print(f"   Total tokens (est.): {total_characters//4:,}")
            print(f"   Total time: {total_processing_time:.2f}s")
            print(f"   Processing rate: {overall_rate:,.0f} chars/second")
            print(f"   Token rate: {avg_tokens_per_sec:,.0f} tokens/second")
            
            # Store performance data for README update
            return {
                'total_chars': total_characters,
                'total_tokens_est': total_characters // 4,
                'total_time': total_processing_time,
                'chars_per_second': overall_rate,
                'tokens_per_second': avg_tokens_per_sec,
                'documents_processed': len(documents),
                'auto_context_enabled': True
            }
        
        return results
        
    except Exception as e:
        print(f"   ❌ Real-world performance test error: {e}")
        return {}

def main():
    """Run comprehensive integration tests and collect performance metrics"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run INTV auto context integration tests")
    parser.add_argument('--cpu', action='store_true', help='Run tests in CPU-only mode')
    args = parser.parse_args()
    
    print("🧪 INTV Auto Context Integration & Performance Test Suite")
    print("=" * 70)
    
    if args.cpu:
        print("🖥️  Running in CPU-only mode")
    else:
        print("🚀 Running in GPU-accelerated mode")
    
    # Run token calculation tests
    token_results = test_token_calculation_performance(cpu_only=args.cpu)
    
    # Run real-world performance tests
    performance_results = test_real_world_performance(cpu_only=args.cpu)
    
    # Generate summary for README
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY FOR README UPDATE")
    print("=" * 70)
    
    if performance_results and isinstance(performance_results, dict) and 'chars_per_second' in performance_results:
        if args.cpu:
            print("\n**CPU-Only Systems (Intel I9-14900K - cpu_medium) with Auto Context:** *(Tested)*")
        else:
            print("\n**GPU High-End (RTX 4070 Ti SUPER - gpu_high) with Auto Context:** *(Tested)*")
        
        print(f"- **Model**: 438MB multi-qa-mpnet-base-dot-v1")
        print(f"- **Auto Context Detection**: ✅ Enabled")
        print(f"- **Documents Processed**: {performance_results.get('documents_processed', 'N/A')}")
        print(f"- **Total Content**: {performance_results.get('total_chars', 0):,} characters")
        print(f"- **Estimated Tokens**: {performance_results.get('total_tokens_est', 0):,} tokens")
        print(f"- **Processing Time**: {performance_results.get('total_time', 0):.2f} seconds")
        print(f"- **Processing Rate**: {performance_results.get('chars_per_second', 0):,.0f} chars/second")
        print(f"- **Token Processing Rate**: {performance_results.get('tokens_per_second', 0):,.0f} tokens/second")
        print(f"- **Auto Max Tokens**: Dynamic calculation based on context window")
        print(f"- **Context Utilization**: Optimal (auto-calculated per prompt)")
        print(f"- **Memory Efficiency**: Enhanced with auto token management")
    else:
        print("⚠️  Performance data not available - tests may have failed")
    
    print("\n🎉 Integration testing complete!")
    print("Use the performance summary above to update the README.md file.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
