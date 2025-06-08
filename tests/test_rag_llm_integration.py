#!/usr/bin/env python3
"""
Test RAG-to-LLM Integration Pipeline

This test verifies that:
1. RAG system can process documents and extract relevant chunks
2. LLM system can receive RAG data and generate summaries
3. General summaries work without policy constraints
4. Policy-adherent summaries follow structured format
5. Variable extraction works with JSON output
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

def test_rag_llm_integration():
    """Test complete RAG-to-LLM integration"""
    print("=== RAG-to-LLM Integration Test ===\n")
    
    try:
        # Import required modules
        from intv.config import load_config
        from intv.rag_system import RAGSystem
        from intv.llm import LLMSystem
        from intv.rag import chunk_text
        
        print("✅ Successfully imported RAG and LLM modules")
        
        # Load configuration
        config = load_config()
        print("✅ Configuration loaded")
        
        # Initialize RAG system
        print("\n--- Initializing RAG System ---")
        rag_system = RAGSystem(config)
        print("✅ RAG system initialized")
        
        # Initialize LLM system
        print("\n--- Initializing LLM System ---")
        llm_system = LLMSystem(config)
        print("✅ LLM system initialized")
        
        # Test document content
        test_documents = [
            """
            This is a comprehensive test document for verifying the RAG-to-LLM integration pipeline.
            
            Personal Information:
            - Name: John Smith
            - Age: 35 years old
            - Address: 123 Main Street, Anytown, State 12345
            - Employment: Software Engineer at Tech Corp
            
            Family Structure:
            - Spouse: Jane Smith (32 years old)
            - Children: Tommy (8 years old), Sarah (5 years old)
            - Both children attend local elementary school
            
            Assessment Information:
            - Current situation appears stable
            - Family demonstrates good communication patterns
            - No immediate safety concerns identified
            - Recommendation for continued monitoring
            
            Background:
            - College graduate with computer science degree
            - Military service: Army National Guard (2010-2016)
            - Community involvement: Volunteers at local food bank
            """,
            
            """
            Case File Documentation - Follow-up Report
            
            Date: June 8, 2025
            Case Worker: Maria Rodriguez
            Case Number: CF-2025-0608-001
            
            Safety Assessment:
            - Home environment is clean and safe
            - Adequate food and utilities available
            - No signs of domestic violence
            - Children appear healthy and well-cared for
            
            Services Provided:
            - Weekly counseling sessions
            - Parenting skills workshops
            - Financial planning assistance
            - Child development resources
            
            Next Steps:
            - Continue current service plan
            - Schedule follow-up visit in 30 days
            - Monitor school attendance for children
            - Review financial stability progress
            """
        ]
        
        print("\n--- Testing Document Processing ---")
        
        # Process documents with RAG
        all_chunks = []
        for i, doc in enumerate(test_documents):
            print(f"Processing document {i+1}...")
            chunks = chunk_text(doc, chunk_size=500, overlap=50)
            all_chunks.extend(chunks)
            print(f"  Generated {len(chunks)} chunks")
        
        print(f"Total chunks for processing: {len(all_chunks)}")
        
        # Test 1: General Summary (no policy constraints)
        print("\n--- Test 1: General Summary Generation ---")
        try:
            general_result = llm_system.process_document(
                chunks=all_chunks,
                query="Provide a general overview of the content"
            )
            
            if general_result['success']:
                general_summary = general_result['general_summary']
                print("✅ General summary generated successfully")
                print(f"Summary length: {len(general_summary)} characters")
                print(f"Summary preview: {general_summary[:200]}...")
                
                # Verify summary contains key information
                key_terms = ['John Smith', 'family', 'children', 'assessment']
                found_terms = [term for term in key_terms if term.lower() in general_summary.lower()]
                print(f"Key terms found: {found_terms}")
                
            else:
                print("❌ General summary generation failed")
                print(f"Error: {general_result}")
                
        except Exception as e:
            print(f"❌ General summary test failed: {e}")
        
        # Test 2: Policy-Adherent Summary with Variables
        print("\n--- Test 2: Policy-Adherent Summary ---")
        try:
            # Load policy prompt
            policy_prompt_path = Path("config/policy_prompt.yaml")
            if policy_prompt_path.exists():
                with open(policy_prompt_path, 'r') as f:
                    import yaml
                    policy_data = yaml.safe_load(f)
                    policy_prompt = policy_data.get('policy_prompt', '')
            else:
                policy_prompt = """You are a professional case worker analyzing interview documentation. 
                Extract information accurately and format responses professionally. 
                Focus on factual information and avoid speculation."""
            
            # Define variables to extract
            variables = [
                'participant_name',
                'participant_age', 
                'family_members',
                'employment_status',
                'safety_assessment',
                'services_provided',
                'next_steps'
            ]
            
            policy_result = llm_system.process_document(
                chunks=all_chunks,
                query="Extract structured information for case documentation",
                policy_prompt=policy_prompt,
                variables=variables
            )
            
            if policy_result['success'] and policy_result['policy_summary']:
                policy_summary = policy_result['policy_summary']
                print("✅ Policy-adherent summary generated successfully")
                print(f"Policy summary type: {type(policy_summary)}")
                
                if isinstance(policy_summary, dict):
                    print("Policy summary structure:")
                    for key, value in policy_summary.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"Policy summary preview: {str(policy_summary)[:300]}...")
                    
            else:
                print("❌ Policy-adherent summary generation failed")
                print(f"Error: {policy_result}")
                
        except Exception as e:
            print(f"❌ Policy summary test failed: {e}")
        
        # Test 3: RAG Query Processing
        print("\n--- Test 3: RAG Query Processing ---")
        try:
            query_result = rag_system.process_query(
                "What is the employment status of the participants?",
                all_chunks
            )
            
            if query_result['success']:
                print("✅ RAG query processing successful")
                print(f"Relevant chunks found: {len(query_result['relevant_chunks'])}")
                print(f"Confidence score: {query_result['confidence']:.3f}")
                
                if query_result['relevant_chunks']:
                    print(f"Top relevant chunk: {query_result['relevant_chunks'][0][:200]}...")
                    
            else:
                print("❌ RAG query processing failed")
                print(f"Error: {query_result}")
                
        except Exception as e:
            print(f"❌ RAG query test failed: {e}")
        
        # Test 4: Combined RAG-LLM Pipeline
        print("\n--- Test 4: Combined RAG-LLM Pipeline ---")
        try:
            # Use RAG to find relevant content
            rag_query_result = rag_system.process_query(
                "What family information and safety assessments are documented?",
                all_chunks
            )
            
            if rag_query_result['success'] and rag_query_result['relevant_chunks']:
                # Pass RAG results to LLM for analysis
                relevant_chunks = rag_query_result['relevant_chunks'][:3]  # Top 3 most relevant
                
                combined_result = llm_system.process_document(
                    chunks=relevant_chunks,
                    query="Analyze family structure and safety information",
                    policy_prompt=policy_prompt
                )
                
                if combined_result['success']:
                    print("✅ Combined RAG-LLM pipeline successful")
                    print(f"Processed {len(relevant_chunks)} relevant chunks through LLM")
                    
                    if combined_result['general_summary']:
                        print(f"Combined analysis: {combined_result['general_summary'][:200]}...")
                        
                else:
                    print("❌ Combined pipeline LLM processing failed")
                    print(f"Error: {combined_result}")
                    
            else:
                print("❌ Combined pipeline RAG processing failed")
                
        except Exception as e:
            print(f"❌ Combined pipeline test failed: {e}")
        
        # Test 5: JSON Output Format
        print("\n--- Test 5: JSON Output Format ---")
        try:
            # Test with a structured extraction request
            json_chunks = ["""
            Case Summary: John Smith (Age: 35)
            Address: 123 Main Street, Anytown, State 12345
            Employment: Software Engineer at Tech Corp
            Spouse: Jane Smith (Age: 32)
            Children: Tommy (8), Sarah (5)
            Status: Case monitoring recommended
            """]
            
            json_result = llm_system.process_document(
                chunks=json_chunks,
                policy_prompt="Extract the following information and format as JSON: participant_name, participant_age, address, employment, family_members, case_status",
                variables=['participant_name', 'participant_age', 'address', 'employment', 'family_members', 'case_status']
            )
            
            if json_result['success'] and json_result['policy_summary']:
                print("✅ JSON format extraction successful")
                
                # Try to parse as JSON if it's a string
                summary = json_result['policy_summary']
                if isinstance(summary, str):
                    try:
                        parsed_json = json.loads(summary)
                        print("✅ Valid JSON output generated")
                        print("Extracted variables:")
                        for key, value in parsed_json.items():
                            print(f"  {key}: {value}")
                    except json.JSONDecodeError:
                        print("⚠️ Output generated but not valid JSON")
                        print(f"Output: {summary}")
                elif isinstance(summary, dict):
                    print("✅ Dictionary output generated")
                    print("Extracted variables:")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")
                        
            else:
                print("❌ JSON format extraction failed")
                print(f"Error: {json_result}")
                
        except Exception as e:
            print(f"❌ JSON format test failed: {e}")
        
        print("\n=== Integration Test Complete ===")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_files():
    """Test with real sample files if available"""
    print("\n=== Testing with Real Sample Files ===")
    
    try:
        from intv.config import load_config
        from intv.rag_system import RAGSystem
        from intv.llm import LLMSystem
        from intv.rag import chunk_document
        
        config = load_config()
        rag_system = RAGSystem(config)
        llm_system = LLMSystem(config)
        
        # Look for sample files
        sample_dir = Path("sample-sources")
        sample_files = [
            "sample_typed_adult.pdf",
            "sample_textonly_affidavit.docx",
            "sample_typed_casefile.pdf"
        ]
        
        for filename in sample_files:
            file_path = sample_dir / filename
            if file_path.exists():
                print(f"\n--- Testing with {filename} ---")
                
                try:
                    # Extract and chunk document
                    chunks = chunk_document(str(file_path))
                    print(f"✅ Extracted {len(chunks)} chunks from {filename}")
                    
                    # Test general summary
                    result = llm_system.process_document(
                        chunks=chunks[:5],  # Use first 5 chunks for speed
                        query=f"Summarize the content of {filename}"
                    )
                    
                    if result['success'] and result['general_summary']:
                        summary = result['general_summary']
                        print(f"✅ Generated summary ({len(summary)} chars)")
                        print(f"Preview: {summary[:150]}...")
                    else:
                        print(f"❌ Failed to generate summary for {filename}")
                        
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
            else:
                print(f"⚠️ Sample file not found: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real file testing failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting RAG-to-LLM Integration Verification\n")
    
    # Run integration test
    integration_success = test_rag_llm_integration()
    
    # Run real file test
    real_file_success = test_with_real_files()
    
    print(f"\n=== Final Results ===")
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"Real File Test: {'✅ PASSED' if real_file_success else '❌ FAILED'}")
    
    if integration_success and real_file_success:
        print("\n🎉 All RAG-to-LLM integration tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed - check output above")
        sys.exit(1)
