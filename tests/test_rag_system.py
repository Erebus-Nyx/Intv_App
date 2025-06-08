#!/usr/bin/env python3
"""
Test script for the enhanced RAG system integration
Tests both embedded and external RAG modes with sample documents
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from intv.rag_system import RAGSystem, SystemCapabilities
from intv.rag import enhanced_rag_pipeline, enhanced_query_documents, get_rag_system
from intv.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_detection():
    """Test system capability detection"""
    logger.info("=== Testing System Capability Detection ===")
    
    try:
        capabilities = SystemCapabilities.detect_system_type()
        logger.info(f"System Type: {capabilities.system_type}")
        logger.info(f"RAM: {capabilities.total_ram_gb:.1f} GB")
        logger.info(f"CPU Cores: {capabilities.cpu_cores}")
        logger.info(f"Has CUDA: {capabilities.has_cuda}")
        logger.info(f"CUDA Devices: {capabilities.cuda_device_count}")
        logger.info(f"Recommended Model: {capabilities.recommended_model}")
        return True
    except Exception as e:
        logger.error(f"System detection failed: {e}")
        return False

def test_embedded_rag():
    """Test embedded RAG with local models"""
    logger.info("=== Testing Embedded RAG System ===")
    
    try:
        # Load config with embedded RAG settings
        config = {
            'rag': {
                'mode': 'embedded',
                'embedded': {
                    'model': 'auto',
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'top_k': 3
                }
            }
        }
        
        # Initialize RAG system
        rag_system = RAGSystem(config)
        logger.info("RAG system initialized successfully")
        
        # Test with sample documents
        sample_dir = project_root / "sample-sources"
        test_files = [
            "sample_typed_adult.pdf",
            "sample_textonly_affidavit.docx"
        ]
        
        documents = []
        file_paths = []
        
        for filename in test_files:
            file_path = sample_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        documents.append(f.read())
                    file_paths.append(str(file_path))
                    logger.info(f"Loaded document: {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        if not documents:
            logger.error("No documents could be loaded for testing")
            return False
        
        # Test document chunking
        logger.info("Testing document chunking...")
        chunk_results = rag_system.chunk_documents(documents, file_paths)
        
        total_chunks = 0
        for i, result in enumerate(chunk_results):
            chunks = result['chunks']
            total_chunks += len(chunks)
            logger.info(f"Document {i+1}: {len(chunks)} chunks")
        
        if total_chunks == 0:
            logger.error("No chunks were generated")
            return False
        
        # Test querying
        logger.info("Testing document querying...")
        all_chunks = []
        for result in chunk_results:
            all_chunks.extend(result['chunks'])
        
        test_queries = [
            "What is the document about?",
            "Who are the parties involved?",
            "What are the key details?"
        ]
        
        for query in test_queries:
            logger.info(f"Query: {query}")
            query_result = rag_system.query(query, all_chunks, top_k=2)
            relevant_chunks = query_result.get('chunks', [])
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            for j, chunk in enumerate(relevant_chunks):
                logger.info(f"  Chunk {j+1}: {chunk[:100]}...")
        
        logger.info("Embedded RAG test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Embedded RAG test failed: {e}")
        return False

def test_enhanced_rag_functions():
    """Test the enhanced RAG functions in rag.py"""
    logger.info("=== Testing Enhanced RAG Functions ===")
    
    try:
        # Load config
        config = load_config()
        
        # Test sample document paths
        sample_dir = project_root / "sample-sources"
        test_files = [
            str(sample_dir / "sample_typed_adult.pdf"),
            str(sample_dir / "sample_textonly_affidavit.docx")
        ]
        
        # Filter existing files
        existing_files = [f for f in test_files if Path(f).exists()]
        if not existing_files:
            logger.error("No test files found")
            return False
        
        # Test enhanced RAG pipeline
        logger.info("Testing enhanced RAG pipeline...")
        result = enhanced_rag_pipeline(
            document_paths=existing_files,
            query="What are the main topics discussed in these documents?",
            config=config
        )
        
        if result['success']:
            logger.info(f"Pipeline success: {len(result['relevant_chunks'])} relevant chunks found")
            logger.info(f"Metadata: {result['metadata']}")
            
            # Show first relevant chunk
            if result['relevant_chunks']:
                first_chunk = result['relevant_chunks'][0]
                logger.info(f"First relevant chunk: {first_chunk[:200]}...")
            
            return True
        else:
            logger.error(f"Enhanced RAG pipeline failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Enhanced RAG functions test failed: {e}")
        return False

def test_rag_integration():
    """Test the complete RAG integration"""
    logger.info("=== Testing Complete RAG Integration ===")
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Test with a sample document
        sample_file = project_root / "sample-sources" / "sample_typed_adult.pdf"
        if not sample_file.exists():
            logger.error(f"Test file not found: {sample_file}")
            return False
        
        # Process document
        logger.info(f"Processing document: {sample_file}")
        result = orchestrator.process(
            input_path=sample_file,
            query="What is this document about?",
            apply_llm=True
        )
        
        if result.success:
            logger.info("Document processing successful")
            logger.info(f"Extraction method: {result.extraction_method}")
            logger.info(f"Text length: {len(result.extracted_text) if result.extracted_text else 0}")
            logger.info(f"Chunks: {len(result.chunks) if result.chunks else 0}")
            
            if result.rag_result:
                logger.info("RAG processing completed")
                if isinstance(result.rag_result, dict):
                    if 'relevant_chunks' in result.rag_result:
                        logger.info(f"Found {len(result.rag_result['relevant_chunks'])} relevant chunks")
            
            return True
        else:
            logger.error(f"Document processing failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"RAG integration test failed: {e}")
        return False

def main():
    """Run all RAG system tests"""
    logger.info("Starting RAG System Tests")
    
    tests = [
        ("System Detection", test_system_detection),
        ("Embedded RAG", test_embedded_rag),
        ("Enhanced RAG Functions", test_enhanced_rag_functions),
        ("RAG Integration", test_rag_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    logger.info(f"{'='*60}")
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! RAG system is working correctly.")
        return 0
    else:
        logger.error(f"{total - passed} tests failed. Please check the logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
