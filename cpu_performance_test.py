#!/usr/bin/env python3
"""
CPU Performance Test for INTV RAG System
Tests the performance of the RAG system in CPU-only mode
"""

import time
import os
import sys
import platform
import psutil

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CPU'] = '1'

def main():
    print("=== INTV CPU Performance Test ===")
    print()
    
    # System information
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    
    try:
        # Import and test RAG system
        print("Importing RAG system...")
        from intv.rag_system import RAGSystem
        print("✅ RAG system imported successfully")
        
        # Test initialization
        print("Initializing RAG system (CPU mode)...")
        start_time = time.time()
        
        # Create with CPU-optimized config
        config = {
            'rag': {
                'mode': 'embedded',
                'embedded': {
                    'model': 'auto',
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'top_k': 3
                }
            },
            'model_dir': 'models'
        }
        
        rag = RAGSystem(config)
        
        init_time = time.time() - start_time
        print(f"✅ Initialization completed in {init_time:.1f} seconds")
        
        # Test document processing
        test_text = """
        This is a comprehensive test document for measuring RAG system performance in CPU-only mode.
        
        The document contains multiple paragraphs to evaluate chunking efficiency and embedding generation speed.
        Performance metrics from this test will provide realistic expectations for CPU-only deployments.
        
        This includes testing text processing, semantic chunking, embedding generation, and query processing.
        The results will help users understand what to expect when running on systems without GPU acceleration.
        
        Additional content is included to ensure we have enough text for meaningful performance measurements.
        The system should handle this content efficiently even on modest hardware configurations.
        """
        
        print("=== Performance Testing ===")
        
        # Import chunking functions
        from intv.rag import chunk_text, get_rag_system
        
        # Test chunking
        print("Testing document chunking...")
        start_time = time.time()
        chunks = chunk_text(test_text, chunk_size=500, overlap=50)
        chunk_time = time.time() - start_time
        print(f"✅ Chunked into {len(chunks)} chunks in {chunk_time:.3f} seconds")
        
        # Test embedding generation by testing query processing
        print("Testing embedding generation...")
        start_time = time.time()
        
        # This will trigger embedding generation for both query and chunks
        result = rag.process_query("What is this document about?", chunks)
        
        processing_time = time.time() - start_time
        print(f"✅ Embeddings generated and processed in {processing_time:.3f} seconds")
        
        # Test additional queries
        print("Testing additional query processing...")
        query_times = []
        
        test_queries = [
            "How does the system handle performance?",
            "What are the hardware requirements?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            query_result = rag.process_query(query, chunks)
            query_time = time.time() - start_time
            query_times.append(query_time)
            relevant_chunks = query_result.get('relevant_chunks', [])
            print(f"✅ Query {i} processed in {query_time:.3f} seconds ({len(relevant_chunks)} results)")
        
        avg_query_time = sum(query_times) / len(query_times)
        
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)
        
        print()
        print("=== CPU Performance Summary ===")
        print(f"Initialization Time: {init_time:.1f} seconds")
        print(f"Text Chunking: {chunk_time:.3f} seconds for {len(chunks)} chunks")
        print(f"First Query Processing: {processing_time:.3f} seconds (includes embeddings)")
        print(f"Average Query Time: {avg_query_time:.3f} seconds")
        print(f"Memory Usage: {memory_mb:.1f} MB")
        
        # Calculate processing rates
        if processing_time > 0:
            chunks_per_second = len(chunks) / processing_time
            print(f"Processing Rate: {chunks_per_second:.2f} chunks/second")
        
        # Performance tier assessment
        print()
        print("=== Performance Assessment ===")
        if init_time < 60:
            print("✅ Good initialization performance")
        elif init_time < 120:
            print("🔶 Moderate initialization performance")
        else:
            print("🔶 Slow initialization (consider lighter models)")
            
        if avg_query_time < 2:
            print("✅ Good query performance")
        elif avg_query_time < 5:
            print("🔶 Moderate query performance")
        else:
            print("🔶 Slow query performance")
            
        return {
            'init_time': init_time,
            'chunk_time': chunk_time,
            'processing_time': processing_time,
            'avg_query_time': avg_query_time,
            'memory_mb': memory_mb,
            'chunks_count': len(chunks),
            'chunks_per_second': len(chunks) / processing_time if processing_time > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Error during CPU performance test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("\n=== Test completed successfully ===")
    else:
        print("\n=== Test failed ===")
        sys.exit(1)
