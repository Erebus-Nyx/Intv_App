#!/usr/bin/env python3
"""
GPU Performance Test for INTV RAG System
Tests the performance of the RAG system with GPU acceleration
"""

import time
import os
import sys
import platform
import psutil

def main():
    print("=== INTV GPU Performance Test ===")
    print()
    
    # System information
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("GPU: CUDA not available")
    except ImportError:
        print("GPU: PyTorch not available")
    print()
    
    try:
        # Import and test RAG system
        print("Importing RAG system...")
        from intv.rag_system import RAGSystem
        from intv.rag import chunk_text
        print("✅ RAG system imported successfully")
        
        # Test initialization with GPU
        print("Initializing RAG system (GPU mode)...")
        start_time = time.time()
        
        # Create with GPU-optimized config
        config = {
            'rag': {
                'mode': 'embedded',
                'embedded': {
                    'model': 'auto',
                    'chunk_size': 1000,
                    'overlap': 100,
                    'top_k': 5
                }
            },
            'model_dir': 'models'
        }
        
        rag = RAGSystem(config)
        
        init_time = time.time() - start_time
        print(f"✅ Initialization completed in {init_time:.1f} seconds")
        
        # Test document processing
        test_text = """
        This is a comprehensive test document for measuring RAG system performance with GPU acceleration.
        
        The document contains multiple paragraphs to evaluate chunking efficiency and embedding generation speed.
        Performance metrics from this test will provide realistic expectations for GPU-accelerated deployments.
        
        This includes testing text processing, semantic chunking, embedding generation, and query processing.
        The results will help users understand the performance benefits of GPU acceleration for RAG operations.
        
        Additional content is included to ensure we have enough text for meaningful performance measurements.
        The system should handle this content very efficiently with GPU acceleration enabled.
        
        GPU acceleration typically provides significant speedups for embedding generation and similarity calculations.
        This test will measure the actual performance improvements compared to CPU-only processing.
        
        Large language models and embedding models benefit greatly from parallel processing capabilities.
        Modern GPUs can process many text chunks simultaneously, leading to substantial performance gains.
        """
        
        print("=== Performance Testing ===")
        
        # Test chunking
        print("Testing document chunking...")
        start_time = time.time()
        chunks = chunk_text(test_text, chunk_size=1000, overlap=100)
        chunk_time = time.time() - start_time
        print(f"✅ Chunked into {len(chunks)} chunks in {chunk_time:.3f} seconds")
        
        # Test embedding generation by testing query processing
        print("Testing embedding generation with GPU...")
        start_time = time.time()
        
        # This will trigger embedding generation for both query and chunks
        result = rag.process_query("What is this document about?", chunks)
        
        first_processing_time = time.time() - start_time
        print(f"✅ First query processed in {first_processing_time:.3f} seconds (includes embedding generation)")
        
        # Test additional queries (embeddings should be cached)
        print("Testing cached query processing...")
        query_times = []
        
        test_queries = [
            "How does GPU acceleration improve performance?",
            "What are the benefits of parallel processing?",
            "How do embedding models work with GPUs?",
            "What performance improvements can be expected?",
            "How does chunking affect processing speed?"
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
        
        # GPU memory usage if available
        gpu_memory_mb = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024**2)
                print(f"GPU Memory Used: {gpu_memory_mb:.1f} MB")
        except:
            pass
        
        print()
        print("=== GPU Performance Summary ===")
        print(f"Initialization Time: {init_time:.1f} seconds")
        print(f"Text Chunking: {chunk_time:.3f} seconds for {len(chunks)} chunks")
        print(f"First Query Processing: {first_processing_time:.3f} seconds (includes embeddings)")
        print(f"Average Query Time: {avg_query_time:.3f} seconds")
        print(f"System Memory Usage: {memory_mb:.1f} MB")
        if gpu_memory_mb > 0:
            print(f"GPU Memory Usage: {gpu_memory_mb:.1f} MB")
        
        # Calculate processing rates
        if first_processing_time > 0:
            chunks_per_second = len(chunks) / first_processing_time
            print(f"Processing Rate: {chunks_per_second:.2f} chunks/second")
        
        # Performance tier assessment
        print()
        print("=== Performance Assessment ===")
        if init_time < 30:
            print("✅ Excellent initialization performance")
        elif init_time < 60:
            print("✅ Good initialization performance")
        else:
            print("🔶 Moderate initialization performance")
            
        if avg_query_time < 0.5:
            print("✅ Excellent query performance")
        elif avg_query_time < 1.0:
            print("✅ Good query performance")
        elif avg_query_time < 2.0:
            print("🔶 Moderate query performance")
        else:
            print("🔶 Slow query performance")
            
        return {
            'init_time': init_time,
            'chunk_time': chunk_time,
            'first_processing_time': first_processing_time,
            'avg_query_time': avg_query_time,
            'memory_mb': memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'chunks_count': len(chunks),
            'chunks_per_second': len(chunks) / first_processing_time if first_processing_time > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Error during GPU performance test: {e}")
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
