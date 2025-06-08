#!/usr/bin/env python3
"""
Complete Performance Test for INTV RAG System
Tests both GPU and CPU modes to get comparison data
"""

import time
import os
import sys
import platform
import psutil

def test_cpu_only_mode():
    """Test with forced CPU-only mode"""
    print("=== CPU-Only Mode Test ===")
    
    # Force CPU mode environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['FORCE_CPU'] = '1'
    
    try:
        from intv.rag_system import RAGSystem
        from intv.rag import chunk_text
        
        # Create CPU-only config
        config = {
            'rag': {
                'mode': 'embedded',
                'embedded': {
                    'model': 'hf.co/sentence-transformers/all-MiniLM-L6-v2',  # Lighter model for CPU
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'top_k': 3
                }
            },
            'model_dir': 'models',
            'force_cpu': True
        }
        
        print("Initializing RAG system (CPU-only)...")
        start_time = time.time()
        rag = RAGSystem(config)
        init_time = time.time() - start_time
        
        # Test with realistic document content
        test_text = """
        This comprehensive performance evaluation document contains multiple sections designed to test the capabilities of the RAG system in CPU-only mode.
        
        Section 1: System Requirements
        The system must be able to process documents efficiently even on hardware without GPU acceleration. This includes handling text chunking, embedding generation, and semantic search operations.
        
        Section 2: Performance Expectations
        CPU-only systems should provide reasonable performance for document analysis tasks. While slower than GPU-accelerated systems, they should still complete operations in acceptable timeframes.
        
        Section 3: Memory Usage
        Memory consumption should be optimized for systems with limited RAM. The system should gracefully handle large documents by using appropriate chunking strategies.
        
        Section 4: Hardware Compatibility
        The system should work on various CPU architectures including x86_64, ARM, and other platforms. Automatic model downscaling should occur on lower-end hardware.
        
        Section 5: Processing Speed
        Processing speeds will vary based on CPU capabilities, but the system should maintain functionality across different hardware tiers.
        """
        
        # Test chunking
        start_time = time.time()
        chunks = chunk_text(test_text, chunk_size=400, overlap=50)
        chunk_time = time.time() - start_time
        
        # Test query processing
        queries = [
            "What are the system requirements?",
            "How does memory usage work?",
            "What about hardware compatibility?",
            "Tell me about processing speed",
            "What are the performance expectations?"
        ]
        
        query_times = []
        for query in queries:
            start_time = time.time()
            result = rag.process_query(query, chunks)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        
        return {
            'mode': 'CPU-Only',
            'init_time': init_time,
            'chunk_time': chunk_time,
            'chunks_count': len(chunks),
            'avg_query_time': avg_query_time,
            'memory_mb': memory_mb,
            'model': config['rag']['embedded']['model']
        }
        
    except Exception as e:
        print(f"CPU-only test failed: {e}")
        return None

def test_gpu_mode():
    """Test with GPU mode (if available)"""
    print("\n=== GPU Mode Test ===")
    
    # Clear CPU forcing
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    if 'FORCE_CPU' in os.environ:
        del os.environ['FORCE_CPU']
    
    try:
        from intv.rag_system import RAGSystem
        from intv.rag import chunk_text
        
        # Auto-config (should select GPU model if available)
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
        
        print("Initializing RAG system (Auto-detect)...")
        start_time = time.time()
        rag = RAGSystem(config)
        init_time = time.time() - start_time
        
        # Same test content as CPU test
        test_text = """
        This comprehensive performance evaluation document contains multiple sections designed to test the capabilities of the RAG system in GPU-accelerated mode.
        
        Section 1: System Requirements
        The system must be able to process documents efficiently with GPU acceleration. This includes handling text chunking, embedding generation, and semantic search operations.
        
        Section 2: Performance Expectations
        GPU-accelerated systems should provide high performance for document analysis tasks. Processing should be significantly faster than CPU-only operations.
        
        Section 3: Memory Usage
        GPU memory and system RAM usage should be optimized for high-performance processing while maintaining efficiency.
        
        Section 4: Hardware Compatibility
        The system should leverage available GPU resources when present, with automatic fallback to CPU processing.
        
        Section 5: Processing Speed
        Processing speeds should be optimized for GPU acceleration, providing rapid document analysis and query processing.
        """
        
        # Test chunking
        start_time = time.time()
        chunks = chunk_text(test_text, chunk_size=400, overlap=50)
        chunk_time = time.time() - start_time
        
        # Test query processing
        queries = [
            "What are the system requirements?",
            "How does memory usage work?",
            "What about hardware compatibility?",
            "Tell me about processing speed",
            "What are the performance expectations?"
        ]
        
        query_times = []
        for query in queries:
            start_time = time.time()
            result = rag.process_query(query, chunks)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        
        return {
            'mode': 'Auto-Detect (GPU if available)',
            'init_time': init_time,
            'chunk_time': chunk_time,
            'chunks_count': len(chunks),
            'avg_query_time': avg_query_time,
            'memory_mb': memory_mb,
            'model': 'auto-selected'
        }
        
    except Exception as e:
        print(f"GPU mode test failed: {e}")
        return None

def main():
    print("=== INTV RAG System Complete Performance Test ===")
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
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"GPU: {gpu_name} ({gpu_count} device(s))")
        else:
            print("GPU: Not available or CUDA not installed")
    except ImportError:
        print("GPU: PyTorch not available")
    
    print()
    
    # Run tests
    results = []
    
    # Test CPU-only mode
    cpu_result = test_cpu_only_mode()
    if cpu_result:
        results.append(cpu_result)
    
    # Test auto-detect mode (may use GPU)
    gpu_result = test_gpu_mode()
    if gpu_result:
        results.append(gpu_result)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['mode']}:")
        print(f"  Model: {result['model']}")
        print(f"  Initialization: {result['init_time']:.1f} seconds")
        print(f"  Chunking: {result['chunk_time']:.3f} seconds ({result['chunks_count']} chunks)")
        print(f"  Average Query: {result['avg_query_time']:.3f} seconds")
        print(f"  Memory Usage: {result['memory_mb']:.1f} MB")
        
        # Performance assessment
        if result['init_time'] < 30:
            init_rating = "Excellent"
        elif result['init_time'] < 60:
            init_rating = "Good"
        elif result['init_time'] < 120:
            init_rating = "Moderate"
        else:
            init_rating = "Slow"
            
        if result['avg_query_time'] < 0.1:
            query_rating = "Excellent"
        elif result['avg_query_time'] < 1:
            query_rating = "Good"
        elif result['avg_query_time'] < 3:
            query_rating = "Moderate"
        else:
            query_rating = "Slow"
            
        print(f"  Assessment: {init_rating} init, {query_rating} queries")
    
    # Generate README metrics
    if results:
        print("\n" + "="*60)
        print("README METRICS (for documentation update)")
        print("="*60)
        
        for result in results:
            if 'CPU' in result['mode']:
                print(f"\n**CPU-Only Systems:**")
                print(f"- **Initialization**: ~{result['init_time']:.0f} seconds")
                print(f"- **Query Speed**: {result['avg_query_time']:.2f} seconds average")
                print(f"- **Memory Usage**: {result['memory_mb']:.0f} MB")
                print(f"- **Model**: {result['model']}")
            else:
                print(f"\n**GPU/Auto-Detect Systems:**")
                print(f"- **Initialization**: ~{result['init_time']:.0f} seconds")
                print(f"- **Query Speed**: {result['avg_query_time']:.2f} seconds average")
                print(f"- **Memory Usage**: {result['memory_mb']:.0f} MB")

if __name__ == "__main__":
    main()
