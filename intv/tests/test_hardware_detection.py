#!/usr/bin/env python3
"""Test hardware detection and classification"""

import psutil
import platform
import subprocess
import sys

def detect_gpu():
    """Detect GPU using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                if line.strip():
                    memory, name = line.split(',', 1)
                    gpus.append({'name': name.strip(), 'memory_mb': int(memory), 'memory_gb': int(memory)/1024})
            return gpus
    except Exception as e:
        print(f"GPU detection failed: {e}")
    return []

def classify_system():
    """Classify system based on hardware"""
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"CPU cores: {cpu_count}")
    print(f"RAM: {memory_gb:.1f} GB")
    
    gpus = detect_gpu()
    if gpus:
        for gpu in gpus:
            print(f"GPU: {gpu['name']} ({gpu['memory_gb']:.1f} GB VRAM)")
        
        # Check for high-end GPU system
        max_gpu_memory = max(gpu['memory_gb'] for gpu in gpus)
        if cpu_count >= 16 and memory_gb >= 32 and max_gpu_memory >= 12:
            return 'gpu_high'
        elif cpu_count >= 8 and memory_gb >= 16 and max_gpu_memory >= 8:
            return 'gpu_medium'
        elif max_gpu_memory >= 4:
            return 'gpu_low'
    
    # CPU-only classification
    if cpu_count >= 16 and memory_gb >= 32:
        return 'cpu_high'
    elif cpu_count >= 8 and memory_gb >= 16:
        return 'cpu_medium'
    else:
        return 'cpu_low'

if __name__ == "__main__":
    print("=== HARDWARE DETECTION TEST ===")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    classification = classify_system()
    print(f"\nSystem classification: {classification}")
    
    # Model recommendations
    models = {
        'cpu_low': 'sentence-transformers/all-MiniLM-L6-v2',
        'cpu_medium': 'sentence-transformers/all-mpnet-base-v2', 
        'cpu_high': 'sentence-transformers/all-mpnet-base-v2',
        'gpu_low': 'sentence-transformers/all-mpnet-base-v2',
        'gpu_medium': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
        'gpu_high': 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    }
    
    recommended_model = models.get(classification, 'sentence-transformers/all-MiniLM-L6-v2')
    print(f"Recommended model: {recommended_model}")
