#!/usr/bin/env python3
"""Test model detection and local file handling"""

from intv.rag_system import ModelDownloader
from pathlib import Path

def test_model_detection():
    """Test the ModelDownloader with various scenarios"""
    downloader = ModelDownloader('models')

    print('=== Testing Model Detection ===')

    # Test 1: Existing HF model (should be detected as already downloaded)
    test_model = 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1'
    print(f'1. HF Model {test_model}:')
    print(f'   Downloaded = {downloader.is_model_downloaded(test_model)}')

    # Test 2: Local file patterns
    local_tests = [
        'my-model.gguf',
        './local-model.safetensors', 
        '/absolute/path/model.bin',
        'models/existing-file.gguf',
        '../another-model.pt'
    ]

    print('\n2. Local Path Detection:')
    for test_file in local_tests:
        is_local = downloader.is_local_path(test_file)
        print(f'   "{test_file}": Is local = {is_local}')
        
    # Test 3: Create a fake local file and test detection
    print('\n3. Testing Fake Local File:')
    fake_local = Path('models/test-local.gguf')
    fake_local.parent.mkdir(exist_ok=True)
    fake_local.touch()

    print(f'   Created: {fake_local}')
    print(f'   Is local path: {downloader.is_local_path(str(fake_local))}')
    print(f'   Is downloaded: {downloader.is_model_downloaded(str(fake_local))}')

    # Cleanup
    fake_local.unlink()
    print('   Cleaned up fake file')

    print('\n=== Testing Parse Model String ===')
    test_strings = [
        'hf.co/microsoft/DialoGPT-medium',
        'microsoft/DialoGPT-medium:pytorch_model.bin',
        'my-local-model.gguf',
        './relative/path/model.safetensors',
        '/absolute/path/to/model'
    ]

    for test_str in test_strings:
        repo_id, filename = downloader.parse_model_string(test_str)
        print(f'   "{test_str}" -> repo_id: {repo_id}, filename: {filename}')

    print('\n=== Testing Cache Detection ===')
    # Now test that the existing model is detected properly
    existing_result = downloader.is_model_downloaded(test_model)
    print(f'HF model cache detection: {existing_result}')
    
    if existing_result:
        print('✅ Cache detection working - no unnecessary downloads!')
    else:
        print('❌ Cache detection failed - model would be re-downloaded')

if __name__ == '__main__':
    test_model_detection()
