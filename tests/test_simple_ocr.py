#!/usr/bin/env python3
"""
Simple OCR test with pre-existing image file
"""

import os
import tempfile
import subprocess
import json

def test_simple_ocr():
    """Test OCR with a simple approach"""
    print("Testing Simple OCR Processing")
    print("=" * 40)
    
    # First, let's check if we can find any existing image files
    test_dirs = ["/home/nyx/intv", "/tmp"]
    
    for test_dir in test_dirs:
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(root, file)
                    print(f"Found image: {image_path}")
                    
                    # Test OCR on this image
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_output:
                        output_file = tmp_output.name
                    
                    cmd = [
                        "/home/nyx/intv/.venv/bin/python", "-m", "intv.pipeline_cli",
                        "--files", image_path,
                        "--format", "text",
                        "--verbose",
                        "--output", output_file
                    ]
                    
                    print(f"Testing OCR on: {image_path}")
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        if result.returncode == 0:
                            print("✅ OCR processing successful")
                            if os.path.exists(output_file):
                                with open(output_file, 'r') as f:
                                    content = f.read().strip()
                                    print(f"✅ Extracted {len(content)} characters")
                                    if content:
                                        print(f"Preview: {content[:200]}...")
                            else:
                                print("⚠️  Output file not created")
                        else:
                            print(f"❌ OCR failed: {result.stderr}")
                            
                        # Clean up
                        if os.path.exists(output_file):
                            os.unlink(output_file)
                            
                        return  # Exit after testing first image
                        
                    except subprocess.TimeoutExpired:
                        print("❌ OCR processing timed out")
                        return
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        return
    
    print("No existing images found. Creating a simple test image...")
    
    # Create a simple test image using command line tools
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        image_path = tmp_img.name
    
    # Create simple image with ImageMagick if available
    try:
        subprocess.run([
            'convert', '-size', '400x100', 'xc:white',
            '-pointsize', '20', '-fill', 'black',
            '-annotate', '+50+50', 'This is a test document',
            image_path
        ], check=True, timeout=10)
        
        print(f"Created test image: {image_path}")
        
        # Test OCR
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_output:
            output_file = tmp_output.name
        
        cmd = [
            "/home/nyx/intv/.venv/bin/python", "-m", "intv.pipeline_cli",
            "--files", image_path,
            "--format", "text",
            "--verbose",
            "--output", output_file
        ]
        
        print("Testing OCR on created image...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ OCR processing successful")
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read().strip()
                    print(f"✅ Extracted {len(content)} characters")
                    if content:
                        print(f"Content: {content}")
        else:
            print(f"❌ OCR failed: {result.stderr}")
        
        # Clean up
        os.unlink(image_path)
        if os.path.exists(output_file):
            os.unlink(output_file)
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ImageMagick not available, skipping image creation test")
        print("OCR functionality test cannot be completed without test images")

if __name__ == "__main__":
    test_simple_ocr()
