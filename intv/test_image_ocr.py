#!/usr/bin/env python3
"""
Test OCR functionality with actual image files
"""

import os
import tempfile
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont

def create_test_image_with_text(text: str, filename: str) -> str:
    """Create a test image with readable text"""
    # Create a white background image
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Draw text on image
    draw.text((50, 80), text, fill='black', font=font)
    
    # Save image
    img.save(filename)
    return filename

def test_image_ocr():
    """Test OCR processing with different image formats"""
    print("Testing Image OCR Processing Pipeline")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test cases
        test_cases = [
            {
                "name": "Simple Text Image (PNG)",
                "text": "This is a test document for OCR processing.\nIt contains multiple lines of text.",
                "format": "PNG"
            },
            {
                "name": "Interview Notes Image (JPG)", 
                "text": "Interview Notes:\n- Candidate has 5 years experience\n- Strong technical skills\n- Good communication",
                "format": "JPEG"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {test_case['name']} ---")
            
            # Create test image
            ext = "png" if test_case["format"] == "PNG" else "jpg"
            image_path = os.path.join(temp_dir, f"test_image_{i}.{ext}")
            create_test_image_with_text(test_case["text"], image_path)
            print(f"Created test image: {image_path}")
            
            # Test basic OCR processing
            output_file = os.path.join(temp_dir, f"ocr_output_{i}.txt")
            cmd = [
                "/home/nyx/intv/.venv/bin/python", "-m", "intv.pipeline_cli",
                "--files", image_path,
                "--format", "text",
                "--verbose",
                "--output", output_file
            ]
            
            print(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ OCR processing successful")
                    
                    # Check if output file was created and has content
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            content = f.read().strip()
                            if content:
                                print(f"✅ OCR extracted text (length: {len(content)} chars)")
                                print(f"Preview: {content[:100]}...")
                            else:
                                print("⚠️  OCR output file is empty")
                    else:
                        print("⚠️  OCR output file not created")
                else:
                    print(f"❌ OCR processing failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("❌ OCR processing timed out")
            except Exception as e:
                print(f"❌ OCR processing error: {e}")
            
            # Test JSON output format
            json_output = os.path.join(temp_dir, f"ocr_output_{i}.json")
            cmd_json = [
                "/home/nyx/intv/.venv/bin/python", "-m", "intv.pipeline_cli",
                "--files", image_path,
                "--format", "json",
                "--verbose",
                "--output", json_output
            ]
            
            print(f"Testing JSON format...")
            try:
                result = subprocess.run(cmd_json, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    if os.path.exists(json_output):
                        with open(json_output, 'r') as f:
                            json_data = json.load(f)
                            print("✅ JSON OCR output is valid")
                            if 'extracted_text' in json_data or 'content' in json_data:
                                print("✅ JSON contains extracted text")
                            else:
                                print("⚠️  JSON missing expected text fields")
                    else:
                        print("⚠️  JSON output file not created")
                else:
                    print(f"❌ JSON OCR processing failed: {result.stderr}")
            except Exception as e:
                print(f"❌ JSON OCR processing error: {e}")
                
            # Test with module processing
            if i == 1:  # Only test module on first image
                module_output = os.path.join(temp_dir, f"ocr_module_output_{i}.json")
                cmd_module = [
                    "/home/nyx/intv/.venv/bin/python", "-m", "intv.pipeline_cli",
                    "--files", image_path,
                    "--format", "json",
                    "--module", "adult",
                    "--verbose",
                    "--output", module_output
                ]
                
                print(f"Testing with module processing...")
                try:
                    result = subprocess.run(cmd_module, capture_output=True, text=True, timeout=90)
                    if result.returncode == 0:
                        if os.path.exists(module_output):
                            with open(module_output, 'r') as f:
                                json_data = json.load(f)
                                print("✅ Module OCR processing successful")
                                if 'llm_analysis' in json_data or 'analysis' in json_data:
                                    print("✅ Module analysis included in output")
                                else:
                                    print("⚠️  Module analysis missing from output")
                        else:
                            print("⚠️  Module output file not created")
                    else:
                        print(f"❌ Module OCR processing failed: {result.stderr}")
                except Exception as e:
                    print(f"❌ Module OCR processing error: {e}")

if __name__ == "__main__":
    test_image_ocr()
