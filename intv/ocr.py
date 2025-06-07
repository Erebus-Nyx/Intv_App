"""
OCR module for INTV - handles Optical Character Recognition
"""

import os
import logging
from typing import Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile


def ocr_image(image_path: str, config_string: str = "--psm 6") -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to image file
        config_string: Tesseract configuration string
        
    Returns:
        Extracted text
    """
    try:
        # Open image with PIL
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image, config=config_string)
        
        return text.strip()
    except Exception as e:
        logging.error(f"OCR failed for image {image_path}: {e}")
        return ""


def ocr_file(file_path: str, config_string: str = "--psm 6") -> str:
    """
    Extract text from various file types using OCR.
    Supports images (PNG, JPG, etc.) and PDFs.
    
    Args:
        file_path: Path to file
        config_string: Tesseract configuration string
        
    Returns:
        Extracted text
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Image extensions
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    
    if ext in image_exts:
        return ocr_image(file_path, config_string)
    elif ext == '.pdf':
        return ocr_pdf(file_path, config_string)
    else:
        logging.warning(f"Unsupported file type for OCR: {ext}")
        return ""


def ocr_pdf(pdf_path: str, config_string: str = "--psm 6") -> str:
    """
    Extract text from PDF using OCR.
    Converts PDF pages to images first, then applies OCR.
    
    Args:
        pdf_path: Path to PDF file
        config_string: Tesseract configuration string
        
    Returns:
        Extracted text from all pages
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        all_text = []
        
        for i, image in enumerate(images):
            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path, 'PNG')
            
            try:
                # Extract text from this page
                page_text = ocr_image(temp_path, config_string)
                if page_text.strip():
                    all_text.append(f"--- Page {i+1} ---\n{page_text}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        return "\n\n".join(all_text)
    
    except Exception as e:
        logging.error(f"OCR failed for PDF {pdf_path}: {e}")
        return ""


def ocr_pdf_page(pdf_path: str, page_num: int, config_string: str = "--psm 6") -> str:
    """
    Extract text from a specific PDF page using OCR.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        config_string: Tesseract configuration string
        
    Returns:
        Extracted text from the specified page
    """
    try:
        # Convert specific page to image
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        
        if not images:
            return ""
        
        image = images[0]
        
        # Create temporary file for the image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path, 'PNG')
        
        try:
            # Extract text from this page
            return ocr_image(temp_path, config_string)
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    except Exception as e:
        logging.error(f"OCR failed for PDF page {page_num} in {pdf_path}: {e}")
        return ""


def preprocess_image(image_path: str, output_path: str = None) -> str:
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to input image
        output_path: Path for preprocessed image (optional)
        
    Returns:
        Path to preprocessed image
    """
    try:
        from PIL import ImageEnhance, ImageFilter
        
        # Open image
        image = Image.open(image_path)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.BLUR)
        
        # Save preprocessed image
        if output_path is None:
            output_path = image_path.replace('.', '_preprocessed.')
        
        image.save(output_path)
        return output_path
        
    except Exception as e:
        logging.error(f"Image preprocessing failed: {e}")
        return image_path  # Return original path if preprocessing fails


def main():
    """Entry point for intv-ocr command"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="INTV OCR - Extract text from images and PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help='Path to image or PDF file')
    parser.add_argument('-o', '--output', help='Output text file path (optional)')
    parser.add_argument('--config', default='--psm 6', help='Tesseract configuration string')
    parser.add_argument('--preprocess', action='store_true', help='Apply image preprocessing before OCR')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        input_path = args.input
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Error: File not found: {input_path}")
            return 1
        
        # Determine file type and process
        if input_path.lower().endswith('.pdf'):
            print(f"Processing PDF: {input_path}")
            text = ocr_pdf(input_path, config_string=args.config)
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            print(f"Processing image: {input_path}")
            
            # Apply preprocessing if requested
            if args.preprocess:
                print("Applying image preprocessing...")
                input_path = preprocess_image(input_path)
            
            text = ocr_image(input_path, config_string=args.config)
        else:
            print(f"Error: Unsupported file type: {input_path}")
            print("Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP")
            return 1
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text extracted and saved to: {args.output}")
        else:
            print("Extracted text:")
            print("-" * 50)
            print(text)
        
        return 0
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
