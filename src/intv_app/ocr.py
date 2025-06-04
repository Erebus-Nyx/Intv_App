import logging
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from typing import Optional

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

def preprocess_image(
    img: Image.Image,
    binarize: bool = True,
    denoise: bool = True,
    resize: Optional[tuple] = None
) -> Image.Image:
    """
    Preprocess an image for OCR: binarization, denoising, resizing.
    - binarize: convert to grayscale and apply thresholding (autocontrast + binary)
    - denoise: apply median filter to reduce noise
    - resize: resize to given (width, height) if provided
    Returns a processed PIL Image ready for OCR.
    """
    if resize:
        img = img.resize(resize, Image.LANCZOS)
    if binarize:
        img = img.convert('L')
        img = ImageOps.autocontrast(img)
        # Otsu's thresholding could be used for more advanced binarization
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def ocr_image(
    img: Image.Image,
    lang: str = 'eng',
    config: str = '--psm 3'
) -> str:
    """
    Run OCR on a PIL image with pytesseract.
    Returns extracted text (str). Logs warning if output is empty or too short.
    """
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        if not text or len(text.strip()) < 5:
            logging.warning("OCR output is empty or too short.")
        return text
    except Exception as e:
        logging.warning(f"OCR failed for image: {e}")
        return ""

def ocr_file(
    filepath: str,
    lang: str = 'eng',
    config: str = '--psm 3',
    preprocess: bool = True,
    resize: Optional[tuple] = None
) -> str:
    """
    Run OCR on an image file (any format supported by PIL).
    """
    try:
        img = Image.open(filepath)
        if preprocess:
            img = preprocess_image(img, resize=resize)
        return ocr_image(img, lang=lang, config=config)
    except Exception as e:
        logging.warning(f"OCR failed for image {filepath}: {e}")
        return ""

def ocr_pdf_page(
    pdf_path: str,
    page_number: int,
    lang: str = 'eng',
    config: str = '--psm 3',
    preprocess: bool = True,
    dpi: int = 300,
    resize: Optional[tuple] = None
) -> str:
    """
    Run OCR on a single page of a PDF (using pdf2image and pytesseract).
    page_number is 0-based.
    """
    if convert_from_path is None:
        raise ImportError('pdf2image is required for PDF OCR.')
    try:
        images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1, dpi=dpi)
        if images:
            img = images[0]
            if preprocess:
                img = preprocess_image(img, resize=resize)
            return ocr_image(img, lang=lang, config=config)
    except Exception as e:
        logging.warning(f"OCR failed for PDF page {page_number+1} in {pdf_path}: {e}")
    return ""
