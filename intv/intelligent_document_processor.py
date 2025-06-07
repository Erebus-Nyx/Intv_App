"""
Intelligent Document Processing Pathway for INTV Application
Determines whether documents contain extractible text or require OCR processing.
Provides a single unified endpoint for all document types.
"""

import os
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import tempfile

# PDF text extraction libraries
try:
    from PyPDF2 import PdfReader
    import fitz  # PyMuPDF - better for text detection
except ImportError:
    PdfReader = None
    fitz = None

# Document libraries  
try:
    import docx
except ImportError:
    docx = None

# Image processing libraries
try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

# Local imports
from .ocr import ocr_file, ocr_image
from .rag import chunk_text
from .config import load_config


class ExtractionMethod(Enum):
    """Methods for text extraction"""
    NATIVE_TEXT = "native_text"      # Direct text extraction from document
    OCR_REQUIRED = "ocr_required"    # OCR needed for image-based content
    HYBRID = "hybrid"                # Mix of native text and OCR
    UNSUPPORTED = "unsupported"      # File type not supported


class ExtractionQuality(Enum):
    """Quality assessment of extracted text"""
    HIGH = "high"           # Clean, well-formatted text
    MEDIUM = "medium"       # Some formatting issues but readable
    LOW = "low"             # Poor quality, may need manual review
    FAILED = "failed"       # Extraction failed


@dataclass
class TextExtractionResult:
    """Result of intelligent text extraction"""
    success: bool
    method_used: ExtractionMethod
    quality: ExtractionQuality
    extracted_text: Optional[str] = None
    confidence_score: float = 0.0
    chunks: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    error_message: Optional[str] = None
    extraction_details: Optional[Dict] = None


class IntelligentDocumentProcessor:
    """
    Intelligent document processor that determines the best extraction method
    for documents, PDFs, and images based on content analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the intelligent document processor"""
        self.config = load_config(config_path) if config_path else load_config()
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already done
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def process_document(self, file_path: Union[str, Path], 
                        force_method: Optional[ExtractionMethod] = None,
                        quality_threshold: float = 0.5) -> TextExtractionResult:
        """
        Main entry point for intelligent document processing.
        Automatically determines the best extraction method based on content analysis.
        
        Args:
            file_path: Path to the document to process
            force_method: Force a specific extraction method (optional)
            quality_threshold: Minimum quality threshold for text extraction
            
        Returns:
            TextExtractionResult with extracted text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return TextExtractionResult(
                success=False,
                method_used=ExtractionMethod.UNSUPPORTED,
                quality=ExtractionQuality.FAILED,
                error_message=f"File not found: {file_path}"
            )
        
        self.logger.info(f"Processing document: {file_path}")
        
        # Determine file type and supported extraction methods
        file_type = self._detect_file_type(file_path)
        supported_methods = self._get_supported_methods(file_type)
        
        if not supported_methods:
            return TextExtractionResult(
                success=False,
                method_used=ExtractionMethod.UNSUPPORTED,
                quality=ExtractionQuality.FAILED,
                error_message=f"Unsupported file type: {file_type}"
            )
        
        # If forced method is specified, use it if supported
        if force_method and force_method in supported_methods:
            return self._extract_with_method(file_path, force_method)
        
        # Intelligent method selection
        best_method = self._determine_best_extraction_method(file_path, file_type)
        
        # Attempt extraction with the best method
        result = self._extract_with_method(file_path, best_method)
        
        # If extraction failed or quality is too low, try fallback methods
        if not result.success or result.confidence_score < quality_threshold:
            fallback_methods = [m for m in supported_methods if m != best_method]
            for fallback_method in fallback_methods:
                self.logger.info(f"Trying fallback method: {fallback_method.value}")
                fallback_result = self._extract_with_method(file_path, fallback_method)
                
                if (fallback_result.success and 
                    fallback_result.confidence_score > result.confidence_score):
                    result = fallback_result
                    break
        
        # Chunk the extracted text if successful
        if result.success and result.extracted_text:
            try:
                result.chunks = chunk_text(
                    result.extracted_text,
                    chunk_size=self.config.get('chunk_size', 1000),
                    overlap=self.config.get('chunk_overlap', 100),
                    config=self.config
                )
                self.logger.info(f"Text chunked into {len(result.chunks)} chunks")
            except Exception as e:
                self.logger.warning(f"Chunking failed: {e}")
                result.chunks = [result.extracted_text]
        
        self.logger.info(f"Document processing completed. Method: {result.method_used.value}, "
                        f"Quality: {result.quality.value}, Success: {result.success}")
        
        return result
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension and MIME type"""
        ext = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Map extensions to standardized types
        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx', 
            '.doc': 'doc',
            '.txt': 'txt',
            '.rtf': 'rtf',
            '.md': 'markdown',
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image',
            '.bmp': 'image', '.tiff': 'image', '.tif': 'image',
            '.gif': 'image', '.webp': 'image'
        }
        
        return type_mapping.get(ext, 'unknown')
    
    def _get_supported_methods(self, file_type: str) -> List[ExtractionMethod]:
        """Get supported extraction methods for a file type"""
        method_map = {
            'pdf': [ExtractionMethod.NATIVE_TEXT, ExtractionMethod.OCR_REQUIRED, ExtractionMethod.HYBRID],
            'docx': [ExtractionMethod.NATIVE_TEXT],
            'doc': [ExtractionMethod.NATIVE_TEXT],  # If supported
            'txt': [ExtractionMethod.NATIVE_TEXT],
            'rtf': [ExtractionMethod.NATIVE_TEXT],
            'markdown': [ExtractionMethod.NATIVE_TEXT],
            'image': [ExtractionMethod.OCR_REQUIRED]
        }
        
        return method_map.get(file_type, [])
    
    def _determine_best_extraction_method(self, file_path: Path, file_type: str) -> ExtractionMethod:
        """Intelligently determine the best extraction method based on content analysis"""
        
        if file_type == 'image':
            return ExtractionMethod.OCR_REQUIRED
        
        if file_type in ['txt', 'rtf', 'markdown']:
            return ExtractionMethod.NATIVE_TEXT
        
        if file_type == 'docx':
            return ExtractionMethod.NATIVE_TEXT
        
        if file_type == 'pdf':
            # For PDFs, analyze content to determine if text is extractible
            text_quality = self._analyze_pdf_text_quality(file_path)
            
            if text_quality['is_text_based'] and text_quality['text_ratio'] > 0.8:
                return ExtractionMethod.NATIVE_TEXT
            elif text_quality['is_text_based'] and text_quality['text_ratio'] > 0.3:
                return ExtractionMethod.HYBRID
            else:
                return ExtractionMethod.OCR_REQUIRED
        
        # Default fallback
        return ExtractionMethod.NATIVE_TEXT
    
    def _analyze_pdf_text_quality(self, file_path: Path) -> Dict:
        """Analyze PDF to determine if it contains extractible text or is image-based"""
        analysis = {
            'is_text_based': False,
            'text_ratio': 0.0,
            'has_images': False,
            'total_pages': 0,
            'pages_with_text': 0,
            'pages_with_images': 0,
            'confidence': 0.0
        }
        
        try:
            if fitz:  # PyMuPDF - more reliable for analysis
                doc = fitz.open(str(file_path))
                analysis['total_pages'] = len(doc)
                
                total_text_length = 0
                total_image_area = 0
                total_page_area = 0
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # Extract text
                    text = page.get_text()
                    if text and text.strip():
                        analysis['pages_with_text'] += 1
                        total_text_length += len(text)
                    
                    # Check for images
                    image_list = page.get_images()
                    if image_list:
                        analysis['has_images'] = True
                        analysis['pages_with_images'] += 1
                        
                        # Calculate image coverage
                        for img in image_list:
                            try:
                                rect = page.get_image_bbox(img)
                                if rect:
                                    total_image_area += rect.width * rect.height
                            except:
                                pass
                    
                    # Calculate page area
                    rect = page.rect
                    total_page_area += rect.width * rect.height
                
                doc.close()
                
                # Calculate ratios and confidence
                if analysis['total_pages'] > 0:
                    text_page_ratio = analysis['pages_with_text'] / analysis['total_pages']
                    image_coverage_ratio = total_image_area / total_page_area if total_page_area > 0 else 0
                    
                    analysis['text_ratio'] = text_page_ratio
                    analysis['is_text_based'] = text_page_ratio > 0.5 and total_text_length > 100
                    
                    # Confidence based on multiple factors
                    if analysis['is_text_based'] and text_page_ratio > 0.8:
                        analysis['confidence'] = 0.9
                    elif analysis['is_text_based'] and text_page_ratio > 0.5:
                        analysis['confidence'] = 0.7
                    elif image_coverage_ratio > 0.8:
                        analysis['confidence'] = 0.8  # High confidence it needs OCR
                    else:
                        analysis['confidence'] = 0.5  # Uncertain
                
            elif PdfReader:  # Fallback to PyPDF2
                reader = PdfReader(str(file_path))
                analysis['total_pages'] = len(reader.pages)
                
                total_text_length = 0
                
                for page in reader.pages:
                    text = page.extract_text() or ''
                    if text.strip():
                        analysis['pages_with_text'] += 1
                        total_text_length += len(text)
                
                if analysis['total_pages'] > 0:
                    text_ratio = analysis['pages_with_text'] / analysis['total_pages']
                    analysis['text_ratio'] = text_ratio
                    analysis['is_text_based'] = text_ratio > 0.5 and total_text_length > 100
                    analysis['confidence'] = 0.7 if analysis['is_text_based'] else 0.3
                
        except Exception as e:
            self.logger.warning(f"PDF analysis failed: {e}")
            analysis['confidence'] = 0.0
        
        return analysis
    
    def _extract_with_method(self, file_path: Path, method: ExtractionMethod) -> TextExtractionResult:
        """Extract text using the specified method"""
        
        try:
            if method == ExtractionMethod.NATIVE_TEXT:
                return self._extract_native_text(file_path)
            
            elif method == ExtractionMethod.OCR_REQUIRED:
                return self._extract_with_ocr(file_path)
            
            elif method == ExtractionMethod.HYBRID:
                return self._extract_hybrid(file_path)
            
            else:
                return TextExtractionResult(
                    success=False,
                    method_used=method,
                    quality=ExtractionQuality.FAILED,
                    error_message=f"Unsupported extraction method: {method.value}"
                )
                
        except Exception as e:
            self.logger.error(f"Extraction failed with method {method.value}: {e}")
            return TextExtractionResult(
                success=False,
                method_used=method,
                quality=ExtractionQuality.FAILED,
                error_message=str(e)
            )
    
    def _extract_native_text(self, file_path: Path) -> TextExtractionResult:
        """Extract text using native document parsing"""
        ext = file_path.suffix.lower()
        text = ""
        
        try:
            if ext == '.pdf':
                if PdfReader:
                    reader = PdfReader(str(file_path))
                    text = "\n".join(page.extract_text() or '' for page in reader.pages)
                else:
                    raise Exception("PyPDF2 not available for PDF processing")
            
            elif ext == '.docx':
                if docx:
                    doc = docx.Document(str(file_path))
                    text = "\n".join(para.text for para in doc.paragraphs)
                else:
                    raise Exception("python-docx not available for DOCX processing")
            
            elif ext in ['.txt', '.md', '.rtf']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            else:
                raise Exception(f"Native text extraction not supported for {ext}")
            
            # Assess quality
            quality, confidence = self._assess_text_quality(text)
            
            if not text.strip():
                return TextExtractionResult(
                    success=False,
                    method_used=ExtractionMethod.NATIVE_TEXT,
                    quality=ExtractionQuality.FAILED,
                    error_message="No text extracted from document"
                )
            
            return TextExtractionResult(
                success=True,
                method_used=ExtractionMethod.NATIVE_TEXT,
                quality=quality,
                extracted_text=text,
                confidence_score=confidence,
                metadata={
                    'text_length': len(text),
                    'extraction_method': 'native_parsing'
                }
            )
            
        except Exception as e:
            return TextExtractionResult(
                success=False,
                method_used=ExtractionMethod.NATIVE_TEXT,
                quality=ExtractionQuality.FAILED,
                error_message=str(e)
            )
    
    def _extract_with_ocr(self, file_path: Path) -> TextExtractionResult:
        """Extract text using OCR"""
        try:
            # Extract tesseract configuration string from config
            config_string = self.config.get('tesseract_config', '--psm 6')
            text = ocr_file(str(file_path), config_string=config_string)
            
            if not text or not text.strip():
                return TextExtractionResult(
                    success=False,
                    method_used=ExtractionMethod.OCR_REQUIRED,
                    quality=ExtractionQuality.FAILED,
                    error_message="OCR failed to extract text"
                )
            
            # Assess quality
            quality, confidence = self._assess_text_quality(text, is_ocr=True)
            
            return TextExtractionResult(
                success=True,
                method_used=ExtractionMethod.OCR_REQUIRED,
                quality=quality,
                extracted_text=text,
                confidence_score=confidence,
                metadata={
                    'text_length': len(text),
                    'extraction_method': 'ocr',
                    'tesseract_config': config_string
                }
            )
            
        except Exception as e:
            return TextExtractionResult(
                success=False,
                method_used=ExtractionMethod.OCR_REQUIRED,
                quality=ExtractionQuality.FAILED,
                error_message=str(e)
            )
    
    def _extract_hybrid(self, file_path: Path) -> TextExtractionResult:
        """Extract text using hybrid approach (native + OCR for missing parts)"""
        try:
            # First try native extraction
            native_result = self._extract_native_text(file_path)
            
            if native_result.success and native_result.confidence_score > 0.7:
                # Native extraction is good enough
                native_result.method_used = ExtractionMethod.HYBRID
                return native_result
            
            # Native extraction is insufficient, try OCR
            ocr_result = self._extract_with_ocr(file_path)
            
            if not ocr_result.success:
                # OCR failed, return native result if available
                if native_result.success:
                    native_result.method_used = ExtractionMethod.HYBRID
                    return native_result
                else:
                    return TextExtractionResult(
                        success=False,
                        method_used=ExtractionMethod.HYBRID,
                        quality=ExtractionQuality.FAILED,
                        error_message="Both native and OCR extraction failed"
                    )
            
            # Combine results or choose the better one
            if native_result.success:
                # Compare quality and choose the better result
                if native_result.confidence_score >= ocr_result.confidence_score:
                    best_text = native_result.extracted_text
                    best_confidence = native_result.confidence_score
                    best_quality = native_result.quality
                else:
                    best_text = ocr_result.extracted_text
                    best_confidence = ocr_result.confidence_score
                    best_quality = ocr_result.quality
                
                # Could also implement text combination logic here
                
            else:
                best_text = ocr_result.extracted_text
                best_confidence = ocr_result.confidence_score
                best_quality = ocr_result.quality
            
            return TextExtractionResult(
                success=True,
                method_used=ExtractionMethod.HYBRID,
                quality=best_quality,
                extracted_text=best_text,
                confidence_score=best_confidence,
                metadata={
                    'text_length': len(best_text),
                    'extraction_method': 'hybrid',
                    'native_available': native_result.success,
                    'ocr_available': ocr_result.success
                }
            )
            
        except Exception as e:
            return TextExtractionResult(
                success=False,
                method_used=ExtractionMethod.HYBRID,
                quality=ExtractionQuality.FAILED,
                error_message=str(e)
            )
    
    def _assess_text_quality(self, text: str, is_ocr: bool = False) -> Tuple[ExtractionQuality, float]:
        """Assess the quality of extracted text"""
        if not text or not text.strip():
            return ExtractionQuality.FAILED, 0.0
        
        # Basic quality metrics
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        # Character diversity (higher is better)
        char_diversity = len(set(text.lower())) / len(text) if len(text) > 0 else 0
        
        # Ratio of alphabetic characters (higher is usually better)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if len(text) > 0 else 0
        
        # OCR-specific quality assessment
        if is_ocr:
            # OCR often produces artifacts, so we're more lenient
            if word_count < 5:
                return ExtractionQuality.FAILED, 0.1
            elif alpha_ratio < 0.3:
                return ExtractionQuality.LOW, 0.3
            elif alpha_ratio < 0.6:
                return ExtractionQuality.MEDIUM, 0.6
            else:
                return ExtractionQuality.HIGH, 0.8
        
        else:
            # Native text should be higher quality
            if word_count < 10:
                return ExtractionQuality.LOW, 0.3
            elif char_diversity < 0.05 or alpha_ratio < 0.5:
                return ExtractionQuality.MEDIUM, 0.5
            elif alpha_ratio > 0.7 and char_diversity > 0.1:
                return ExtractionQuality.HIGH, 0.9
            else:
                return ExtractionQuality.MEDIUM, 0.7
    
    def batch_process(self, file_paths: List[Union[str, Path]], **kwargs) -> List[TextExtractionResult]:
        """Process multiple documents in batch"""
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing failed for {file_path}: {e}")
                results.append(TextExtractionResult(
                    success=False,
                    method_used=ExtractionMethod.UNSUPPORTED,
                    quality=ExtractionQuality.FAILED,
                    error_message=str(e)
                ))
        
        return results


def create_intelligent_processor(config_path: Optional[str] = None) -> IntelligentDocumentProcessor:
    """Factory function to create an intelligent document processor"""
    return IntelligentDocumentProcessor(config_path)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python intelligent_document_processor.py <file_path> [force_method]")
        print("Force methods: native_text, ocr_required, hybrid")
        sys.exit(1)
    
    file_path = sys.argv[1]
    force_method = None
    
    if len(sys.argv) > 2:
        method_map = {
            'native_text': ExtractionMethod.NATIVE_TEXT,
            'ocr_required': ExtractionMethod.OCR_REQUIRED,
            'hybrid': ExtractionMethod.HYBRID
        }
        force_method = method_map.get(sys.argv[2])
    
    processor = create_intelligent_processor()
    result = processor.process_document(file_path, force_method=force_method)
    
    print(f"Processing result: {result.success}")
    print(f"Method used: {result.method_used.value}")
    print(f"Quality: {result.quality.value}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Text length: {len(result.extracted_text) if result.extracted_text else 0}")
    print(f"Chunks: {len(result.chunks) if result.chunks else 0}")
    
    if not result.success:
        print(f"Error: {result.error_message}")
    
    if result.metadata:
        print(f"Metadata: {result.metadata}")
