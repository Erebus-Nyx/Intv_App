#!/usr/bin/env python3
"""
INTV Unified Document and Image Processor

This module provides a single entrypoint for processing both documents and images,
with intelligent format detection and appropriate processing method selection.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# Import dependency manager
from .dependency_manager import get_dependency_manager, check_feature_dependencies

class ProcessingMethod(Enum):
    """Available processing methods"""
    AUTO = "auto"
    OCR = "ocr"
    TEXT_EXTRACTION = "text_extraction"
    HYBRID = "hybrid"

class DocumentImageProcessor:
    """Unified processor for documents and images"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.dependency_manager = get_dependency_manager()
        
        # Check available processing capabilities
        self.has_ocr = self.dependency_manager.has_group('ocr')
        self.has_core = self.dependency_manager.has_group('core')
        self.has_ml = self.dependency_manager.has_group('ml')
    
    def detect_file_type(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Detect file type and recommend processing method"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'type': 'unknown',
                'is_document': False,
                'is_image': False,
                'recommended_method': ProcessingMethod.AUTO,
                'error': 'File not found'
            }
        
        suffix = file_path.suffix.lower()
        
        # Document types
        document_types = {
            '.pdf': {'type': 'pdf', 'is_document': True, 'is_image': False},
            '.docx': {'type': 'docx', 'is_document': True, 'is_image': False},
            '.doc': {'type': 'doc', 'is_document': True, 'is_image': False},
            '.txt': {'type': 'text', 'is_document': True, 'is_image': False},
            '.rtf': {'type': 'rtf', 'is_document': True, 'is_image': False},
            '.odt': {'type': 'odt', 'is_document': True, 'is_image': False},
        }
        
        # Image types
        image_types = {
            '.jpg': {'type': 'jpeg', 'is_document': False, 'is_image': True},
            '.jpeg': {'type': 'jpeg', 'is_document': False, 'is_image': True},
            '.png': {'type': 'png', 'is_document': False, 'is_image': True},
            '.tiff': {'type': 'tiff', 'is_document': False, 'is_image': True},
            '.tif': {'type': 'tiff', 'is_document': False, 'is_image': True},
            '.bmp': {'type': 'bmp', 'is_document': False, 'is_image': True},
            '.gif': {'type': 'gif', 'is_document': False, 'is_image': True},
            '.webp': {'type': 'webp', 'is_document': False, 'is_image': True},
        }
        
        if suffix in document_types:
            result = document_types[suffix].copy()
            result['recommended_method'] = ProcessingMethod.TEXT_EXTRACTION
            if suffix == '.pdf':
                result['recommended_method'] = ProcessingMethod.HYBRID  # PDFs might need OCR
        elif suffix in image_types:
            result = image_types[suffix].copy()
            result['recommended_method'] = ProcessingMethod.OCR
        else:
            result = {
                'type': 'unknown',
                'is_document': False,
                'is_image': False,
                'recommended_method': ProcessingMethod.AUTO
            }
        
        result['file_size'] = file_path.stat().st_size
        return result
    
    def get_processing_plan(self, file_path: Union[str, Path], method: ProcessingMethod = ProcessingMethod.AUTO) -> Dict[str, Any]:
        """Create a processing plan based on file type and available dependencies"""
        file_info = self.detect_file_type(file_path)
        
        if 'error' in file_info:
            return {'error': file_info['error'], 'can_process': False}
        
        plan = {
            'file_info': file_info,
            'method': method if method != ProcessingMethod.AUTO else file_info['recommended_method'],
            'can_process': False,
            'missing_dependencies': [],
            'processing_steps': [],
            'fallback_available': False
        }
        
        # Determine required dependencies based on processing method
        if plan['method'] == ProcessingMethod.OCR:
            if not self.has_ocr:
                plan['missing_dependencies'].extend(self.dependency_manager.get_missing_for_feature('ocr'))
            else:
                plan['can_process'] = True
                plan['processing_steps'] = ['ocr_extraction', 'text_cleanup']
        
        elif plan['method'] == ProcessingMethod.TEXT_EXTRACTION:
            if not self.has_core:
                plan['missing_dependencies'].extend(self.dependency_manager.get_missing_for_feature('document'))
            else:
                plan['can_process'] = True
                if file_info['type'] == 'pdf':
                    plan['processing_steps'] = ['pdf_text_extraction']
                elif file_info['type'] == 'docx':
                    plan['processing_steps'] = ['docx_text_extraction']
                else:
                    plan['processing_steps'] = ['basic_text_read']
        
        elif plan['method'] == ProcessingMethod.HYBRID:
            # Try text extraction first, fallback to OCR
            if self.has_core:
                plan['can_process'] = True
                plan['processing_steps'] = ['pdf_text_extraction', 'ocr_fallback']
                if not self.has_ocr:
                    plan['missing_dependencies'].extend(['pytesseract', 'Pillow'])
                    plan['processing_steps'] = ['pdf_text_extraction']  # No OCR fallback
            else:
                plan['missing_dependencies'].extend(self.dependency_manager.get_missing_for_feature('document'))
        
        # Check for basic fallback
        if not plan['can_process'] and file_info.get('type') == 'text':
            plan['fallback_available'] = True
            plan['can_process'] = True
            plan['processing_steps'] = ['basic_text_read']
        
        return plan
    
    def process_file(self, file_path: Union[str, Path], 
                    method: ProcessingMethod = ProcessingMethod.AUTO,
                    **kwargs) -> Dict[str, Any]:
        """Process a file using the unified interface"""
        
        file_path = Path(file_path)
        plan = self.get_processing_plan(file_path, method)
        
        if not plan['can_process']:
            return {
                'success': False,
                'error': 'Cannot process file - missing dependencies',
                'missing_dependencies': plan['missing_dependencies'],
                'install_guide': self._get_install_guide(plan['missing_dependencies'])
            }
        
        result = {
            'success': False,
            'extracted_text': '',
            'metadata': {
                'file_path': str(file_path),
                'file_info': plan['file_info'],
                'processing_method': plan['method'].value,
                'processing_steps': plan['processing_steps']
            }
        }
        
        try:
            for step in plan['processing_steps']:
                step_result = self._execute_processing_step(file_path, step, **kwargs)
                
                if step_result.get('success') and step_result.get('extracted_text'):
                    result['extracted_text'] = step_result['extracted_text']
                    result['metadata'].update(step_result.get('metadata', {}))
                    result['success'] = True
                    result['metadata']['successful_step'] = step
                    break  # Success, don't try other steps
                elif step == 'ocr_fallback' and step_result.get('success'):
                    # OCR fallback succeeded
                    result['extracted_text'] = step_result['extracted_text']
                    result['metadata'].update(step_result.get('metadata', {}))
                    result['success'] = True
                    result['metadata']['successful_step'] = step
                    break
            
            if not result['success'] and plan['fallback_available']:
                # Try basic text read as last resort
                fallback_result = self._execute_processing_step(file_path, 'basic_text_read', **kwargs)
                if fallback_result.get('success'):
                    result.update(fallback_result)
                    result['metadata']['successful_step'] = 'basic_text_read_fallback'
        
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Processing failed for {file_path}: {e}")
        
        return result
    
    def _execute_processing_step(self, file_path: Path, step: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific processing step"""
        
        if step == 'basic_text_read':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                return {
                    'success': True,
                    'extracted_text': text,
                    'metadata': {'extraction_method': 'basic_text_read'}
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        elif step == 'pdf_text_extraction':
            return self._extract_pdf_text(file_path)
        
        elif step == 'docx_text_extraction':
            return self._extract_docx_text(file_path)
        
        elif step == 'ocr_extraction':
            return self._extract_ocr_text(file_path)
        
        elif step == 'ocr_fallback':
            # Try OCR on PDF pages that might be scanned
            return self._extract_pdf_ocr(file_path)
        
        else:
            return {'success': False, 'error': f'Unknown processing step: {step}'}
    
    def _extract_pdf_text(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using PyPDF2"""
        if not self.has_core:
            return {'success': False, 'error': 'PyPDF2 not available'}
        
        try:
            from PyPDF2 import PdfReader
            
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                
                extracted_text = '\n\n'.join(text_parts)
                
                return {
                    'success': True,
                    'extracted_text': extracted_text,
                    'metadata': {
                        'extraction_method': 'pdf_text_extraction',
                        'page_count': len(reader.pages),
                        'pages_with_text': len(text_parts)
                    }
                }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_docx_text(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX using python-docx"""
        if not self.has_core:
            return {'success': False, 'error': 'python-docx not available'}
        
        try:
            import docx
            
            doc = docx.Document(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            extracted_text = '\n\n'.join(paragraphs)
            
            return {
                'success': True,
                'extracted_text': extracted_text,
                'metadata': {
                    'extraction_method': 'docx_text_extraction',
                    'paragraph_count': len(paragraphs)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_ocr_text(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        if not self.has_ocr:
            return {'success': False, 'error': 'OCR dependencies not available'}
        
        try:
            from PIL import Image
            import pytesseract
            
            with Image.open(file_path) as img:
                text = pytesseract.image_to_string(img)
            
            return {
                'success': True,
                'extracted_text': text,
                'metadata': {
                    'extraction_method': 'ocr_extraction',
                    'image_size': f"{img.size[0]}x{img.size[1]}" if 'img' in locals() else 'unknown'
                }
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_pdf_ocr(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using OCR (for scanned PDFs)"""
        if not self.has_ocr:
            return {'success': False, 'error': 'OCR dependencies not available'}
        
        try:
            # This would require pdf2image, which should be in OCR dependencies
            from pdf2image import convert_from_path
            from PIL import Image
            import pytesseract
            
            pages = convert_from_path(file_path)
            text_parts = []
            
            for page_num, page in enumerate(pages):
                page_text = pytesseract.image_to_string(page)
                if page_text.strip():
                    text_parts.append(page_text)
            
            extracted_text = '\n\n'.join(text_parts)
            
            return {
                'success': True,
                'extracted_text': extracted_text,
                'metadata': {
                    'extraction_method': 'pdf_ocr_extraction',
                    'page_count': len(pages),
                    'pages_with_text': len(text_parts)
                }
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_install_guide(self, missing_packages: List[str]) -> str:
        """Generate installation guide for missing packages"""
        if not missing_packages:
            return ""
        
        return f"""
Missing dependencies: {', '.join(missing_packages)}

Install with pipx:
  pipx inject intv {' '.join(missing_packages)}

Or see full installation guide:
  python -c "from intv.dependency_manager import print_installation_guide; print_installation_guide()"
"""

# Global instance for easy access
_unified_processor = None

def get_unified_processor(config: Dict[str, Any] = None) -> DocumentImageProcessor:
    """Get global unified processor instance"""
    global _unified_processor
    if _unified_processor is None:
        _unified_processor = DocumentImageProcessor(config)
    return _unified_processor

def process_document_or_image(file_path: Union[str, Path], 
                             method: ProcessingMethod = ProcessingMethod.AUTO,
                             **kwargs) -> Dict[str, Any]:
    """Convenience function for processing files"""
    processor = get_unified_processor()
    return processor.process_file(file_path, method, **kwargs)
