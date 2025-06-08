# filepath: /home/nyx/intv/intv/pipeline_orchestrator.py
"""
Pipeline Orchestrator for INTV Application
Coordinates processing of different input types through appropriate pipelines
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Core imports
from .config import load_config

# Optional imports with fallbacks
try:
    from .utils import is_valid_filetype, validate_file
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    def is_valid_filetype(*args, **kwargs):
        return True
    def validate_file(*args, **kwargs):
        pass

try:
    from .intelligent_document_processor import create_intelligent_processor, ExtractionMethod
    HAS_DOC_PROCESSOR = True
except ImportError:
    HAS_DOC_PROCESSOR = False
    class ExtractionMethod:
        AUTO = "auto"
        OCR = "ocr"

try:
    from .rag import enhanced_chunk_document, enhanced_query_documents, enhanced_rag_pipeline
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    def enhanced_chunk_document(*args, **kwargs):
        return {'success': False, 'error': 'RAG module not available'}
    def enhanced_query_documents(*args, **kwargs):
        return {'success': False, 'error': 'RAG module not available'}
    def enhanced_rag_pipeline(*args, **kwargs):
        return {'success': False, 'error': 'RAG module not available'}

# Audio processing
try:
    from .audio_transcribe import transcribe_audio_fastwhisper, stream_microphone_transcription_interactive
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Module processing
try:
    from .modules.dynamic_module import dynamic_module_output
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False

# Import dependency manager
from .dependency_manager import get_dependency_manager, check_feature_dependencies

# Import unified processor
from .unified_processor import get_unified_processor, ProcessingMethod


class InputType(Enum):
    """Supported input types for the pipeline"""
    DOCUMENT = "document"       # PDF, DOCX, TXT files
    IMAGE = "image"            # JPG, PNG, etc. (with OCR)
    AUDIO = "audio"            # WAV, MP3, etc.
    MICROPHONE = "microphone"  # Live microphone input
    UNKNOWN = "unknown"


@dataclass
class ProcessingResult:
    """Result of pipeline processing"""
    success: bool
    input_type: InputType
    extracted_text: Optional[str] = None
    transcript: Optional[str] = None
    chunks: Optional[List[str]] = None
    segments: Optional[List[Dict]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    rag_result: Optional[Any] = None
    llm_output: Optional[Any] = None


class PipelineOrchestrator:
    """
    Main orchestrator for processing different input types.
    Routes inputs through appropriate processing pipelines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline orchestrator"""
        self.config_path = config_path
        self.config = None
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            self.config = load_config(config_path)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
            self.config = {}
        
        # Initialize dependency manager
        self.dependency_manager = get_dependency_manager()
        
        # Initialize unified processor
        self.unified_processor = get_unified_processor(self.config)
        
        # Initialize processors (legacy)
        self.doc_processor = None
        self._init_processors()
    
    def _init_processors(self):
        """Initialize processing components"""
        try:
            if HAS_DOC_PROCESSOR:
                self.doc_processor = create_intelligent_processor(self.config_path)
            else:
                self.doc_processor = None
        except Exception as e:
            self.logger.warning(f"Could not initialize document processor: {e}")
            self.doc_processor = None
    
    def detect_input_type(self, input_path: Union[str, Path]) -> InputType:
        """Detect the type of input based on file extension"""
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        
        if not input_path.exists():
            return InputType.UNKNOWN
        
        ext = input_path.suffix.lower()
        
        # Document types
        document_exts = {'.pdf', '.docx', '.doc', '.txt', '.rtf', '.md', '.toml', '.yml', '.yaml', '.json'}
        if ext in document_exts:
            return InputType.DOCUMENT
        
        # Image types
        image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
        if ext in image_exts:
            return InputType.IMAGE
        
        # Audio types
        audio_exts = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}
        if ext in audio_exts:
            return InputType.AUDIO
        
        return InputType.UNKNOWN
    
    def process(self, 
                input_path: Union[str, Path],
                module_key: Optional[str] = None,
                query: Optional[str] = None,
                apply_llm: bool = True) -> ProcessingResult:
        """
        Main processing method that routes input through appropriate pipeline
        
        Args:
            input_path: Path to input file
            module_key: Optional module key for LLM processing
            query: Optional query for RAG processing
            apply_llm: Whether to apply LLM/module processing
            
        Returns:
            ProcessingResult with all processing outputs
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        
        # Detect input type
        input_type = self.detect_input_type(input_path)
        
        # Route to appropriate processor
        if input_type == InputType.DOCUMENT or input_type == InputType.IMAGE:
            return self.process_document_or_image(input_path, module_key, query, apply_llm)
        elif input_type == InputType.AUDIO:
            return self.process_audio_file(input_path, module_key, query, apply_llm)
        else:
            return ProcessingResult(
                success=False,
                input_type=input_type,
                error_message=f"Unsupported input type: {input_type.value}"
            )
    
    def process_document_or_image(self,
                                 file_path: Path,
                                 module_key: Optional[str] = None,
                                 query: Optional[str] = None,
                                 apply_llm: bool = True) -> ProcessingResult:
        """
        Unified processing method for both documents and images
        Uses the unified processor to handle both types intelligently
        """
        try:
            # Validate file
            if HAS_UTILS:
                validate_file(str(file_path))
            
            input_type = self.detect_input_type(file_path)
            
            # For simple text files, try processing without dependencies first
            simple_text_exts = {'.txt', '.md', '.toml', '.yml', '.yaml', '.json'}
            if file_path.suffix.lower() in simple_text_exts:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        extracted_text = f.read()
                    
                    if extracted_text:
                        # Create result with basic text extraction
                        result = ProcessingResult(
                            success=True,
                            input_type=input_type,
                            extracted_text=extracted_text,
                            metadata={'method': 'simple_text_read', 'file_type': file_path.suffix.lower()}
                        )
                        
                        # Try to chunk the text if RAG is available
                        try:
                            if HAS_RAG:
                                from .rag import chunk_text
                                chunks = chunk_text(extracted_text)
                                result.chunks = chunks
                        except Exception as e:
                            self.logger.warning(f"Text chunking failed: {e}")
                        
                        # Apply RAG and LLM processing
                        result = self._apply_rag_and_llm(result, query, module_key, apply_llm)
                        return result
                        
                except Exception as e:
                    self.logger.warning(f"Simple text extraction failed: {e}")
            
            # Check dependencies
            input_type = self.detect_input_type(file_path)
            required_features = ['core']
            if input_type == InputType.IMAGE:
                required_features.append('ocr')
            
            # Check if dependencies are available
            missing_features = []
            for feature in required_features:
                dm = get_dependency_manager()
                missing = dm.get_missing_for_feature(feature)
                available = len(missing) == 0
                if not available:
                    missing_features.append(feature)
            
            if missing_features:
                return ProcessingResult(
                    success=False,
                    input_type=input_type,
                    error_message=f"Missing dependencies for {missing_features}. Install with: pipx inject intv PyPDF2 python-docx"
                )
            
            # Process using unified processor
            processing_result = self.unified_processor.process_file(str(file_path))
            
            if not processing_result['success']:
                return ProcessingResult(
                    success=False,
                    input_type=input_type,
                    error_message=processing_result.get('error', 'Processing failed'),
                    metadata=processing_result.get('metadata', {})
                )
            
            extracted_text = processing_result.get('text', '')
            if not extracted_text:
                return ProcessingResult(
                    success=False,
                    input_type=input_type,
                    error_message="No text could be extracted from file"
                )
            
            # Create result
            result = ProcessingResult(
                success=True,
                input_type=input_type,
                extracted_text=extracted_text,
                metadata=processing_result.get('metadata', {})
            )
            
            # Chunk the document/text for RAG processing
            try:
                if HAS_RAG:
                    if input_type == InputType.DOCUMENT:
                        # For documents, use enhanced chunking
                        chunk_result = enhanced_chunk_document(str(file_path), self.config)
                        if chunk_result and chunk_result.get('success'):
                            result.chunks = chunk_result.get('chunks', [])
                            if 'chunk_metadata' not in result.metadata:
                                result.metadata['chunk_metadata'] = {}
                            result.metadata['chunk_metadata'].update(chunk_result.get('metadata', {}))
                    else:
                        # For images, chunk the extracted text
                        from .rag import chunk_text
                        chunks = chunk_text(extracted_text)
                        result.chunks = chunks
            except Exception as e:
                self.logger.warning(f"Text chunking failed: {e}")
            
            # Apply RAG query if provided
            if query and result.chunks and HAS_RAG:
                try:
                    rag_result = enhanced_query_documents(
                        query, 
                        result.chunks, 
                        self.config
                    )
                    result.rag_result = rag_result
                except Exception as e:
                    self.logger.warning(f"RAG processing failed: {e}")
            
            # Apply module processing if requested
            if module_key and HAS_MODULES and apply_llm:
                try:
                    llm_result = dynamic_module_output(
                        module_key=module_key,
                        provided_data=extracted_text
                    )
                    result.llm_output = llm_result
                except Exception as e:
                    self.logger.warning(f"Module processing failed: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Document/Image processing failed: {e}")
            return ProcessingResult(
                success=False,
                input_type=self.detect_input_type(file_path),
                error_message=str(e)
            )
    
    def process_document(self,
                        file_path: Path,
                        module_key: Optional[str] = None,
                        query: Optional[str] = None,
                        apply_llm: bool = True) -> ProcessingResult:
        """
        Process document files (PDF, DOCX, TXT, etc.)
        
        DEPRECATED: Use process_document_or_image() for unified processing
        This method is maintained for backward compatibility
        """
        self.logger.warning("process_document() is deprecated. Use process_document_or_image() instead.")
        return self.process_document_or_image(file_path, module_key, query, apply_llm)
    
    def process_image(self,
                     file_path: Path,
                     module_key: Optional[str] = None,
                     query: Optional[str] = None,
                     apply_llm: bool = True) -> ProcessingResult:
        """
        Process image files with OCR
        
        DEPRECATED: Use process_document_or_image() for unified processing
        This method is maintained for backward compatibility
        """
        self.logger.warning("process_image() is deprecated. Use process_document_or_image() instead.")
        return self.process_document_or_image(file_path, module_key, query, apply_llm)
    
    def process_audio_file(self, 
                          file_path: Path,
                          module_key: Optional[str] = None,
                          query: Optional[str] = None,
                          apply_llm: bool = True) -> ProcessingResult:
        """Process audio files using complete pipeline: Audio → Transcription → VAD → Diarization → RAG → LLM"""
        try:
            if not HAS_AUDIO:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.AUDIO,
                    error_message="Audio processing dependencies not available"
                )
            
            # Validate file
            if HAS_UTILS:
                validate_file(str(file_path))
            
            # Step 1: Transcribe audio (with VAD integration already handled in transcribe_audio_fastwhisper)
            segments = transcribe_audio_fastwhisper(
                str(file_path),
                return_segments=True,
                config=self.config
            )
            
            if not segments:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.AUDIO,
                    error_message="Transcription failed or returned no segments"
                )
            
            # Extract transcript text
            transcript = " ".join([seg.get('text', '') for seg in segments])
            
            # Step 2: Apply Enhanced Diarization (speaker separation)
            diarization_results = []
            if self.config.get('enable_diarization', True):
                try:
                    from .audio_diarization import diarize_audio
                    diarization_results = diarize_audio(
                        str(file_path),
                        num_speakers=self.config.get('num_speakers'),
                        min_speakers=self.config.get('min_speakers', 1),
                        max_speakers=self.config.get('max_speakers', 10),
                        config=self.config  # Pass config for enhanced pyannote processing
                    )
                    
                    # Merge diarization with transcription segments
                    for i, seg in enumerate(segments):
                        seg['speaker'] = None
                        seg_start = seg.get('start', 0)
                        seg_end = seg.get('end', seg_start)
                        
                        # Find matching speaker from diarization
                        for diar_seg in diarization_results:
                            diar_start = diar_seg.get('start_time', 0)
                            diar_end = diar_seg.get('end_time', diar_start)
                            
                            # Check for overlap
                            if (seg_start >= diar_start and seg_start < diar_end) or \
                               (seg_end > diar_start and seg_end <= diar_end) or \
                               (seg_start <= diar_start and seg_end >= diar_end):
                                seg['speaker'] = diar_seg.get('speaker_id')
                                break
                    
                except Exception as e:
                    self.logger.warning(f"Diarization failed: {e}")
            
            # Step 3: Chunk the transcript for RAG processing
            from .rag import chunk_text
            chunks = chunk_text(transcript)
            
            # Create result with enhanced metadata
            result = ProcessingResult(
                success=True,
                input_type=InputType.AUDIO,
                transcript=transcript,
                segments=segments,
                chunks=chunks,
                metadata={
                    'transcription_segments': len(segments),
                    'diarization_enabled': self.config.get('enable_diarization', True),
                    'diarization_speakers': len(set(seg.get('speaker') for seg in segments if seg.get('speaker'))),
                    'vad_enabled': self.config.get('enable_vad', True),
                    'audio_processing_pipeline': 'Audio → Transcription → VAD → Diarization → RAG → LLM'
                }
            )
            
            # Store diarization results if available
            if diarization_results:
                result.metadata['diarization_results'] = diarization_results
            
            # Step 4: Apply RAG query if provided
            if query and result.chunks and HAS_RAG:
                try:
                    rag_result = enhanced_query_documents(
                        query, 
                        result.chunks, 
                        self.config
                    )
                    result.rag_result = rag_result
                    self.logger.info(f"RAG processing completed: {len(result.chunks)} chunks processed")
                except Exception as e:
                    self.logger.warning(f"RAG processing failed: {e}")
            
            # Step 5: Apply LLM module processing if requested
            if module_key and HAS_MODULES and apply_llm:
                try:
                    # Use the new enhanced dynamic module processor
                    from .modules.enhanced_dynamic_module import enhanced_dynamic_module_output
                    
                    llm_result = enhanced_dynamic_module_output(
                        text_content=transcript,
                        module_key=module_key,
                        metadata={
                            "input_type": "audio",
                            "segments": len(segments) if segments else 0,
                            "speakers": result.metadata.get('diarization_speakers', 0),
                            "processing_pipeline": "complete_audio_pipeline"
                        }
                    )
                    result.llm_output = llm_result
                    self.logger.info(f"Enhanced LLM processing completed for module: {module_key}")
                except Exception as e:
                    # Fallback to legacy processor if enhanced one fails
                    self.logger.warning(f"Enhanced module processing failed, trying legacy: {e}")
                    try:
                        llm_result = dynamic_module_output(
                            module_key=module_key,
                            provided_data=transcript
                        )
                        result.llm_output = llm_result
                        self.logger.info(f"Legacy LLM processing completed for module: {module_key}")
                    except Exception as e2:
                        self.logger.warning(f"Module processing failed completely: {e2}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return ProcessingResult(
                success=False,
                input_type=InputType.AUDIO,
                error_message=str(e)
            )
    
    def process_microphone(self,
                          module_key: Optional[str] = None,
                          query: Optional[str] = None,
                          apply_llm: bool = True) -> ProcessingResult:
        """Process live microphone input using complete pipeline: Audio → Transcription → VAD → Diarization → RAG → LLM"""
        try:
            if not HAS_AUDIO:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.MICROPHONE,
                    error_message="Audio processing dependencies not available"
                )
            
            # Start microphone transcription (VAD is integrated in transcription)
            segments = stream_microphone_transcription_interactive(config=self.config)
            
            if not segments:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.MICROPHONE,
                    error_message="Microphone transcription failed or returned no segments"
                )
            
            # Extract transcript text
            transcript = " ".join([seg.get('text', '') for seg in segments])
            
            # Note: Diarization for microphone input is typically not needed as it's single speaker
            # But we can still apply it if requested
            diarization_results = []
            if self.config.get('enable_diarization', False):  # Default to False for microphone
                try:
                    # For microphone, we'd need to save the audio first to run diarization
                    # This is a placeholder - real implementation would need audio file path
                    self.logger.info("Diarization skipped for microphone input (real-time processing)")
                except Exception as e:
                    self.logger.warning(f"Microphone diarization failed: {e}")
            
            # Chunk the transcript
            from .rag import chunk_text
            chunks = chunk_text(transcript)
            
            # Create result with enhanced metadata
            result = ProcessingResult(
                success=True,
                input_type=InputType.MICROPHONE,
                transcript=transcript,
                segments=segments,
                chunks=chunks,
                metadata={
                    'transcription_segments': len(segments),
                    'diarization_enabled': False,  # Typically disabled for microphone
                    'vad_enabled': self.config.get('enable_vad', True),
                    'audio_processing_pipeline': 'Microphone → Transcription → VAD → RAG → LLM'
                }
            )
            
            # Apply RAG query if provided
            if query and result.chunks and HAS_RAG:
                try:
                    rag_result = enhanced_query_documents(
                        query, 
                        result.chunks, 
                        self.config
                    )
                    result.rag_result = rag_result
                    self.logger.info(f"RAG processing completed: {len(result.chunks)} chunks processed")
                except Exception as e:
                    self.logger.warning(f"RAG processing failed: {e}")
            
            # Apply module processing if requested
            if module_key and HAS_MODULES and apply_llm:
                try:
                    # Use the new enhanced dynamic module processor
                    from .modules.enhanced_dynamic_module import enhanced_dynamic_module_output
                    
                    llm_result = enhanced_dynamic_module_output(
                        text_content=transcript,
                        module_key=module_key,
                        metadata={
                            "input_type": "microphone",
                            "segments": len(segments) if segments else 0,
                            "processing_pipeline": "complete_microphone_pipeline"
                        }
                    )
                    result.llm_output = llm_result
                    self.logger.info(f"Enhanced LLM processing completed for module: {module_key}")
                except Exception as e:
                    # Fallback to legacy processor if enhanced one fails
                    self.logger.warning(f"Enhanced module processing failed, trying legacy: {e}")
                    try:
                        llm_result = dynamic_module_output(
                            module_key=module_key,
                            provided_data=transcript
                        )
                        result.llm_output = llm_result
                        self.logger.info(f"Legacy LLM processing completed for module: {module_key}")
                    except Exception as e2:
                        self.logger.warning(f"Module processing failed completely: {e2}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Microphone processing failed: {e}")
            return ProcessingResult(
                success=False,
                input_type=InputType.MICROPHONE,
                error_message=str(e)
            )
    
    def batch_process(self, 
                     input_paths: List[Union[str, Path]],
                     module_key: Optional[str] = None,
                     query: Optional[str] = None,
                     apply_llm: bool = True) -> List[ProcessingResult]:
        """Process multiple inputs in batch"""
        results = []
        
        for input_path in input_paths:
            try:
                result = self.process(input_path, module_key, query, apply_llm)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch processing failed for {input_path}: {e}")
                results.append(ProcessingResult(
                    success=False,
                    input_type=InputType.UNKNOWN,
                    error_message=str(e),
                    metadata={'file_path': str(input_path)}
                ))
        
        return results

    def _apply_rag_and_llm(self, 
                          result: ProcessingResult, 
                          query: Optional[str], 
                          module_key: Optional[str], 
                          apply_llm: bool) -> ProcessingResult:
        """Apply RAG query and LLM processing to a result"""
        # Apply RAG query if provided
        if query and result.chunks and HAS_RAG:
            try:
                rag_result = enhanced_query_documents(
                    query, 
                    result.chunks, 
                    self.config
                )
                result.rag_result = rag_result
            except Exception as e:
                self.logger.warning(f"RAG processing failed: {e}")
        
        # Apply module processing if requested
        if module_key and HAS_MODULES and apply_llm:
            try:
                # Use the new enhanced dynamic module processor
                from .modules.enhanced_dynamic_module import enhanced_dynamic_module_output
                
                text_content = result.extracted_text or result.transcript or ""
                llm_result = enhanced_dynamic_module_output(
                    text_content=text_content,
                    module_key=module_key,
                    metadata={
                        "input_type": result.input_type.value if hasattr(result, 'input_type') else None,
                        "processing_metadata": result.metadata
                    }
                )
                result.llm_output = llm_result
            except Exception as e:
                # Fallback to legacy processor if enhanced one fails
                self.logger.warning(f"Enhanced module processing failed, trying legacy: {e}")
                try:
                    llm_result = dynamic_module_output(
                        module_key=module_key,
                        provided_data=result.extracted_text or result.transcript or ""
                    )
                    result.llm_output = llm_result
                except Exception as e2:
                    self.logger.warning(f"Module processing failed completely: {e2}")
        
        return result


# Factory function for convenience
def create_pipeline_orchestrator(config_path: Optional[str] = None) -> PipelineOrchestrator:
    """Create a pipeline orchestrator instance"""
    return PipelineOrchestrator(config_path)
