"""
Unified Pipeline Orchestrator for INTV Application
Handles routing and processing of different input types through appropriate pipelines.
"""

import os
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Local imports
from .rag import chunk_document, process_with_retriever_and_llm
from .ocr import ocr_file, ocr_image
from .audio_transcribe import transcribe_audio_fastwhisper
from .audio_diarization import diarize_audio
from .llm import rag_llm_pipeline, analyze_chunks
from .config import load_config


class InputType(Enum):
    """Supported input types for pipeline processing"""
    DOCUMENT = "document"  # Text/PDF/DOCX for RAG
    IMAGE = "image"        # Image for OCR -> RAG
    AUDIO_FILE = "audio_file"    # Audio file for transcription -> RAG
    AUDIO_STREAM = "audio_stream"  # Realtime audio -> transcription -> RAG
    UNKNOWN = "unknown"


@dataclass
class ProcessingResult:
    """Result of pipeline processing"""
    success: bool
    input_type: InputType
    extracted_text: Optional[str] = None
    chunks: Optional[List[str]] = None
    transcript: Optional[str] = None
    segments: Optional[List[Dict]] = None
    rag_result: Optional[Dict] = None
    llm_output: Optional[Dict] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None


class PipelineOrchestrator:
    """
    Main orchestrator for processing different input types through appropriate pipelines
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline orchestrator"""
        self.config = load_config(config_path) if config_path else load_config()
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already done
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def detect_input_type(self, input_path: Union[str, Path]) -> InputType:
        """
        Detect the type of input file to determine processing pipeline
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        
        if not input_path.exists():
            return InputType.UNKNOWN
        
        # Get file extension and MIME type
        ext = input_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(input_path))
        
        # Document types for RAG pipeline
        document_extensions = {'.txt', '.md', '.rtf', '.docx', '.pdf'}
        # Image types for OCR pipeline  
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        # Audio types for transcription pipeline
        audio_extensions = {'.wav', '.mp3', '.m4a', '.mp4', '.flac', '.ogg', '.webm'}
        
        if ext in document_extensions:
            return InputType.DOCUMENT
        elif ext in image_extensions:
            return InputType.IMAGE
        elif ext in audio_extensions:
            return InputType.AUDIO_FILE
        elif mime_type:
            if mime_type.startswith('text/'):
                return InputType.DOCUMENT
            elif mime_type.startswith('image/'):
                return InputType.IMAGE
            elif mime_type.startswith('audio/') or mime_type.startswith('video/'):
                return InputType.AUDIO_FILE
        
        return InputType.UNKNOWN
    
    def process_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        Process document through RAG pipeline
        """
        try:
            self.logger.info(f"Processing document: {file_path}")
            
            # Chunk the document
            chunks = chunk_document(str(file_path), config=self.config)
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.DOCUMENT,
                    error_message="No text could be extracted from document"
                )
            
            # Extract full text
            extracted_text = "\n".join(chunks)
            
            self.logger.info(f"Document chunked into {len(chunks)} chunks")
            
            return ProcessingResult(
                success=True,
                input_type=InputType.DOCUMENT,
                extracted_text=extracted_text,
                chunks=chunks,
                metadata={
                    'num_chunks': len(chunks),
                    'total_length': len(extracted_text)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            return ProcessingResult(
                success=False,
                input_type=InputType.DOCUMENT,
                error_message=str(e)
            )
    
    def process_image(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        Process image through OCR -> RAG pipeline
        """
        try:
            self.logger.info(f"Processing image with OCR: {file_path}")
            
            # Extract text using OCR
            extracted_text = ocr_file(str(file_path), config=self.config)
            
            if not extracted_text or not extracted_text.strip():
                return ProcessingResult(
                    success=False,
                    input_type=InputType.IMAGE,
                    error_message="No text could be extracted from image"
                )
            
            # Chunk the extracted text
            from .utils import chunk_text
            chunks = chunk_text(extracted_text, config=self.config)
            
            self.logger.info(f"Extracted {len(extracted_text)} characters from image, chunked into {len(chunks)} chunks")
            
            return ProcessingResult(
                success=True,
                input_type=InputType.IMAGE,
                extracted_text=extracted_text,
                chunks=chunks,
                metadata={
                    'num_chunks': len(chunks),
                    'text_length': len(extracted_text)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing image {file_path}: {e}")
            return ProcessingResult(
                success=False,
                input_type=InputType.IMAGE,
                error_message=str(e)
            )
    
    def process_audio_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        Process audio file through transcription + diarization -> RAG pipeline
        """
        try:
            self.logger.info(f"Processing audio file: {file_path}")
            
            # Transcribe audio
            segments = transcribe_audio_fastwhisper(
                str(file_path),
                return_segments=True,
                config=self.config
            )
            
            if not segments:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.AUDIO_FILE,
                    error_message="No transcript could be generated from audio"
                )
            
            # Extract transcript text
            transcript = " ".join([seg.get('text', '') for seg in segments])
            
            # Apply diarization if enabled
            diarized_segments = None
            if self.config.get('enable_diarization', False):
                try:
                    diarized_segments = diarize_audio(
                        str(file_path),
                        config=self.config
                    )
                    self.logger.info(f"Diarization completed with {len(diarized_segments)} segments")
                except Exception as e:
                    self.logger.warning(f"Diarization failed: {e}, continuing with transcription only")
            
            # Chunk the transcript for RAG
            from .utils import chunk_text
            chunks = chunk_text(transcript, config=self.config)
            
            self.logger.info(f"Audio transcribed to {len(transcript)} characters, chunked into {len(chunks)} chunks")
            
            return ProcessingResult(
                success=True,
                input_type=InputType.AUDIO_FILE,
                extracted_text=transcript,
                transcript=transcript,
                chunks=chunks,
                segments=segments,
                metadata={
                    'num_segments': len(segments),
                    'num_chunks': len(chunks),
                    'transcript_length': len(transcript),
                    'has_diarization': diarized_segments is not None,
                    'diarized_segments': diarized_segments
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing audio file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                input_type=InputType.AUDIO_FILE,
                error_message=str(e)
            )
    
    def process_audio_stream(self, output_path: Optional[str] = None) -> ProcessingResult:
        """
        Process realtime audio stream through transcription + diarization -> RAG pipeline
        """
        try:
            self.logger.info("Starting realtime audio stream processing")
            
            try:
                # Check if stream_microphone_transcription_interactive is available
                from .audio_transcribe import stream_microphone_transcription_interactive
                
                # Stream and transcribe microphone input with start/stop control
                transcript, segments = stream_microphone_transcription_interactive(output_path)
                
            except (ImportError, AttributeError) as e:
                return ProcessingResult(
                    success=False,
                    input_type=InputType.AUDIO_STREAM,
                    error_message=f"Microphone recording requires additional dependencies or is not implemented: {str(e)}"
                )
            
            if not transcript or not transcript.strip():
                return ProcessingResult(
                    success=False,
                    input_type=InputType.AUDIO_STREAM,
                    error_message="No transcript generated from microphone input"
                )
            
            # Apply diarization if enabled and we have audio file
            diarized_segments = None
            if output_path and self.config.get('enable_diarization', False):
                try:
                    diarized_segments = diarize_audio(
                        output_path,
                        config=self.config
                    )
                    self.logger.info(f"Diarization completed with {len(diarized_segments)} segments")
                except Exception as e:
                    self.logger.warning(f"Diarization failed: {e}, continuing with transcription only")
            
            # Chunk the transcript for RAG
            from .utils import chunk_text
            chunks = chunk_text(transcript, config=self.config)
            
            self.logger.info(f"Stream transcribed to {len(transcript)} characters, chunked into {len(chunks)} chunks")
            
            return ProcessingResult(
                success=True,
                input_type=InputType.AUDIO_STREAM,
                extracted_text=transcript,
                transcript=transcript,
                chunks=chunks,
                segments=segments,
                metadata={
                    'num_segments': len(segments) if segments else 0,
                    'num_chunks': len(chunks),
                    'transcript_length': len(transcript),
                    'has_diarization': diarized_segments is not None,
                    'diarized_segments': diarized_segments,
                    'output_file': output_path
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing audio stream: {e}")
            return ProcessingResult(
                success=False,
                input_type=InputType.AUDIO_STREAM,
                error_message=str(e)
            )
    
    def apply_rag_llm(self, result: ProcessingResult, module_key: Optional[str] = None, 
                      query: Optional[str] = None) -> ProcessingResult:
        """
        Apply RAG + LLM processing to extracted content
        """
        if not result.success or not result.chunks:
            return result
        
        try:
            if module_key:
                self.logger.info(f"Applying RAG + LLM processing with module: {module_key}")
            else:
                self.logger.info("Applying basic RAG processing without module")
            
            # Default query if none provided
            if not query:
                query = "Analyze and summarize this content"
            
            # Apply RAG retrieval and LLM processing
            rag_result = process_with_retriever_and_llm(
                chunks=result.chunks,
                query=query,
                config=self.config
            )
            
            # Apply full LLM pipeline if module specified
            llm_output = None
            if module_key and result.extracted_text:
                # Save extracted text to temporary file for LLM pipeline
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                    tmp_file.write(result.extracted_text)
                    tmp_path = tmp_file.name
                
                try:
                    llm_output = rag_llm_pipeline(
                        document_path=tmp_path,
                        module_key=module_key,
                        config=self.config
                    )
                finally:
                    os.unlink(tmp_path)
            
            # Update result with RAG/LLM outputs
            result.rag_result = rag_result
            result.llm_output = llm_output
            
            if module_key:
                self.logger.info("RAG + LLM processing completed successfully")
            else:
                self.logger.info("Basic RAG processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in RAG + LLM processing: {e}")
            result.error_message = f"RAG/LLM processing failed: {str(e)}"
        
        return result
    
    def process(self, input_path: Union[str, Path], 
                module_key: Optional[str] = None,
                query: Optional[str] = None,
                apply_llm: bool = True) -> ProcessingResult:
        """
        Main processing method - detects input type and routes to appropriate pipeline
        """
        self.logger.info(f"Starting pipeline processing for: {input_path}")
        
        # Detect input type
        input_type = self.detect_input_type(input_path)
        self.logger.info(f"Detected input type: {input_type.value}")
        
        # Route to appropriate processing pipeline
        if input_type == InputType.DOCUMENT:
            result = self.process_document(input_path)
        elif input_type == InputType.IMAGE:
            result = self.process_image(input_path)
        elif input_type == InputType.AUDIO_FILE:
            result = self.process_audio_file(input_path)
        else:
            return ProcessingResult(
                success=False,
                input_type=input_type,
                error_message=f"Unsupported input type: {input_type.value}"
            )
        
        # Apply RAG + LLM if requested and processing was successful
        if apply_llm and result.success:
            result = self.apply_rag_llm(result, module_key, query)
        
        self.logger.info(f"Pipeline processing completed. Success: {result.success}")
        return result
    
    def process_microphone(self, module_key: Optional[str] = None,
                          query: Optional[str] = None,
                          output_path: Optional[str] = None,
                          apply_llm: bool = True) -> ProcessingResult:
        """
        Process realtime microphone input
        """
        self.logger.info("Starting microphone processing")
        
        # Process audio stream
        result = self.process_audio_stream(output_path)
        
        # Apply RAG + LLM if requested and processing was successful
        if apply_llm and result.success and module_key:
            result = self.apply_rag_llm(result, module_key, query)
        
        self.logger.info(f"Microphone processing completed. Success: {result.success}")
        return result


def create_pipeline_orchestrator(config_path: Optional[str] = None) -> PipelineOrchestrator:
    """Factory function to create a pipeline orchestrator"""
    return PipelineOrchestrator(config_path)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Simple CLI test interface
    if len(sys.argv) < 2:
        print("Usage: python pipeline_orchestrator.py <file_path> [module_key]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    module_key = sys.argv[2] if len(sys.argv) > 2 else "adult"
    
    orchestrator = create_pipeline_orchestrator()
    result = orchestrator.process(file_path, module_key)
    
    print(f"Processing result: {result.success}")
    if result.success:
        print(f"Input type: {result.input_type.value}")
        print(f"Text length: {len(result.extracted_text) if result.extracted_text else 0}")
        print(f"Chunks: {len(result.chunks) if result.chunks else 0}")
        if result.transcript:
            print(f"Transcript length: {len(result.transcript)}")
        if result.rag_result:
            print(f"RAG result available")
        if result.llm_output:
            print(f"LLM output available")
    else:
        print(f"Error: {result.error_message}")
