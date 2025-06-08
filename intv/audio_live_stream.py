"""
Live speech processing module for INTV
Continuous microphone processing with silence detection and automatic RAG integration
"""

import logging
import threading
import queue
import time
import tempfile
import os
from typing import Optional, List, Dict, Callable, Any
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    logger.warning("Audio dependencies not available for live streaming")

try:
    from .audio_transcribe import transcribe_audio_fastwhisper
    from .audio_vad import detect_voice_activity_pyannote, apply_vad_filter_enhanced
    HAS_TRANSCRIPTION = True
except ImportError:
    HAS_TRANSCRIPTION = False


class LiveSpeechProcessor:
    """
    Continuous speech processing with automatic silence detection and RAG integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize live speech processor"""
        self.config = config or {}
        self.sample_rate = self.config.get('audio_sample_rate', 16000)
        self.chunk_duration = self.config.get('live_chunk_duration', 2.0)  # seconds
        self.silence_threshold = self.config.get('silence_threshold', 1.0)  # seconds of silence before processing
        self.max_buffer_duration = self.config.get('max_buffer_duration', 300.0)  # 5 minutes max
        
        # Processing state
        self.is_recording = False
        self.audio_buffer = []
        self.silence_counter = 0
        self.last_speech_time = time.time()
        self.processing_callback = None
        
        # Threading
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.recording_thread = None
        
        logger.info(f"LiveSpeechProcessor initialized: {self.sample_rate}Hz, {self.chunk_duration}s chunks")
    
    def set_processing_callback(self, callback: Callable[[str, List[Dict]], None]):
        """Set callback function to handle processed transcripts"""
        self.processing_callback = callback
    
    def start_continuous_processing(self, 
                                  auto_stop: bool = True,
                                  manual_control: bool = False) -> bool:
        """
        Start continuous speech processing
        
        Args:
            auto_stop: Automatically stop after silence detection
            manual_control: Allow manual start/stop control
            
        Returns:
            Success status
        """
        if not HAS_AUDIO:
            logger.error("Audio dependencies not available")
            return False
        
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        try:
            self.is_recording = True
            self.audio_buffer = []
            self.silence_counter = 0
            self.last_speech_time = time.time()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_worker,
                daemon=True
            )
            self.processing_thread.start()
            
            # Start recording
            logger.info("Starting continuous speech processing...")
            logger.info("Speak into the microphone. Processing will begin automatically.")
            
            if manual_control:
                logger.info("Press Enter to stop recording manually.")
                input_thread = threading.Thread(target=self._wait_for_stop, daemon=True)
                input_thread.start()
            
            # Audio recording loop
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback
            ):
                while self.is_recording:
                    # Check for silence timeout
                    if auto_stop and time.time() - self.last_speech_time > self.silence_threshold:
                        if len(self.audio_buffer) > 0:
                            logger.info("Silence detected - processing accumulated audio...")
                            self._process_buffer()
                            self.audio_buffer = []
                            self.last_speech_time = time.time()
                    
                    # Check for max buffer duration
                    if len(self.audio_buffer) > 0:
                        buffer_duration = len(np.concatenate(self.audio_buffer)) / self.sample_rate
                        if buffer_duration > self.max_buffer_duration:
                            logger.info("Maximum buffer duration reached - processing...")
                            self._process_buffer()
                            self.audio_buffer = []
                            self.last_speech_time = time.time()
                    
                    time.sleep(0.1)  # Check every 100ms
            
            # Process any remaining audio
            if len(self.audio_buffer) > 0:
                logger.info("Processing final audio buffer...")
                self._process_buffer()
            
            logger.info("Continuous speech processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            self.is_recording = False
            return False
        finally:
            self.is_recording = False
    
    def stop_processing(self):
        """Stop continuous processing"""
        logger.info("Stopping speech processing...")
        self.is_recording = False
    
    def _audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            logger.warning(f"Audio input status: {status}")
        
        if self.is_recording:
            # Simple voice activity detection (energy-based)
            audio_chunk = indata[:, 0]  # Get mono channel
            energy = np.mean(audio_chunk ** 2)
            
            # Threshold for voice activity (adjust based on your environment)
            voice_threshold = self.config.get('voice_energy_threshold', 0.001)
            
            if energy > voice_threshold:
                self.audio_buffer.append(audio_chunk.copy())
                self.last_speech_time = time.time()
                self.silence_counter = 0
            else:
                self.silence_counter += 1
    
    def _wait_for_stop(self):
        """Wait for manual stop signal"""
        try:
            input()  # Wait for Enter key
            self.stop_processing()
        except:
            pass
    
    def _processing_worker(self):
        """Background worker for processing audio"""
        logger.info("Audio processing worker started")
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Process any queued audio
                try:
                    audio_data, timestamp = self.audio_queue.get(timeout=1.0)
                    self._process_audio_data(audio_data, timestamp)
                    self.audio_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _process_buffer(self):
        """Process the current audio buffer"""
        if not self.audio_buffer:
            return
        
        try:
            # Concatenate audio buffer
            audio_data = np.concatenate(self.audio_buffer)
            timestamp = time.time()
            
            # Queue for processing
            self.audio_queue.put((audio_data, timestamp))
            
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
    
    def _process_audio_data(self, audio_data: np.ndarray, timestamp: float):
        """Process audio data and trigger callbacks"""
        try:
            if not HAS_TRANSCRIPTION:
                logger.warning("Transcription not available")
                return
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_data, self.sample_rate)
            temp_file.close()
            
            # Transcribe audio
            segments = transcribe_audio_fastwhisper(
                temp_file.name,
                return_segments=True,
                config=self.config
            )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if segments:
                # Extract transcript
                transcript = " ".join([seg.get('text', '') for seg in segments])
                
                if transcript.strip():
                    logger.info(f"Transcribed: {transcript}")
                    
                    # Trigger callback if set
                    if self.processing_callback:
                        try:
                            self.processing_callback(transcript, segments)
                        except Exception as e:
                            logger.error(f"Error in processing callback: {e}")
                    
                    # Optional: Trigger RAG processing
                    if self.config.get('auto_rag_processing', False):
                        self._trigger_rag_processing(transcript, segments)
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
    
    def _trigger_rag_processing(self, transcript: str, segments: List[Dict]):
        """Trigger RAG processing for the transcript"""
        try:
            from .rag import enhanced_chunk_document, enhanced_query_documents
            
            # Chunk the transcript
            chunks = enhanced_chunk_document(transcript, self.config)
            
            # Process with RAG if query is configured
            rag_query = self.config.get('auto_rag_query')
            if rag_query and chunks:
                rag_result = enhanced_query_documents(rag_query, chunks, self.config)
                logger.info(f"RAG processing completed: {len(chunks)} chunks")
                
                # Optional: Trigger LLM processing
                if self.config.get('auto_llm_processing', False):
                    self._trigger_llm_processing(transcript, rag_result)
        
        except ImportError:
            logger.warning("RAG modules not available")
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
    
    def _trigger_llm_processing(self, transcript: str, rag_result: Any):
        """Trigger LLM processing for the transcript"""
        try:
            from .modules.dynamic_module import dynamic_module_output
            
            module_key = self.config.get('auto_llm_module')
            if module_key:
                llm_result = dynamic_module_output(
                    module_key=module_key,
                    provided_data=transcript
                )
                logger.info(f"LLM processing completed for module: {module_key}")
        
        except ImportError:
            logger.warning("LLM modules not available")
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")


def create_live_processor(config: Optional[Dict] = None) -> Optional[LiveSpeechProcessor]:
    """Create and return a live speech processor"""
    if not HAS_AUDIO:
        logger.error("Audio dependencies not available for live processing")
        return None
    
    return LiveSpeechProcessor(config)


def start_continuous_speech_processing(
    config: Optional[Dict] = None,
    processing_callback: Optional[Callable[[str, List[Dict]], None]] = None,
    auto_stop: bool = True,
    manual_control: bool = False
) -> bool:
    """
    Convenience function to start continuous speech processing
    
    Args:
        config: Configuration dictionary
        processing_callback: Function to handle processed transcripts
        auto_stop: Automatically stop after silence detection
        manual_control: Allow manual start/stop control
        
    Returns:
        Success status
    """
    processor = create_live_processor(config)
    if not processor:
        return False
    
    if processing_callback:
        processor.set_processing_callback(processing_callback)
    
    return processor.start_continuous_processing(auto_stop, manual_control)


def main():
    """CLI entry point for live speech processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Speech Processing')
    parser.add_argument('--manual', '-m', action='store_true', help='Manual start/stop control')
    parser.add_argument('--no-auto-stop', action='store_true', help='Disable automatic stop on silence')
    parser.add_argument('--duration', '-d', type=float, default=1.0, help='Silence threshold in seconds')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    config['silence_threshold'] = args.duration
    
    def processing_callback(transcript: str, segments: List[Dict]):
        """Simple callback to print transcripts"""
        print(f"\n[TRANSCRIPT] {transcript}\n")
    
    try:
        success = start_continuous_speech_processing(
            config=config,
            processing_callback=processing_callback,
            auto_stop=not args.no_auto_stop,
            manual_control=args.manual
        )
        
        if not success:
            print("Failed to start live speech processing")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
