import logging
from typing import Optional, List, Dict
import os
import sys
import requests
import tempfile
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import soundfile as sf

# Optional imports for microphone recording
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

def download_model_if_needed(model_size: str, models_dir: str = "models") -> str:
    """
    Ensure the Whisper model is present in the models directory. Download if missing.
    Downloads from HuggingFace if not found locally, with progress bar.
    Returns the path to the model directory.
    """
    model_path = Path(models_dir) / model_size
    if model_path.exists():
        return str(model_path)
    # Download model from HuggingFace (or other source)
    print(f"[INFO] Model '{model_size}' not found. Downloading to {model_path} ...")
    # Use requests to stream download with progress bar
    # Example: https://huggingface.co/Systran/faster-whisper-large-v3-turbo/resolve/main/model.bin
    # You may need to adjust the URL for your model repo
    base_url = f"https://huggingface.co/Systran/faster-whisper-{model_size}/resolve/main/model.bin"
    model_path.mkdir(parents=True, exist_ok=True)
    dest_file = model_path / "model.bin"
    response = requests.get(base_url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192
    with open(dest_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = int(downloaded * 100 / total) if total else 0
                sys.stdout.write(f"\r[DOWNLOAD] {percent}% complete")
                sys.stdout.flush()
    print("\n[INFO] Model download complete.")
    return str(model_path)

MODEL_CACHE = {}

def get_whisper_pipe(model_id=None, config=None):
    """Return a HuggingFace ASR pipeline for the given model_id, with caching. Fallback to default if model fails."""
    global MODEL_CACHE
    DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
    # Use config-driven settings
    if config is None:
        from config import load_config
        config = load_config()
    model_id = model_id or config.get('whisper_model', DEFAULT_MODEL)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    os.makedirs(models_dir, exist_ok=True)
    # Handle HuggingFace/OpenAI model
    if '/' in model_id and not model_id.endswith('.gguf'):
        local_path = os.path.join(models_dir, model_id.replace('/', '_'))
        if not os.path.exists(local_path):
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                print(f"[INFO] Downloading Whisper model {model_id} to {local_path}...")
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
                processor = AutoProcessor.from_pretrained(model_id)
                model.save_pretrained(local_path)
                processor.save_pretrained(local_path)
            except Exception as e:
                print(f"[ERROR] Could not download or save Whisper model '{model_id}': {e}")
                raise
        model_path = local_path
    # Handle GGUF model
    elif model_id.endswith('.gguf'):
        model_path = os.path.join(models_dir, model_id)
        if not os.path.exists(model_path):
            print(f"[ERROR] GGUF model file '{model_path}' not found. Please place it in the models/ directory.")
            raise FileNotFoundError(f"GGUF model file '{model_path}' not found.")
    # Handle local path
    else:
        model_path = model_id if os.path.exists(model_id) else os.path.join(models_dir, model_id)
        if not os.path.exists(model_path):
            print(f"[ERROR] Whisper model '{model_path}' not found. Please check your config.")
            raise FileNotFoundError(f"Whisper model '{model_path}' not found.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        MODEL_CACHE[model_id] = pipe
        return pipe
    except Exception as e:
        print(f"[WARNING] Could not load Whisper model '{model_id}': {e}. Falling back to default '{DEFAULT_MODEL}'.")
        if DEFAULT_MODEL in MODEL_CACHE:
            return MODEL_CACHE[DEFAULT_MODEL]
        # Try to load default model
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                DEFAULT_MODEL, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(DEFAULT_MODEL)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            MODEL_CACHE[DEFAULT_MODEL] = pipe
            return pipe
        except Exception as e2:
            print(f"[ERROR] Could not load default Whisper model '{DEFAULT_MODEL}': {e2}. Transcription will fail.")
            raise

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

def get_transcribe_pipe(model_id=None, config=None):
    """Return a transcription pipeline for the given model_id, using faster-whisper if requested, else transformers."""
    if model_id and model_id.startswith("faster-whisper/"):
        if not HAS_FASTER_WHISPER:
            raise ImportError("faster-whisper is not installed. Please run 'pip install faster-whisper'.")
        model_name = model_id.replace("faster-whisper/", "")
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"[INFO] Downloading faster-whisper model '{model_name}' to {model_path} ...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"Systran/faster-whisper-{model_name}", local_dir=model_path, local_dir_use_symlinks=False)
        return ("faster-whisper", model_path)
    else:
        return ("transformers", get_whisper_pipe(model_id, config))

# Dynamically import improved VAD
vad_path = Path(__file__).parent / "audio_vad.py"
if str(vad_path.parent) not in sys.path:
    sys.path.insert(0, str(vad_path.parent))
import importlib.util
vad_spec = importlib.util.spec_from_file_location("audio_vad", str(vad_path))
audio_vad = importlib.util.module_from_spec(vad_spec)
vad_spec.loader.exec_module(audio_vad)
run_vad = audio_vad.run_vad

def transcribe_audio_fastwhisper(audio_path, return_segments=True, language=None, whisper_model=None, config=None):
    """
    Transcribe audio using faster-whisper if requested, else HuggingFace Whisper pipeline.
    Returns a list of segments (with start/end if available) or just the text.
    """
    if config is None:
        from config import load_config
        config = load_config()
    whisper_model = whisper_model or config.get('whisper_model', 'base')
    # Use faster-whisper if requested
    backend, pipe = get_transcribe_pipe(whisper_model, config)
    import soundfile as sf
    import numpy as np
    audio, sr = sf.read(audio_path)
    # --- VAD: restrict transcription to speech segments only ---
    vad_segments = None
    if config is not None and config.get('enable_vad', True):
        vad_segments = run_vad(audio, sr, config)
        if vad_segments and len(vad_segments) > 0:
            # Only keep speech regions
            audio = np.concatenate([audio[start:end] for start, end in vad_segments])
    if backend == "faster-whisper":
        model = FasterWhisperModel(pipe, device="cuda" if torch.cuda.is_available() else "cpu")
        segments, info = model.transcribe(audio, beam_size=5, language=language, word_timestamps=True)
        segs = []
        for seg in segments:
            segs.append({
                "text": seg.text,
                "start": seg.start,
                "end": seg.end
            })
        return segs if return_segments else " ".join([s["text"] for s in segs])
    else:
        input_audio = {"array": audio, "sampling_rate": sr}
        result = pipe(input_audio, return_timestamps=True, generate_kwargs={"task": "transcribe"})
        if return_segments:
            segments = []
            for seg in result.get("chunks", []):
                segments.append({
                    "text": seg["text"],
                    "start": seg["timestamp"][0] if "timestamp" in seg else None,
                    "end": seg["timestamp"][1] if "timestamp" in seg else None
                })
            if not segments:
                segments = [{"text": result["text"], "start": None, "end": None}]
            return segments
        else:
            return result["text"]


def stream_microphone_transcription(output_path: Optional[str] = None, 
                                   duration: int = 10,
                                   config=None) -> tuple[str, List[Dict]]:
    """
    Record audio from microphone in real-time and transcribe it.
    
    Args:
        output_path: Optional path to save the recorded audio
        duration: Recording duration in seconds (default 10)
        config: Configuration object
        
    Returns:
        Tuple of (transcript_text, segments_list)
    """
    try:
        import sounddevice as sd
        import tempfile
        
        if config is None:
            from .config import load_config
            config = load_config()
            
        sample_rate = config.get('audio_sample_rate', 16000)
        
        logging.info(f"Recording audio from microphone for {duration} seconds...")
        
        # Record audio from microphone
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        
        # Save to file (temporary or specified path)
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        sf.write(output_path, audio_data, sample_rate)
        logging.info(f"Audio saved to: {output_path}")
        
        # Transcribe the recorded audio
        segments = transcribe_audio_fastwhisper(
            output_path, 
            return_segments=True, 
            config=config
        )
        
        if not segments:
            return "", []
            
        # Extract full transcript
        transcript = " ".join([seg.get('text', '') for seg in segments])
        
        logging.info(f"Transcription completed: {len(transcript)} characters")
        return transcript, segments
        
    except ImportError:
        logging.error("sounddevice is required for microphone recording. Install with: pip install sounddevice")
        raise ImportError("sounddevice is required for microphone recording")
    except Exception as e:
        logging.error(f"Error in stream_microphone_transcription: {e}")
        raise


def record_and_transcribe_chunks(chunk_duration: float = 2.0,
                                total_duration: float = 30.0,
                                config=None) -> tuple[str, List[Dict]]:
    """
    Record audio in chunks and transcribe each chunk for real-time processing.
    
    Args:
        chunk_duration: Duration of each audio chunk in seconds
        total_duration: Total recording duration in seconds
        config: Configuration object
        
    Returns:
        Tuple of (full_transcript, all_segments)
    """
    try:
        import sounddevice as sd
        import queue
        import threading
        import time
        
        if config is None:
            from .config import load_config
            config = load_config()
            
        sample_rate = config.get('audio_sample_rate', 16000)
        chunk_samples = int(chunk_duration * sample_rate)
        
        # Queue for audio chunks
        audio_queue = queue.Queue()
        all_segments = []
        full_transcript = ""
        
        def audio_callback(indata, frames, time, status):
            """Callback function for audio input"""
            if status:
                logging.warning(f"Audio input status: {status}")
            audio_queue.put(indata.copy())
        
        def transcribe_worker():
            """Worker thread for transcribing audio chunks"""
            nonlocal full_transcript, all_segments
            
            while True:
                try:
                    # Get audio chunk from queue
                    chunk_data = audio_queue.get(timeout=1.0)
                    
                    # Save chunk to temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    sf.write(temp_file.name, chunk_data, sample_rate)
                    
                    # Transcribe chunk
                    segments = transcribe_audio_fastwhisper(
                        temp_file.name,
                        return_segments=True,
                        config=config
                    )
                    
                    # Add to results
                    if segments:
                        chunk_text = " ".join([seg.get('text', '') for seg in segments])
                        full_transcript += chunk_text + " "
                        all_segments.extend(segments)
                        logging.info(f"Transcribed chunk: {chunk_text}")
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)
                    audio_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error transcribing chunk: {e}")
                    audio_queue.task_done()
        
        # Start transcription worker thread
        transcribe_thread = threading.Thread(target=transcribe_worker, daemon=True)
        transcribe_thread.start()
        
        logging.info(f"Starting chunked recording for {total_duration} seconds...")
        
        # Start audio stream
        with sd.InputStream(samplerate=sample_rate,
                           channels=1,
                           dtype=np.float32,
                           blocksize=chunk_samples,
                           callback=audio_callback):
            
            # Record for specified duration
            time.sleep(total_duration)
        
        # Wait for remaining chunks to be processed
        audio_queue.join()
        
        logging.info(f"Chunked transcription completed: {len(full_transcript)} characters")
        return full_transcript.strip(), all_segments
        
    except ImportError:
        logging.error("sounddevice is required for chunked recording. Install with: pip install sounddevice")
        raise ImportError("sounddevice is required for chunked recording")
    except Exception as e:
        logging.error(f"Error in record_and_transcribe_chunks: {e}")
        raise


def stream_microphone_transcription_interactive(output_path: Optional[str] = None, 
                                               config=None) -> tuple[str, List[Dict]]:
    """
    Record audio from microphone with start/stop control (press Enter to stop).
    
    Args:
        output_path: Optional path to save the recorded audio
        config: Configuration object
        
    Returns:
        Tuple of (transcript_text, segments_list)
    """
    try:
        import sounddevice as sd
        import tempfile
        import threading
        import queue
        import select
        import sys
        
        if config is None:
            from .config import load_config
            config = load_config()
            
        sample_rate = config.get('audio_sample_rate', 16000)
        
        logging.info("Starting microphone recording - press Enter to stop...")
        
        # Audio queue and recording flag
        audio_chunks = []
        recording = threading.Event()
        recording.set()
        
        def audio_callback(indata, frames, time, status):
            """Callback function for audio input"""
            if status:
                logging.warning(f"Audio input status: {status}")
            if recording.is_set():
                audio_chunks.append(indata.copy())
        
        def wait_for_enter():
            """Wait for Enter key press"""
            input()  # Wait for Enter key
            recording.clear()
            logging.info("Recording stopped by user")
        
        # Start input monitoring thread
        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()
        
        # Start audio stream
        with sd.InputStream(samplerate=sample_rate,
                           channels=1,
                           dtype=np.float32,
                           callback=audio_callback):
            
            # Wait until recording is stopped
            while recording.is_set():
                sd.sleep(100)  # Sleep 100ms and check again
        
        # Combine all audio chunks
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0)
        else:
            logging.warning("No audio data recorded")
            return "", []
        
        # Save to file (temporary or specified path)
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        sf.write(output_path, audio_data, sample_rate)
        logging.info(f"Audio saved to: {output_path}")
        
        # Transcribe the recorded audio
        segments = transcribe_audio_fastwhisper(
            output_path, 
            return_segments=True, 
            config=config
        )
        
        if not segments:
            return "", []
            
        # Extract full transcript
        transcript = " ".join([seg.get('text', '') for seg in segments])
        
        logging.info(f"Transcription completed: {len(transcript)} characters")
        return transcript, segments
        
    except ImportError:
        logging.error("sounddevice is required for microphone recording. Install with: pip install sounddevice")
        raise ImportError("sounddevice is required for microphone recording")
    except Exception as e:
        logging.error(f"Error in stream_microphone_transcription_interactive: {e}")
        raise


def main():
    """Entry point for intv-audio command"""
    import argparse
    
    parser = argparse.ArgumentParser(description='INTV Audio Processing Utilities')
    parser.add_argument('--transcribe', '-t', type=str, help='Audio file to transcribe')
    parser.add_argument('--microphone', '-m', action='store_true', help='Record from microphone and transcribe')
    parser.add_argument('--duration', '-d', type=int, default=10, help='Recording duration in seconds (default: 10)')
    parser.add_argument('--output', '-o', type=str, help='Output file path for recorded audio')
    parser.add_argument('--chunks', '-c', action='store_true', help='Use chunked real-time transcription')
    parser.add_argument('--model', type=str, help='Whisper model to use')
    
    args = parser.parse_args()
    
    try:
        from .config import load_config
        config = load_config()
    except:
        config = {}
    
    if args.model:
        config['whisper_model'] = args.model
    
    try:
        if args.transcribe:
            print(f"Transcribing audio file: {args.transcribe}")
            segments = transcribe_audio_fastwhisper(args.transcribe, config=config)
            if isinstance(segments, list):
                for seg in segments:
                    print(f"[{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s]: {seg['text']}")
            else:
                print(segments)
                
        elif args.microphone:
            if args.chunks:
                print(f"Starting chunked recording for {args.duration} seconds...")
                transcript, segments = record_and_transcribe_chunks(
                    total_duration=args.duration,
                    config=config
                )
            else:
                print(f"Recording from microphone for {args.duration} seconds...")
                transcript, segments = stream_microphone_transcription(
                    args.output, 
                    args.duration, 
                    config
                )
            
            print(f"\nTranscript: {transcript}")
            print(f"Total segments: {len(segments)}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
