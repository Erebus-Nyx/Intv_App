"""
Voice Activity Detection (VAD) module for INTV
Enhanced with pyannote/segmentation-3.0 and solero-vad improvements
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os
import tempfile

logger = logging.getLogger(__name__)

# Try to import pyannote for advanced VAD
try:
    from pyannote.audio import Pipeline, Model
    from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection
    import torch
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False
    logger.warning("pyannote.audio not available, falling back to basic VAD")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from .secrets file"""
    try:
        secrets_path = os.path.join(os.path.dirname(__file__), '../../.secrets')
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN=') or line.startswith('HUGGINGFACE_TOKEN='):
                        return line.split('=', 1)[1].strip().strip('"\'')
        
        # Also check environment variables
        import os
        return os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    except Exception as e:
        logger.warning(f"Could not read HuggingFace token: {e}")
        return None

def detect_voice_activity_pyannote(audio_path: str, 
                                 config: Optional[Dict] = None) -> List[Tuple[float, float]]:
    """
    Advanced VAD using pyannote/segmentation-3.0 with solero-vad improvements
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        
    Returns:
        List of (start_time, end_time) tuples for speech segments
    """
    if not HAS_PYANNOTE:
        logger.warning("pyannote.audio not available, falling back to basic VAD")
        return detect_voice_activity_basic(audio_path, config)
    
    try:
        # Get HuggingFace token for pyannote models
        hf_token = get_hf_token()
        if not hf_token:
            logger.warning("No HuggingFace token found. Some models may not be accessible.")
        
        # Initialize VAD pipeline with pyannote/segmentation-3.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use the segmentation model for VAD
        model_name = "pyannote/segmentation-3.0"
        if config and config.get('vad_model'):
            model_name = config.get('vad_model')
        
        logger.info(f"Loading VAD model: {model_name}")
        
        # Load the segmentation model
        model = Model.from_pretrained(model_name, use_auth_token=hf_token)
        
        # Create VAD pipeline with optimized parameters
        vad_pipeline = VoiceActivityDetection(segmentation=model)
        
        # Configure pipeline parameters for better performance
        vad_config = {
            "onset": config.get('vad_onset', 0.5) if config else 0.5,
            "offset": config.get('vad_offset', 0.5) if config else 0.5,
            "min_duration_on": config.get('vad_min_duration_on', 0.0) if config else 0.0,
            "min_duration_off": config.get('vad_min_duration_off', 0.0) if config else 0.0,
        }
        
        vad_pipeline.instantiate(vad_config)
        
        # Apply VAD to audio file
        logger.info(f"Applying VAD to: {audio_path}")
        vad_result = vad_pipeline(audio_path)
        
        # Convert pyannote timeline to list of tuples
        speech_segments = []
        for segment in vad_result.get_timeline():
            speech_segments.append((segment.start, segment.end))
        
        logger.info(f"VAD detected {len(speech_segments)} speech segments")
        return speech_segments
        
    except Exception as e:
        logger.warning(f"Pyannote VAD failed: {e}, falling back to basic VAD")
        return detect_voice_activity_basic(audio_path, config)

def detect_voice_activity_basic(audio_path: str, 
                              config: Optional[Dict] = None) -> List[Tuple[float, float]]:
    """
    Basic energy-based VAD fallback
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        
    Returns:
        List of (start_time, end_time) tuples for speech segments
    """
    try:
        if not HAS_SOUNDFILE:
            logger.error("soundfile not available for basic VAD")
            return []
        
        # Load audio file
        audio_data, sample_rate = sf.read(audio_path)
        
        # Use the original basic VAD function
        return detect_voice_activity(audio_data, sample_rate, 
                                   frame_duration=config.get('vad_frame_duration', 0.02) if config else 0.02,
                                   threshold=config.get('vad_threshold', 0.01) if config else 0.01)
        
    except Exception as e:
        logger.error(f"Basic VAD failed: {e}")
        return []

def detect_voice_activity(audio_data: np.ndarray, sample_rate: int = 16000, 
                         frame_duration: float = 0.02, threshold: float = 0.01) -> List[Tuple[float, float]]:
    """
    Simple voice activity detection using energy-based approach.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_duration: Frame duration in seconds
        threshold: Energy threshold for voice detection
        
    Returns:
        List of (start_time, end_time) tuples for voice segments
    """
    if len(audio_data) == 0:
        return []
    
    frame_size = int(sample_rate * frame_duration)
    hop_size = frame_size // 2
    
    voice_segments = []
    in_voice_segment = False
    segment_start = 0
    
    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i + frame_size]
        energy = np.mean(frame ** 2)
        
        time_pos = i / sample_rate
        
        if energy > threshold:
            if not in_voice_segment:
                segment_start = time_pos
                in_voice_segment = True
        else:
            if in_voice_segment:
                voice_segments.append((segment_start, time_pos))
                in_voice_segment = False
    
    # Close final segment if needed
    if in_voice_segment:
        voice_segments.append((segment_start, len(audio_data) / sample_rate))
    
    return voice_segments


def apply_vad_filter(audio_data: np.ndarray, sample_rate: int = 16000,
                    min_duration: float = 0.1) -> np.ndarray:
    """
    Apply VAD filtering to remove silent segments.
    
    Args:
        audio_data: Input audio samples
        sample_rate: Sample rate in Hz
        min_duration: Minimum segment duration to keep
        
    Returns:
        Filtered audio data
    """
    voice_segments = detect_voice_activity(audio_data, sample_rate)
    
    if not voice_segments:
        return np.array([])
    
    # Filter segments by minimum duration
    valid_segments = [(start, end) for start, end in voice_segments 
                     if end - start >= min_duration]
    
    if not valid_segments:
        return np.array([])
    
    # Concatenate voice segments
    filtered_audio = []
    for start, end in valid_segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        filtered_audio.append(audio_data[start_sample:end_sample])
    
    return np.concatenate(filtered_audio) if filtered_audio else np.array([])


def get_speech_timestamps_enhanced(audio_path: str, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Enhanced speech timestamp detection using best available VAD method
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        
    Returns:
        List of timestamp dictionaries with confidence scores
    """
    # Try pyannote VAD first
    if HAS_PYANNOTE and config and config.get('use_pyannote_vad', True):
        try:
            voice_segments = detect_voice_activity_pyannote(audio_path, config)
        except Exception as e:
            logger.warning(f"Pyannote VAD failed: {e}, falling back to basic VAD")
            voice_segments = detect_voice_activity_basic(audio_path, config)
    else:
        voice_segments = detect_voice_activity_basic(audio_path, config)
    
    # Convert to timestamp format with confidence scores
    timestamps = []
    for i, (start, end) in enumerate(voice_segments):
        timestamps.append({
            'start': start,
            'end': end,
            'confidence': 0.9 if HAS_PYANNOTE else 0.7,  # Higher confidence for pyannote
            'segment_id': i,
            'duration': end - start
        })
    
    return timestamps

def apply_vad_filter_enhanced(audio_path: str, 
                            config: Optional[Dict] = None,
                            min_duration: float = 0.1) -> str:
    """
    Apply enhanced VAD filtering and return path to filtered audio file
    
    Args:
        audio_path: Path to input audio file
        config: Configuration dictionary
        min_duration: Minimum segment duration to keep
        
    Returns:
        Path to filtered audio file
    """
    try:
        if not HAS_SOUNDFILE:
            logger.warning("soundfile not available, returning original audio")
            return audio_path
        
        # Get speech segments
        voice_segments = detect_voice_activity_pyannote(audio_path, config) if HAS_PYANNOTE else detect_voice_activity_basic(audio_path, config)
        
        if not voice_segments:
            logger.warning("No speech segments detected")
            return audio_path
        
        # Load original audio
        audio_data, sample_rate = sf.read(audio_path)
        
        # Filter segments by minimum duration
        valid_segments = [(start, end) for start, end in voice_segments 
                         if end - start >= min_duration]
        
        if not valid_segments:
            logger.warning("No valid speech segments after filtering")
            return audio_path
        
        # Extract and concatenate speech segments
        filtered_segments = []
        for start, end in valid_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            start_sample = max(0, min(start_sample, len(audio_data)))
            end_sample = max(start_sample, min(end_sample, len(audio_data)))
            
            if end_sample > start_sample:
                filtered_segments.append(audio_data[start_sample:end_sample])
        
        if not filtered_segments:
            logger.warning("No audio segments extracted")
            return audio_path
        
        # Concatenate filtered audio
        filtered_audio = np.concatenate(filtered_segments)
        
        # Save filtered audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, filtered_audio, sample_rate)
        temp_file.close()
        
        logger.info(f"VAD filtering completed: {len(valid_segments)} segments, {len(filtered_audio)/sample_rate:.2f}s total")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"VAD filtering failed: {e}")
        return audio_path

def get_speech_timestamps(audio_data: np.ndarray, sample_rate: int = 16000) -> List[dict]:
    """
    Get speech timestamps in a format compatible with external VAD libraries.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        
    Returns:
        List of timestamp dictionaries
    """
    voice_segments = detect_voice_activity(audio_data, sample_rate)
    
    timestamps = []
    for start, end in voice_segments:
        timestamps.append({
            'start': start,
            'end': end,
            'confidence': 0.8  # Simple confidence score
        })
    
    return timestamps

def get_speech_timestamps_enhanced(audio_path: str, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Enhanced speech timestamp detection using best available VAD method
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
        
    Returns:
        List of timestamp dictionaries with confidence scores
    """
    # Try pyannote VAD first
    if HAS_PYANNOTE and config and config.get('use_pyannote_vad', True):
        try:
            voice_segments = detect_voice_activity_pyannote(audio_path, config)
        except Exception as e:
            logger.warning(f"Pyannote VAD failed: {e}, falling back to basic VAD")
            voice_segments = detect_voice_activity_basic(audio_path, config)
    else:
        voice_segments = detect_voice_activity_basic(audio_path, config)
    
    # Convert to timestamp format with confidence scores
    timestamps = []
    for i, (start, end) in enumerate(voice_segments):
        timestamps.append({
            'start': start,
            'end': end,
            'confidence': 0.9 if HAS_PYANNOTE else 0.7,  # Higher confidence for pyannote
            'segment_id': i,
            'duration': end - start
        })
    
    return timestamps

def apply_vad_filter_enhanced(audio_path: str, 
                            config: Optional[Dict] = None,
                            min_duration: float = 0.1) -> str:
    """
    Apply enhanced VAD filtering and return path to filtered audio file
    
    Args:
        audio_path: Path to input audio file
        config: Configuration dictionary
        min_duration: Minimum segment duration to keep
        
    Returns:
        Path to filtered audio file
    """
    try:
        if not HAS_SOUNDFILE:
            logger.warning("soundfile not available, returning original audio")
            return audio_path
        
        # Get speech segments
        voice_segments = detect_voice_activity_pyannote(audio_path, config) if HAS_PYANNOTE else detect_voice_activity_basic(audio_path, config)
        
        if not voice_segments:
            logger.warning("No speech segments detected")
            return audio_path
        
        # Load original audio
        audio_data, sample_rate = sf.read(audio_path)
        
        # Filter segments by minimum duration
        valid_segments = [(start, end) for start, end in voice_segments 
                         if end - start >= min_duration]
        
        if not valid_segments:
            logger.warning("No valid speech segments after filtering")
            return audio_path
        
        # Extract and concatenate speech segments
        filtered_segments = []
        for start, end in valid_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            start_sample = max(0, min(start_sample, len(audio_data)))
            end_sample = max(start_sample, min(end_sample, len(audio_data)))
            
            if end_sample > start_sample:
                filtered_segments.append(audio_data[start_sample:end_sample])
        
        if not filtered_segments:
            logger.warning("No audio segments extracted")
            return audio_path
        
        # Concatenate filtered audio
        filtered_audio = np.concatenate(filtered_segments)
        
        # Save filtered audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, filtered_audio,
