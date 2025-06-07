"""
Voice Activity Detection (VAD) module for INTV
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

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
