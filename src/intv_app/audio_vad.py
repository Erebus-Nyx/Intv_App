import logging
import numpy as np
import soundfile as sf
from typing import List, Tuple


def voice_activity_detection(
    audio_path: str,
    frame_duration: float = 0.03,
    energy_threshold: float = 0.01,
    min_speech_duration: float = 0.3
) -> List[Tuple[float, float]]:
    """
    Simple energy-based Voice Activity Detection (VAD).
    Returns a list of (start, end) times for detected speech segments.
    """
    audio, sr = sf.read(audio_path)
    frame_size = int(sr * frame_duration)
    num_frames = int(np.ceil(len(audio) / frame_size))
    speech_segments = []
    in_speech = False
    seg_start = 0
    for i in range(num_frames):
        start = i * frame_size
        end = min((i + 1) * frame_size, len(audio))
        frame = audio[start:end]
        energy = np.mean(frame ** 2)
        t_start = start / sr
        t_end = end / sr
        if energy > energy_threshold:
            if not in_speech:
                seg_start = t_start
                in_speech = True
        else:
            if in_speech:
                if t_end - seg_start >= min_speech_duration:
                    speech_segments.append((seg_start, t_end))
                in_speech = False
    if in_speech and (t_end - seg_start >= min_speech_duration):
        speech_segments.append((seg_start, t_end))
    logging.info(f"VAD found {len(speech_segments)} speech segments in {audio_path}.")
    return speech_segments
