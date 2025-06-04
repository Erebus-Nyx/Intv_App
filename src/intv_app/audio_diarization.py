import logging
from pyannote.audio import Pipeline
from typing import List, Dict, Optional
import os
import sys
from pathlib import Path

# Dynamically import voice_activity_detection to avoid import errors
vad_path = Path(__file__).parent / "audio_vad.py"
if str(vad_path.parent) not in sys.path:
    sys.path.insert(0, str(vad_path.parent))
import importlib.util
vad_spec = importlib.util.spec_from_file_location("audio_vad", str(vad_path))
audio_vad = importlib.util.module_from_spec(vad_spec)
vad_spec.loader.exec_module(audio_vad)
voice_activity_detection = audio_vad.voice_activity_detection

def diarize_audio(
    audio_path: str,
    pipeline_token: Optional[str] = None,
    use_vad: bool = True
) -> List[Dict]:
    """
    Perform speaker diarization on an audio file using pyannote.audio.
    Optionally restrict diarization to VAD-detected speech regions.
    Returns a list of segments with start, end, and speaker label.
    """
    if pipeline_token is None:
        pipeline_token = os.environ.get("PYANNOTE_TOKEN")
    if not pipeline_token:
        raise ValueError("pyannote pipeline token required. Set PYANNOTE_TOKEN env variable or pass as argument.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=pipeline_token)
    vad_segments = None
    if use_vad:
        vad_segments = voice_activity_detection(audio_path)
        logging.info(f"VAD detected {len(vad_segments)} speech segments. Restricting diarization to these regions.")
    diarization = pipeline(audio_path)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # If VAD is enabled, only include segments that overlap with VAD speech
        if vad_segments:
            overlaps = any(
                (turn.start < vad_end and turn.end > vad_start)
                for vad_start, vad_end in vad_segments
            )
            if not overlaps:
                continue
        results.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    logging.info(f"Diarized {audio_path} with {len(results)} segments (after VAD filter).")
    return results
