import logging
from faster_whisper import WhisperModel
from typing import Optional, List, Dict
import os
import sys
import requests
from pathlib import Path

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

def transcribe_audio_fastwhisper(
    audio_path: str,
    model_size: str = "large-v3-turbo",  # Default to turbo model
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,  # Enable VAD by default for noise robustness
    vad_parameters: Optional[Dict] = None,
    min_segment_duration: float = 0.5,
    min_confidence: float = -2.0,
    min_no_speech_prob: float = 0.5
) -> List[Dict]:
    """
    Transcribe audio using faster-whisper turbo model (large-v3-turbo by default).
    Filters segments by confidence and no_speech_prob, merges short/low-confidence segments.
    Returns a list of segments with text, start, end, and confidence.
    """
    # Always use English
    language = "en"
    # Ensure model is present
    model_dir = download_model_if_needed(model_size)
    model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=model_dir)
    segments, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters or {}
    )
    results = []
    for seg in segments:
        # Filter out low-confidence or no-speech segments
        if seg.no_speech_prob > min_no_speech_prob or seg.avg_logprob < min_confidence:
            continue
        # Merge very short segments with previous if possible
        if results and (seg.start - results[-1]['end'] < min_segment_duration):
            results[-1]['end'] = seg.end
            results[-1]['text'] += ' ' + seg.text
            continue
        results.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text,
            'avg_logprob': seg.avg_logprob,
            'no_speech_prob': seg.no_speech_prob,
            'temperature': seg.temperature
        })
    logging.info(f"Transcribed {audio_path} with {len(results)} segments (turbo model: {model_size}).")
    return results
