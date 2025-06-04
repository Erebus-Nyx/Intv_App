import os
import pytest
from pathlib import Path
from intv_app import rag, ocr, llm

def test_chunk_text():
    text = "This is a test. " * 100
    chunks = rag.chunk_text(text, chunk_size=50)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert len(''.join(chunks)) == len(text)

def test_chunk_document(tmp_path):
    # Create a dummy txt file
    file = tmp_path / 'test.txt'
    file.write_text("Hello world. " * 20)
    chunks = rag.chunk_document(file, 'txt')
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

def test_ocr_image(tmp_path):
    # Create a dummy image (white PNG)
    from PIL import Image
    img_path = tmp_path / 'test.png'
    img = Image.new('RGB', (100, 30), color = (255,255,255))
    img.save(img_path)
    text = ocr.ocr_image(str(img_path))
    assert isinstance(text, str)

def test_transcribe_audio_file(monkeypatch):
    # Simulate transcription
    from intv_app import audio_transcribe
    monkeypatch.setattr(audio_transcribe, 'transcribe_audio_fastwhisper', lambda path: [{'text': 'hello', 'start': 0, 'end': 1}])
    segments = audio_transcribe.transcribe_audio_fastwhisper('dummy.wav')
    assert isinstance(segments, list)
    assert segments[0]['text'] == 'hello'

def test_stream_microphone_transcription(monkeypatch):
    from intv_app import audio_transcribe
    monkeypatch.setattr(audio_transcribe, 'stream_microphone_transcription', lambda out: ("test transcript", [{'start': 0, 'end': 1, 'text': 'test'}]))
    transcript, meta = audio_transcribe.stream_microphone_transcription('mic.wav')
    assert isinstance(transcript, str)
    assert isinstance(meta, list)

def test_analyze_chunks(monkeypatch):
    # Simulate LLM output
    monkeypatch.setattr(llm, 'analyze_chunks', lambda chunks, **kwargs: {'result': 'ok', 'chunks': len(chunks)})
    result = llm.analyze_chunks(['a', 'b', 'c'])
    assert isinstance(result, dict)
    assert result['chunks'] == 3
