import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
from intv import audio_utils

@pytest.mark.audio
@pytest.mark.unit
def test_load_audio_file_success():
    """Test successful audio file loading"""
    # Mock audio data
    mock_audio = np.array([0.1, 0.2, 0.3, 0.4])
    mock_sr = 16000
    
    with patch('librosa.load', return_value=(mock_audio, mock_sr)):
        audio, sr = audio_utils.load_audio_file("test.wav")
        assert np.array_equal(audio, mock_audio)
        assert sr == mock_sr

@pytest.mark.audio
@pytest.mark.unit
def test_load_audio_file_not_found():
    """Test audio file loading when file doesn't exist"""
    with patch('librosa.load', side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            audio_utils.load_audio_file("nonexistent.wav")

@pytest.mark.audio
@pytest.mark.unit
def test_normalize_audio():
    """Test audio normalization"""
    # Test audio with values outside [-1, 1] range
    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    normalized = audio_utils.normalize_audio(audio)
    
    # Check that values are within [-1, 1] range
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)
    
    # Check that max absolute value is 1.0 (assuming max normalization)
    assert np.max(np.abs(normalized)) == pytest.approx(1.0, rel=1e-6)

@pytest.mark.audio
@pytest.mark.unit
def test_resample_audio():
    """Test audio resampling"""
    # Mock audio data at 44100 Hz
    original_audio = np.random.randn(44100)  # 1 second of audio
    original_sr = 44100
    target_sr = 16000
    
    with patch('librosa.resample') as mock_resample:
        mock_resample.return_value = np.random.randn(16000)  # Resampled to 16kHz
        
        resampled = audio_utils.resample_audio(original_audio, original_sr, target_sr)
        
        mock_resample.assert_called_once_with(
            original_audio, orig_sr=original_sr, target_sr=target_sr
        )
        assert len(resampled) == 16000

@pytest.mark.audio
@pytest.mark.unit
def test_split_audio_chunks():
    """Test splitting audio into chunks"""
    # Create 5 seconds of audio at 16kHz
    audio = np.random.randn(80000)  # 5 * 16000 samples
    sr = 16000
    chunk_duration = 2.0  # 2 seconds
    
    chunks = audio_utils.split_audio_chunks(audio, sr, chunk_duration)
    
    # Should have 3 chunks: 2s, 2s, 1s
    assert len(chunks) == 3
    assert len(chunks[0]) == 32000  # 2 * 16000
    assert len(chunks[1]) == 32000  # 2 * 16000
    assert len(chunks[2]) == 16000  # 1 * 16000

@pytest.mark.audio
@pytest.mark.unit
def test_detect_silence():
    """Test silence detection"""
    # Create audio with silence (low amplitude) and speech (high amplitude)
    silence = np.random.randn(1600) * 0.01  # Very quiet
    speech = np.random.randn(1600) * 0.5    # Louder
    audio = np.concatenate([silence, speech, silence])
    
    silence_mask = audio_utils.detect_silence(audio, threshold=0.1)
    
    # First and last parts should be detected as silence
    assert np.mean(silence_mask[:1600]) > 0.8  # Mostly silence
    assert np.mean(silence_mask[1600:3200]) < 0.2  # Mostly not silence
    assert np.mean(silence_mask[3200:]) > 0.8  # Mostly silence

@pytest.mark.audio
@pytest.mark.slow
def test_save_audio_file():
    """Test saving audio to file"""
    audio = np.random.randn(16000)  # 1 second of audio
    sr = 16000
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        with patch('soundfile.write') as mock_write:
            audio_utils.save_audio_file(audio, sr, tmp_path)
            mock_write.assert_called_once_with(tmp_path, audio, sr)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@pytest.mark.audio
@pytest.mark.unit
def test_compute_audio_features():
    """Test audio feature computation"""
    audio = np.random.randn(16000)  # 1 second of audio
    sr = 16000
    
    with patch('librosa.feature.mfcc') as mock_mfcc:
        with patch('librosa.feature.spectral_centroid') as mock_centroid:
            mock_mfcc.return_value = np.random.randn(13, 32)  # 13 MFCC coefficients
            mock_centroid.return_value = np.random.randn(1, 32)  # Spectral centroid
            
            features = audio_utils.compute_audio_features(audio, sr)
            
            mock_mfcc.assert_called_once()
            mock_centroid.assert_called_once()
            assert 'mfcc' in features
            assert 'spectral_centroid' in features
