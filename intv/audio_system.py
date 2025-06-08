"""
Audio system capabilities and model selection for INTV
Provides hardware-optimized model selection for audio processing components
"""

import logging
from typing import Dict, Optional, Tuple
import os
import platform

logger = logging.getLogger(__name__)

# Try to import hardware detection dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class AudioSystemCapabilities:
    """Detect system capabilities for automatic audio model selection"""
    
    @staticmethod
    def detect_system_type() -> str:
        """Detect system hardware tier for audio processing"""
        try:
            machine = platform.machine().lower()
            
            # Raspberry Pi detection
            if 'arm' in machine or 'aarch64' in machine:
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read().lower()
                        if 'raspberry pi' in model:
                            return 'raspberry_pi'
                except:
                    pass
                return 'arm_low_end'
            
            # Get system specs
            cpu_count = 4  # Default assumption
            memory_gb = 4  # Default assumption
            
            if HAS_PSUTIL:
                cpu_count = psutil.cpu_count()
                memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Check for GPU (multiple methods)
            has_cuda = False
            gpu_memory_gb = 0
            
            # Method 1: Check with torch if available
            if HAS_TORCH:
                has_cuda = torch.cuda.is_available()
                if has_cuda:
                    try:
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    except:
                        gpu_memory_gb = 8  # Conservative estimate
            
            # Method 2: Check nvidia-smi if torch not available
            if not has_cuda:
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_memory_mb = int(result.stdout.strip())
                        gpu_memory_gb = gpu_memory_mb / 1024
                        has_cuda = True
                except:
                    pass
            
            # Enhanced classification logic for audio processing
            if has_cuda:
                if cpu_count >= 16 and memory_gb >= 32 and gpu_memory_gb >= 12:
                    return 'gpu_high'
                elif cpu_count >= 8 and memory_gb >= 16 and gpu_memory_gb >= 8:
                    return 'gpu_medium'
                elif gpu_memory_gb >= 4:
                    return 'gpu_low'
            
            # CPU-only classification
            if cpu_count >= 16 and memory_gb >= 32:
                return 'cpu_high'
            elif cpu_count >= 8 and memory_gb >= 16:
                return 'cpu_medium'
            elif cpu_count >= 4 and memory_gb >= 8:
                return 'cpu_low'
            else:
                return 'cpu_minimal'
                
        except Exception as e:
            logger.warning(f"System detection failed: {e}")
            return 'cpu_low'  # Safe default
    
    @staticmethod
    def get_optimal_audio_models(system_type: str, config: Optional[Dict] = None) -> Dict[str, str]:
        """Get optimal audio models based on system capabilities"""
        
        # VAD model selection (pyannote/segmentation-3.0 is lightweight enough for all systems)
        vad_models = {
            'raspberry_pi': 'pyannote/segmentation-3.0',
            'arm_low_end': 'pyannote/segmentation-3.0',
            'cpu_minimal': 'pyannote/segmentation-3.0',
            'cpu_low': 'pyannote/segmentation-3.0',
            'cpu_medium': 'pyannote/segmentation-3.0',
            'cpu_high': 'pyannote/segmentation-3.0',
            'gpu_low': 'pyannote/segmentation-3.0',
            'gpu_medium': 'pyannote/segmentation-3.0',
            'gpu_high': 'pyannote/segmentation-3.0'
        }
        
        # Diarization model selection (all use the same model, but processing differs)
        diarization_models = {
            'raspberry_pi': 'pyannote/speaker-diarization-3.1',
            'arm_low_end': 'pyannote/speaker-diarization-3.1',
            'cpu_minimal': 'pyannote/speaker-diarization-3.1',
            'cpu_low': 'pyannote/speaker-diarization-3.1',
            'cpu_medium': 'pyannote/speaker-diarization-3.1',
            'cpu_high': 'pyannote/speaker-diarization-3.1',
            'gpu_low': 'pyannote/speaker-diarization-3.1',
            'gpu_medium': 'pyannote/speaker-diarization-3.1',
            'gpu_high': 'pyannote/speaker-diarization-3.1'
        }
        
        # Whisper model selection based on hardware capabilities
        whisper_models = {
            'raspberry_pi': 'faster-whisper/tiny',
            'arm_low_end': 'faster-whisper/tiny',
            'cpu_minimal': 'faster-whisper/tiny',
            'cpu_low': 'faster-whisper/base',
            'cpu_medium': 'faster-whisper/small',
            'cpu_high': 'faster-whisper/medium',
            'gpu_low': 'faster-whisper/base',
            'gpu_medium': 'faster-whisper/medium',
            'gpu_high': 'faster-whisper/large-v3'
        }
        
        # Override with config if provided
        if config:
            audio_config = config.get('audio', {})
            models_config = audio_config.get('models', {})
            
            if 'vad_models' in models_config:
                vad_models.update(models_config['vad_models'])
            if 'diarization_models' in models_config:
                diarization_models.update(models_config['diarization_models'])
            if 'whisper_models' in models_config:
                whisper_models.update(models_config['whisper_models'])
        
        return {
            'vad_model': vad_models.get(system_type, 'pyannote/segmentation-3.0'),
            'diarization_model': diarization_models.get(system_type, 'pyannote/speaker-diarization-3.1'),
            'whisper_model': whisper_models.get(system_type, 'faster-whisper/base')
        }
    
    @staticmethod
    def get_processing_config(system_type: str) -> Dict[str, any]:
        """Get processing configuration optimized for system type"""
        
        configs = {
            'raspberry_pi': {
                'batch_size': 1,
                'use_gpu': False,
                'max_concurrent_streams': 1,
                'chunk_size': 1024,
                'enable_optimization': True
            },
            'arm_low_end': {
                'batch_size': 1,
                'use_gpu': False,
                'max_concurrent_streams': 1,
                'chunk_size': 2048,
                'enable_optimization': True
            },
            'cpu_minimal': {
                'batch_size': 1,
                'use_gpu': False,
                'max_concurrent_streams': 1,
                'chunk_size': 2048,
                'enable_optimization': True
            },
            'cpu_low': {
                'batch_size': 2,
                'use_gpu': False,
                'max_concurrent_streams': 2,
                'chunk_size': 4096,
                'enable_optimization': True
            },
            'cpu_medium': {
                'batch_size': 4,
                'use_gpu': False,
                'max_concurrent_streams': 3,
                'chunk_size': 8192,
                'enable_optimization': False
            },
            'cpu_high': {
                'batch_size': 8,
                'use_gpu': False,
                'max_concurrent_streams': 4,
                'chunk_size': 16384,
                'enable_optimization': False
            },
            'gpu_low': {
                'batch_size': 4,
                'use_gpu': True,
                'max_concurrent_streams': 3,
                'chunk_size': 8192,
                'enable_optimization': False
            },
            'gpu_medium': {
                'batch_size': 8,
                'use_gpu': True,
                'max_concurrent_streams': 4,
                'chunk_size': 16384,
                'enable_optimization': False
            },
            'gpu_high': {
                'batch_size': 16,
                'use_gpu': True,
                'max_concurrent_streams': 6,
                'chunk_size': 32768,
                'enable_optimization': False
            }
        }
        
        return configs.get(system_type, configs['cpu_low'])
    
    @staticmethod
    def detect_and_configure(config: Optional[Dict] = None) -> Tuple[str, Dict[str, str], Dict[str, any]]:
        """
        Detect system capabilities and return optimal configuration
        
        Returns:
            Tuple of (system_type, optimal_models, processing_config)
        """
        system_type = AudioSystemCapabilities.detect_system_type()
        optimal_models = AudioSystemCapabilities.get_optimal_audio_models(system_type, config)
        processing_config = AudioSystemCapabilities.get_processing_config(system_type)
        
        logger.info(f"Audio system detected: {system_type}")
        logger.info(f"Optimal models: {optimal_models}")
        logger.info(f"Processing config: {processing_config}")
        
        return system_type, optimal_models, processing_config

def get_audio_model_recommendations(config: Optional[Dict] = None) -> Dict[str, str]:
    """
    Get audio model recommendations for the current system
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with recommended models for VAD, diarization, and transcription
    """
    _, optimal_models, _ = AudioSystemCapabilities.detect_and_configure(config)
    return optimal_models

def log_system_info():
    """Log detailed system information for audio processing"""
    try:
        system_type, models, processing = AudioSystemCapabilities.detect_and_configure()
        
        print(f"\n=== AUDIO SYSTEM INFORMATION ===")
        print(f"System Type: {system_type}")
        print(f"Platform: {platform.platform()}")
        print(f"Architecture: {platform.machine()}")
        
        if HAS_PSUTIL:
            print(f"CPU Cores: {psutil.cpu_count()}")
            print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if HAS_TORCH and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("GPU: Not available")
        
        print(f"\n=== RECOMMENDED AUDIO MODELS ===")
        print(f"VAD Model: {models['vad_model']}")
        print(f"Diarization Model: {models['diarization_model']}")
        print(f"Whisper Model: {models['whisper_model']}")
        
        print(f"\n=== PROCESSING CONFIGURATION ===")
        for key, value in processing.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error getting system info: {e}")

if __name__ == "__main__":
    log_system_info()
