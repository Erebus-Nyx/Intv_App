"""
INTV LLM System - Comprehensive Large Language Model Support

This module provides a unified interface for LLM operations with support for:
- Embedded llama.cpp support for local model inference
- System-driven default models downloaded on install  
- External API provider support (OpenAI compatible, KoboldCpp, Ollama)
- Hybrid model processing with JSON-driven configuration
- General summary without policy prompt
- Policy-adherent summary generation
- Pre-defined output format compliance
- Automatic model selection based on system capabilities
- Progressive model downloading with status indicators
"""

import os
import sys
import platform
import psutil
import logging
import requests
import json
import time
import subprocess
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
import tempfile
import shutil

# Check for optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# GPU detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# llama.cpp support
try:
    import llama_cpp
    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False

# Transformers support
try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Setup logging
logger = logging.getLogger(__name__)


class SystemCapabilities:
    """Detect system capabilities for automatic LLM model selection"""
    
    @staticmethod
    def detect_system_type():
        """Detect if running on Raspberry Pi, low-end, or high-end system"""
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
        
        # x86/x64 systems
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
                    gpu_memory_gb = 8  # Assume reasonable default
        
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
        
        # Method 3: Check for NVIDIA in lspci
        if not has_cuda:
            try:
                import subprocess
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'NVIDIA' in result.stdout:
                    has_cuda = True
                    gpu_memory_gb = 8  # Conservative estimate
            except:
                pass
        
        # Enhanced classification logic
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
    
    @staticmethod
    def get_default_llm_model(system_type: str) -> str:
        """Get default LLM model based on system capabilities"""
        models = {
            'raspberry_pi': 'hf.co/microsoft/DialoGPT-small',
            'arm_low_end': 'hf.co/microsoft/DialoGPT-small',
            'cpu_minimal': 'hf.co/microsoft/DialoGPT-medium',
            'cpu_low': 'hf.co/microsoft/Phi-3-mini-4k-instruct',
            'cpu_medium': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q4_K_M.gguf',
            'cpu_high': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q4_K_M.gguf',
            'gpu_low': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q4_K_M.gguf',
            'gpu_medium': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q5_K_M.gguf',
            'gpu_high': 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q5_K_M.gguf'
        }
        return models.get(system_type, 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q4_K_M.gguf')

class LLMModelDownloader:
    """Handle LLM model downloading with progress indicators"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def parse_model_string(self, model_string: str) -> tuple:
        """Parse model string to extract repo_id and filename if specified"""
        # Check if it's a local file/directory path
        if self.is_local_path(model_string):
            return None, model_string  # Return None for repo_id to indicate local file
            
        if model_string.startswith('hf.co/'):
            model_string = model_string[6:]  # Remove 'hf.co/' prefix
            
        if ':' in model_string:
            repo_id, filename = model_string.split(':', 1)
            return repo_id, filename
        else:
            return model_string, None
    
    def is_local_path(self, model_string: str) -> bool:
        """Check if model_string refers to a local file or directory"""
        # Skip HuggingFace URLs
        if model_string.startswith('hf.co/'):
            return False
            
        # Check for absolute paths
        if model_string.startswith('/'):
            return True
        
        # Check for relative paths starting with ./ or ../
        if model_string.startswith('./') or model_string.startswith('../'):
            return True
            
        # Check for common file extensions without repo format (but not with repo format)
        if ':' not in model_string:  # Ensure it's not repo:file format
            file_extensions = ['.gguf', '.safetensors', '.bin', '.pt', '.pth', '.onnx']
            if any(model_string.endswith(ext) for ext in file_extensions):
                return True
            
        # Check if it's a directory that exists locally
        path = Path(model_string)
        if path.exists():
            return True
            
        # Check if it's in the models directory
        models_path = self.model_dir / model_string
        if models_path.exists():
            return True
            
        return False
    
    def download_with_progress(self, url: str, dest_path: Path, desc: str = "Downloading"):
        """Download file with progress bar"""
        try:
            import requests
            from tqdm import tqdm
        except ImportError:
            # Fallback without progress bar
            response = requests.get(url)
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            return
            
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def is_model_downloaded(self, model_string: str) -> bool:
        """Check if model is already downloaded"""
        repo_id, filename = self.parse_model_string(model_string)
        
        # Handle local files
        if repo_id is None:  # This means it's a local file
            # filename contains the full path for local files
            local_path = Path(filename)
            
            # Check absolute path
            if local_path.is_absolute() and local_path.exists():
                return True
                
            # Check relative to current directory
            if local_path.exists():
                return True
                
            # Check relative to models directory
            models_path = self.model_dir / filename
            if models_path.exists():
                return True
                
            # Check if it's just a filename in models directory
            if not local_path.is_absolute() and '/' not in filename:
                direct_path = self.model_dir / filename
                if direct_path.exists():
                    return True
                    
            return False
        
        # Handle HuggingFace repo models (existing logic)
        # For full models (no specific filename)
        if not filename:
            # Check custom directory structure
            model_path = self.model_dir / repo_id.replace('/', '--')
            if model_path.exists() and any(model_path.iterdir()):
                return True
            
            # Check HuggingFace cache structure (used by transformers)
            hf_cache_path = self.model_dir / f"models--{repo_id.replace('/', '--')}"
            if hf_cache_path.exists() and any(hf_cache_path.iterdir()):
                return True
                
            return False
        
        # For specific files from HuggingFace (GGUF, safetensors, etc.)
        # Check multiple possible locations
        possible_paths = [
            # Standard HF cache structure
            self.model_dir / repo_id.replace('/', '--') / filename,
            # HF cache blobs structure
            self.model_dir / f"models--{repo_id.replace('/', '--')}" / "blobs" / filename,
            # Direct in models folder (common for GGUF)
            self.model_dir / filename,
            # In repo folder directly
            self.model_dir / repo_id.replace('/', '--') / filename,
            # Nested cache structure
            self.model_dir / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}" / "blobs" / filename
        ]
        
        # Also check for the actual file hash names in HF cache
        if filename.endswith('.gguf') or filename.endswith('.safetensors'):
            # Check HF cache structure with hash filenames
            cache_dir = self.model_dir / f"models--{repo_id.replace('/', '--')}"
            if cache_dir.exists():
                blobs_dir = cache_dir / "blobs"
                if blobs_dir.exists():
                    # Any file in blobs dir indicates model is downloaded
                    if any(blobs_dir.iterdir()):
                        return True
        
        return any(path.exists() for path in possible_paths)
    
    def download_model(self, model_string: str, force: bool = False) -> Path:
        """Download model from HuggingFace with progress indicator"""
        repo_id, filename = self.parse_model_string(model_string)
        
        # Handle local files - don't download, just return path
        if repo_id is None:  # Local file
            local_path = Path(filename)
            
            # Check absolute path
            if local_path.is_absolute() and local_path.exists():
                print(f"📁 Using local model: {local_path}")
                return local_path
                
            # Check relative to current directory
            if local_path.exists():
                print(f"📁 Using local model: {local_path.resolve()}")
                return local_path.resolve()
                
            # Check relative to models directory
            models_path = self.model_dir / filename
            if models_path.exists():
                print(f"📁 Using local model from models dir: {models_path}")
                return models_path
                
            # Check if it's just a filename in models directory
            if not local_path.is_absolute() and '/' not in filename:
                direct_path = self.model_dir / filename
                if direct_path.exists():
                    print(f"📁 Using local model: {direct_path}")
                    return direct_path
                    
            # Local file not found
            raise FileNotFoundError(f"Local model file not found: {filename}")
        
        # Handle HuggingFace models
        if not force and self.is_model_downloaded(model_string):
            print(f"✅ Model already downloaded: {model_string}")
            return self.model_dir / repo_id.replace('/', '--')
        
        local_dir = self.model_dir / repo_id.replace('/', '--')
        local_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading model: {model_string}")
        
        try:
            if filename:
                print(f"📁 Downloading specific file: {filename} from {repo_id}")
                try:
                    from huggingface_hub import hf_hub_download
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(self.model_dir),
                        local_dir=str(local_dir)
                    )
                    print(f"✅ File downloaded to: {file_path}")
                    return Path(file_path).parent
                except ImportError:
                    logger.error("huggingface_hub not available for file download")
                    print("❌ huggingface_hub not available - cannot download GGUF/specific files")
                    raise
            elif HAS_TRANSFORMERS:
                print(f"🤗 Using transformers for full model: {repo_id}")
                # Download full model
                AutoTokenizer.from_pretrained(repo_id, cache_dir=str(local_dir))
                AutoModelForCausalLM.from_pretrained(repo_id, cache_dir=str(local_dir))
            else:
                raise ImportError("No suitable download method available")
                
            print(f"✅ Model downloaded successfully: {model_string}")
            return local_dir
            
        except Exception as e:
            logger.error(f"Failed to download model {model_string}: {e}")
            print(f"❌ Failed to download model {model_string}: {e}")
            raise

class EmbeddedLLM:
    """Embedded LLM using local models and llama.cpp"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.llama_model = None
        self.model_context_size = None  # Will be detected automatically
        self.downloader = LLMModelDownloader(config.get('model_dir', 'models'))
        self._initialize_model()
    
    def _get_configured_context_size(self) -> int:
        """Get the configured context size from config, with fallback to auto-detection"""
        # Check embedded-specific config first (highest priority)
        llm_config = self.config.get('llm', {})
        embedded_config = llm_config.get('embedded', {})
        embedded_context_size = embedded_config.get('context_size')
        
        if embedded_context_size and embedded_context_size != 'auto':
            try:
                context_size = int(embedded_context_size)
                if context_size > 0:  # Ensure positive value
                    logger.info(f"Using embedded-specific context size: {context_size}")
                    return context_size
                else:
                    logger.warning(f"Invalid embedded context_size value '{embedded_context_size}' (must be positive), falling back to global")
            except (ValueError, TypeError):
                logger.warning(f"Invalid embedded context_size value '{embedded_context_size}', falling back to global")
        
        # Check global LLM config second
        configured_size = llm_config.get('context_size')
        
        if configured_size and configured_size != 'auto':
            try:
                context_size = int(configured_size)
                if context_size > 0:  # Ensure positive value
                    logger.info(f"Using configured context size: {context_size}")
                    return context_size
                else:
                    logger.warning(f"Invalid context_size value '{configured_size}' (must be positive), falling back to auto")
            except (ValueError, TypeError):
                logger.warning(f"Invalid context_size value '{configured_size}', falling back to auto")
        
        # Auto-detect based on system capabilities
        system_type = SystemCapabilities.detect_system_type()
        
        if system_type in ['gpu_high_end']:
            context_size = 8192  # High-end GPU can handle larger context
        elif system_type in ['gpu_mid_range', 'cpu_high_end']:
            context_size = 4096  # Standard context for good hardware
        elif system_type in ['raspberry_pi', 'arm_low_end', 'cpu_low_end']:
            context_size = 2048  # Conservative for low-end systems
        else:
            context_size = 4096  # Default fallback
        
        logger.info(f"Auto-detected context size for {system_type}: {context_size}")
        return context_size

    def get_context_window_size(self) -> int:
        """Get the context window size of the loaded model"""
        if self.model_context_size is not None:
            return self.model_context_size
        
        # Try to detect context window from different model backends
        if self.llama_model is not None:
            try:
                # llama.cpp models have context size in their configuration
                context_size = getattr(self.llama_model, 'n_ctx', None)
                if context_size and callable(context_size):
                    context_size = context_size()
                if context_size:
                    self.model_context_size = context_size
                    return context_size
            except Exception as e:
                logger.debug(f"Could not get context size from llama.cpp model: {e}")
        
        if self.model is not None and hasattr(self.model, 'config'):
            try:
                # Transformers models store context size in config
                config = self.model.config
                # Common attribute names for context window
                for attr in ['max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_len']:
                    if hasattr(config, attr):
                        context_size = getattr(config, attr)
                        if context_size and context_size > 0:
                            self.model_context_size = context_size
                            return context_size
            except Exception as e:
                logger.debug(f"Could not get context size from transformers model: {e}")
        
        # Default fallbacks based on common model types
        if self.llama_model is not None:
            # GGUF models typically have 4k, 8k, or 32k context
            self.model_context_size = 4096
            logger.info("Using default context size 4096 for GGUF model")
            return 4096
        elif self.model is not None:
            # Transformers models often have 2k or 4k context
            self.model_context_size = 2048
            logger.info("Using default context size 2048 for transformers model")
            return 2048
        else:
            # Final fallback
            self.model_context_size = 1024
            logger.warning("No model loaded, using minimal context size 1024")
            return 1024

    def calculate_auto_max_tokens(self, prompt: str, reserve_tokens: int = 100) -> int:
        """Calculate max_tokens automatically based on context window and prompt length"""
        context_size = self.get_context_window_size()
        
        # Estimate prompt token count (rough approximation: 1 token ≈ 4 characters)
        estimated_prompt_tokens = len(prompt) // 4
        
        # Calculate available tokens for generation
        available_tokens = context_size - estimated_prompt_tokens - reserve_tokens
        
        # Ensure we have a reasonable minimum
        max_tokens = max(available_tokens, 50)
        
        # Cap at reasonable maximum for performance
        max_tokens = min(max_tokens, 2048)
        
        logger.debug(f"Auto max_tokens calculation: context={context_size}, prompt_est={estimated_prompt_tokens}, available={available_tokens}, final={max_tokens}")
        return max_tokens

    def _initialize_model(self):
        """Initialize the LLM model"""
        llm_config = self.config.get('llm', {}).get('embedded', {})
        model_string = llm_config.get('model', 'auto')
        
        if model_string == 'auto':
            system_type = SystemCapabilities.detect_system_type()
            model_string = SystemCapabilities.get_default_llm_model(system_type)
            print(f"🎯 Auto-selected LLM model for {system_type}: {model_string}")
        
        # Check if it's a local file first
        repo_id, filename = self.downloader.parse_model_string(model_string)
        
        if repo_id is None:  # Local file
            # Handle local models - don't download, just load
            try:
                model_path = self.downloader.download_model(model_string)  # This will just validate and return path
                
                # Try llama.cpp first for GGUF files
                if str(model_path).endswith('.gguf') and HAS_LLAMACPP:
                    try:
                        n_gpu_layers = -1 if HAS_TORCH and torch.cuda.is_available() else 0
                        # Get configured context size or auto-detect
                        n_ctx = self._get_configured_context_size()
                        self.llama_model = llama_cpp.Llama(
                            model_path=str(model_path),
                            n_gpu_layers=n_gpu_layers,
                            n_ctx=n_ctx,
                            verbose=False
                        )
                        print(f"✅ Loaded local GGUF model with llama.cpp: {model_path}")
                        print(f"   Context size: {n_ctx} tokens")
                    except Exception as e:
                        logger.error(f"Failed to load GGUF model with llama.cpp {model_path}: {e}")
                        print(f"❌ Failed to load GGUF model with llama.cpp {model_path}: {e}")
                        self.llama_model = None
                
                # Try transformers for other formats
                elif HAS_TRANSFORMERS:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                        self.model = AutoModelForCausalLM.from_pretrained(str(model_path))
                        print(f"✅ Loaded local transformers model: {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load local transformers model {model_path}: {e}")
                        print(f"❌ Failed to load local transformers model {model_path}: {e}")
                        self.model = None
                        self.tokenizer = None
                else:
                    logger.warning("Neither llama.cpp nor transformers available for local model loading")
                    self.model = None
                    self.tokenizer = None
                    self.llama_model = None
                    
            except FileNotFoundError as e:
                logger.error(f"Local model file not found: {e}")
                print(f"❌ Local model file not found: {e}")
                self.model = None
                self.tokenizer = None
                self.llama_model = None
            except Exception as e:
                logger.error(f"Failed to initialize local model: {e}")
                self.model = None
                self.tokenizer = None
                self.llama_model = None
        else:
            # Handle HuggingFace models (existing logic)
            try:
                model_path = self.downloader.download_model(model_string)
                
                # For GGUF files from HuggingFace, try llama.cpp
                if filename and filename.endswith('.gguf') and HAS_LLAMACPP:
                    try:
                        # Find the actual GGUF file
                        gguf_file = None
                        for possible_path in [
                            model_path / filename,
                            self.downloader.model_dir / filename,
                            self.downloader.model_dir / f"models--{repo_id.replace('/', '--')}" / "blobs" / filename
                        ]:
                            if possible_path.exists():
                                gguf_file = possible_path
                                break
                        
                        if gguf_file:
                            n_gpu_layers = -1 if HAS_TORCH and torch.cuda.is_available() else 0
                            # Get configured context size or auto-detect
                            n_ctx = self._get_configured_context_size()
                            self.llama_model = llama_cpp.Llama(
                                model_path=str(gguf_file),
                                n_gpu_layers=n_gpu_layers,
                                n_ctx=n_ctx,
                                verbose=False
                            )
                            print(f"✅ Loaded GGUF model with llama.cpp from HF: {gguf_file}")
                            print(f"   Context size: {n_ctx} tokens")
                        else:
                            raise FileNotFoundError(f"GGUF file not found: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to load GGUF model with llama.cpp: {e}")
                        print(f"❌ Failed to load GGUF model with llama.cpp: {e}")
                        self.llama_model = None
                
                # Load model with transformers
                elif HAS_TRANSFORMERS:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(self.downloader.model_dir))
                        self.model = AutoModelForCausalLM.from_pretrained(repo_id, cache_dir=str(self.downloader.model_dir))
                        print(f"✅ Loaded transformers model from repo: {repo_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load from repo {repo_id}, trying local path: {e}")
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                            self.model = AutoModelForCausalLM.from_pretrained(str(model_path))
                            print(f"✅ Loaded transformers model from path: {model_path}")
                        except Exception as e2:
                            logger.error(f"Failed to load transformers model: {e2}")
                            self.model = None
                            self.tokenizer = None
                else:
                    logger.warning("Neither llama.cpp nor transformers available, using fallback")
                    self.model = None
                    self.tokenizer = None
                    self.llama_model = None
                    
            except Exception as e:
                logger.error(f"Failed to initialize LLM model: {e}")
                self.model = None
                self.tokenizer = None
                self.llama_model = None
    
    def generate_text(self, prompt: str, max_tokens: Union[int, str] = 200, temperature: float = 0.7, 
                     top_p: float = 0.9, **kwargs) -> str:
        """Generate text using the loaded model"""
        
        # Handle "auto" max_tokens
        if max_tokens == "auto":
            max_tokens = self.calculate_auto_max_tokens(prompt)
            logger.info(f"Using auto-calculated max_tokens: {max_tokens}")
        elif isinstance(max_tokens, str):
            try:
                max_tokens = int(max_tokens)
            except ValueError:
                logger.warning(f"Invalid max_tokens value '{max_tokens}', using default 200")
                max_tokens = 200
        
        if self.llama_model is not None:
            # Use llama.cpp with optimized settings for speed
            try:
                # Only limit tokens if not using auto mode or for very large outputs
                actual_max_tokens = max_tokens
                if actual_max_tokens > 1000:
                    # For large token counts, optimize generation
                    stop_tokens = ["\n\n", "END", "STOP", "<|im_end|>", "</s>"]
                else:
                    # For smaller token counts, keep original optimization for speed
                    actual_max_tokens = min(actual_max_tokens, 100)
                    stop_tokens = ["\n\n", "END", "STOP"]
                
                response = self.llama_model(
                    prompt,
                    max_tokens=actual_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    echo=False,
                    stop=stop_tokens,
                    **kwargs
                )
                return response['choices'][0]['text'].strip()
            except Exception as e:
                logger.error(f"Error generating text with llama.cpp: {e}")
                return f"Error: {str(e)}"

        elif self.model is not None and self.tokenizer is not None:
            # Use transformers with dynamic generation settings
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
                
                # Move to GPU if available
                if HAS_TORCH and torch.cuda.is_available():
                    inputs = inputs.cuda()
                    self.model = self.model.cuda()
                
                # Adjust generation parameters based on max_tokens
                actual_max_tokens = max_tokens
                num_beams = 1  # Default to fast generation
                early_stopping = True
                
                if actual_max_tokens > 1000:
                    # For large token counts, use more sophisticated generation
                    num_beams = 2
                    early_stopping = False
                else:
                    # For smaller token counts, optimize for speed
                    actual_max_tokens = min(actual_max_tokens, 100)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=actual_max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_beams=num_beams,
                        early_stopping=early_stopping,
                        **kwargs
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the original prompt from the response
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                return response
            except Exception as e:
                logger.error(f"Error generating text with transformers: {e}")
                return f"Error: {str(e)}"
        
        else:
            # Fallback: no model available
            logger.warning("No LLM model available, using fallback response")
            return f"I would analyze the following prompt: {prompt[:100]}..."
    
    def analyze_chunk(self, chunk: str, context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze a text chunk with optional context and system prompt"""
        if system_prompt:
            prompt = f"{system_prompt}\n\nContext: {context}\n\nText to analyze: {chunk}\n\nAnalysis:"
        else:
            prompt = f"Analyze the following text:\n\n{chunk}\n\nAnalysis:"
        
        try:
            output = self.generate_text(prompt)
            return {
                'output': output,
                'provider': 'embedded',
                'model': 'local',
                'success': True
            }
        except Exception as e:
            return {
                'output': f"Embedded LLM error: {str(e)}",
                'provider': 'embedded',
                'error': True,
                'success': False
            }

class ExternalAPILLM:
    """External LLM using API providers (OpenAI compatible, KoboldCpp, Ollama)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.full_config = config  # Store full config for global access
        self.config = config.get('llm', {}).get('external', {})
        self.provider = self.config.get('provider', 'koboldcpp')
        self.api_base = self.config.get('api_base', 'http://localhost')
        self.api_key = self.config.get('api_key', '')
        self.api_port = self.config.get('api_port', 5001)
        self.model = self.config.get('model', 'auto')
        self.timeout = self.config.get('timeout', 30)
        self.default_context_size = 4096  # Default assumption for external APIs
        
    def get_context_window_size(self) -> int:
        """Get the context window size for external API models"""
        # For external APIs, we usually can't query the context size directly
        # Use configuration or reasonable defaults based on provider
        
        # Check external-specific config first (highest priority)
        external_context_size = self.config.get('context_size')
        if external_context_size and external_context_size != 'auto':
            try:
                return int(external_context_size)
            except ValueError:
                pass
        
        # Check global LLM config second
        llm_config = self.full_config.get('llm', {})
        configured_size = llm_config.get('context_size')
        
        if configured_size and configured_size != 'auto':
            try:
                return int(configured_size)
            except ValueError:
                pass
        
        # Provider-specific defaults
        if self.provider == 'openai':
            if 'gpt-4' in self.model.lower():
                return 8192
            elif 'gpt-3.5' in self.model.lower():
                return 4096
            else:
                return 4096
        elif self.provider == 'ollama':
            return 2048  # Conservative default for Ollama
        else:  # koboldcpp and others
            return self.default_context_size
    
    def calculate_auto_max_tokens(self, prompt: str, reserve_tokens: int = 100) -> int:
        """Calculate max_tokens automatically based on context window and prompt length"""
        context_size = self.get_context_window_size()
        
        # Estimate prompt token count (rough approximation: 1 token ≈ 4 characters)
        estimated_prompt_tokens = len(prompt) // 4
        
        # Calculate available tokens for generation
        available_tokens = context_size - estimated_prompt_tokens - reserve_tokens
        
        # Ensure we have a reasonable minimum
        max_tokens = max(available_tokens, 50)
        
        # Cap at reasonable maximum for performance
        max_tokens = min(max_tokens, 2048)
        
        logger.debug(f"External API auto max_tokens calculation: context={context_size}, prompt_est={estimated_prompt_tokens}, available={available_tokens}, final={max_tokens}")
        return max_tokens
    
    def analyze_chunk(self, chunk: str, context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze a text chunk using external API"""
        if self.provider == 'koboldcpp':
            return self._analyze_with_koboldcpp(chunk, context, system_prompt)
        elif self.provider == 'openai':
            return self._analyze_with_openai(chunk, context, system_prompt)
        elif self.provider == 'ollama':
            return self._analyze_with_ollama(chunk, context, system_prompt)
        else:
            return {
                'output': f"Unsupported provider: {self.provider}",
                'provider': self.provider,
                'error': True,
                'success': False
            }
    
    def _analyze_with_koboldcpp(self, chunk: str, context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze chunk using KoboldCpp API"""
        url = f"{self.api_base}:{self.api_port}/api/v1/generate"
        
        if system_prompt:
            prompt = f"{system_prompt}\n\nContext: {context}\n\nText to analyze: {chunk}\n\nAnalysis:"
        else:
            prompt = chunk
        
        # Use auto max_tokens calculation
        max_tokens = self.calculate_auto_max_tokens(prompt)
        if max_tokens <= 200:
            max_tokens = min(max_tokens, 200)  # Keep original limit for small contexts
        
        payload = {
            "prompt": prompt,
            "max_length": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data and data['results']:
                output = data['results'][0].get('text', '')
            else:
                output = data.get('text', str(data))
                
            return {
                'output': output,
                'provider': 'koboldcpp',
                'model': self.model,
                'success': True
            }
        except Exception as e:
            return {
                'output': f"KoboldCpp error: {str(e)}",
                'provider': 'koboldcpp',
                'error': True,
                'success': False
            }
    
    def _analyze_with_openai(self, chunk: str, context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze chunk using OpenAI API"""
        url = f"{self.api_base}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = f"Context: {context}\n\nText to analyze: {chunk}" if context else chunk
        messages.append({"role": "user", "content": user_content})
        
        # Calculate auto max_tokens based on full prompt
        full_prompt = f"{system_prompt}\n{user_content}" if system_prompt else user_content
        max_tokens = self.calculate_auto_max_tokens(full_prompt)
        if max_tokens <= 200:
            max_tokens = min(max_tokens, 200)  # Keep original limit for small contexts
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            output = data['choices'][0]['message']['content']
            
            return {
                'output': output,
                'provider': 'openai',
                'model': self.model,
                'success': True
            }
        except Exception as e:
            return {
                'output': f"OpenAI error: {str(e)}",
                'provider': 'openai',
                'error': True,
                'success': False
            }
    
    def _analyze_with_ollama(self, chunk: str, context: str = "", system_prompt: str = "") -> Dict[str, Any]:
        """Analyze chunk using Ollama API"""
        url = f"{self.api_base}:{self.api_port}/api/generate"
        
        if system_prompt:
            prompt = f"{system_prompt}\n\nContext: {context}\n\nText to analyze: {chunk}\n\nAnalysis:"
        else:
            prompt = chunk
        
        # Calculate auto max_tokens for Ollama
        max_tokens = self.calculate_auto_max_tokens(prompt)
        if max_tokens <= 200:
            max_tokens = min(max_tokens, 200)  # Keep original limit for small contexts
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            output = data.get('response', '')
            
            return {
                'output': output,
                'provider': 'ollama',
                'model': self.model,
                'success': True
            }
        except Exception as e:
            return {
                'output': f"Ollama error: {str(e)}",
                'provider': 'ollama',
                'error': True,
                'success': False
            }

class HybridLLMProcessor:
    """Hybrid LLM processor that combines embedded and external models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get('llm', {}).get('mode', 'embedded')
        
        # Initialize appropriate LLM backend
        if self.mode == 'embedded':
            self.backend = EmbeddedLLM(config)
        elif self.mode == 'external':
            self.backend = ExternalAPILLM(config)
        elif self.mode == 'hybrid':
            # Use both embedded and external, prefer embedded for speed
            try:
                self.embedded_backend = EmbeddedLLM(config)
                self.external_backend = ExternalAPILLM(config)
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid mode, falling back to external: {e}")
                self.backend = ExternalAPILLM(config)
                self.mode = 'external'
        else:
            logger.warning(f"Unknown LLM mode: {self.mode}, falling back to embedded")
            self.backend = EmbeddedLLM(config)
            self.mode = 'embedded'
        
        print(f"🤖 LLM System initialized in {self.mode} mode")
    
    def generate_general_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a general summary without policy constraints"""
        # Use shorter text for faster processing unless using auto mode
        truncated_text = text[:1500] if len(text) > 1500 else text
        prompt = f"Summarize briefly:\n\n{truncated_text}\n\nSummary:"
        
        # Check if we should use auto max_tokens
        llm_config = self.config.get('llm', {})
        configured_max_tokens = llm_config.get('max_tokens', max_length)
        
        if configured_max_tokens == "auto":
            # Use full text for auto mode since we can handle larger contexts
            truncated_text = text
            prompt = f"Summarize the following text:\n\n{truncated_text}\n\nSummary:"
            max_tokens = "auto"
        else:
            max_tokens = min(max_length, 80)
        
        if self.mode == 'hybrid':
            # Try embedded first for speed
            try:
                result = self.embedded_backend.generate_text(prompt, max_tokens=max_tokens)
                if result and not result.startswith('Error'):
                    return result
            except Exception:
                pass
            
            # Fallback to external
            try:
                result = self.external_backend.analyze_chunk(truncated_text, system_prompt="Provide a brief summary.")
                return result.get('output', 'Summary generation failed')
            except Exception:
                return 'Summary generation failed'
        
        elif self.mode == 'embedded':
            try:
                return self.backend.generate_text(prompt, max_tokens=max_tokens)
            except Exception:
                return 'Summary generation failed'
        
        else:  # external
            try:
                result = self.backend.analyze_chunk(truncated_text, system_prompt="Provide a brief summary.")
                return result.get('output', 'Summary generation failed')
            except Exception:
                return 'Summary generation failed'
    
    def generate_policy_summary(self, text: str, policy_prompt: str, variables: List[str] = None) -> Dict[str, Any]:
        """Generate policy-adherent summary with structured output"""
        # Check if we should use auto max_tokens
        llm_config = self.config.get('llm', {})
        configured_max_tokens = llm_config.get('max_tokens', 100)
        
        if configured_max_tokens == "auto":
            # Use full text for auto mode since we can handle larger contexts
            truncated_text = text
            max_tokens = "auto"
        else:
            # Truncate text for faster processing in manual mode
            truncated_text = text[:1000] if len(text) > 1000 else text
            max_tokens = 100
        
        if variables:
            var_instruction = f"Extract the following variables: {', '.join(variables[:10])}"  # Increased limit for auto mode
            prompt = f"{policy_prompt}\n\n{var_instruction}\n\nText: {truncated_text}\n\nJSON:"
        else:
            prompt = f"{policy_prompt}\n\nText: {truncated_text}\n\nAnalysis:"
        
        if self.mode == 'hybrid':
            # Try embedded first
            try:
                result = self.embedded_backend.generate_text(prompt, max_tokens=max_tokens)
                if result and not result.startswith('Error'):
                    return {'output': result, 'success': True, 'mode': 'embedded'}
            except Exception:
                pass
            
            # Fallback to external
            try:
                result = self.external_backend.analyze_chunk(truncated_text, system_prompt=policy_prompt)
                return {**result, 'mode': 'external_fallback'}
            except Exception:
                return {'output': 'Policy analysis failed', 'success': False, 'mode': 'hybrid_failed'}
        
        elif self.mode == 'embedded':
            try:
                result = self.embedded_backend.generate_text(prompt, max_tokens=max_tokens)
                return {'output': result, 'success': True, 'mode': 'embedded'}
            except Exception:
                return {'output': 'Policy analysis failed', 'success': False, 'mode': 'embedded_failed'}
        
        else:  # external
            try:
                result = self.backend.analyze_chunk(truncated_text, system_prompt=policy_prompt)
                return {**result, 'mode': 'external'}
            except Exception:
                return {'output': 'Policy analysis failed', 'success': False, 'mode': 'external_failed'}
    
    def analyze_chunks(self, chunks: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Analyze multiple text chunks"""
        results = []
        
        for chunk in chunks:
            try:
                if self.mode == 'hybrid':
                    # Try embedded first
                    try:
                        result = self.embedded_backend.analyze_chunk(chunk, **kwargs)
                        if result.get('success', False):
                            results.append(result)
                            continue
                    except Exception:
                        pass
                    
                    # Fallback to external
                    try:
                        result = self.external_backend.analyze_chunk(chunk, **kwargs)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'output': f"Hybrid analysis failed: {str(e)}",
                            'error': True,
                            'success': False
                        })
                
                else:
                    result = self.backend.analyze_chunk(chunk, **kwargs)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error analyzing chunk: {e}")
                results.append({
                    'output': f"Error: {str(e)}",
                    'error': True,
                    'success': False
                })
        
        return results

class LLMSystem:
    """Unified LLM system with embedded and external support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = HybridLLMProcessor(config)
    
    def process_document(self, chunks: List[str], query: str = None, 
                        policy_prompt: str = None, variables: List[str] = None) -> Dict[str, Any]:
        """Process document chunks with LLM analysis"""
        if not chunks:
            return {'error': 'No chunks provided'}
        
        # Join chunks for full document context
        full_text = "\n---\n".join(chunks)
        
        # Generate general summary
        general_summary = self.processor.generate_general_summary(full_text)
        
        # Generate policy-adherent summary if policy provided
        policy_summary = None
        if policy_prompt:
            policy_result = self.processor.generate_policy_summary(
                full_text, policy_prompt, variables
            )
            policy_summary = policy_result
        
        # Analyze individual chunks if query provided
        chunk_analyses = []
        if query:
            chunk_analyses = self.processor.analyze_chunks(
                chunks, 
                context=query,
                system_prompt=policy_prompt if policy_prompt else ""
            )
        
        return {
            'general_summary': general_summary,
            'policy_summary': policy_summary,
            'chunk_analyses': chunk_analyses,
            'success': True,
            'mode': self.processor.mode
        }

# Legacy compatibility functions
def analyze_chunks(chunks: List[str], model: str = None, api_base: str = None, 
                  api_key: str = None, api_port: int = None, provider: str = 'koboldcpp',
                  extra_params: dict = None) -> List[Dict[str, Any]]:
    """
    Analyze text chunks using LLM API.
    
    Args:
        chunks: List of text chunks to analyze
        model: Model name/ID
        api_base: Base URL for API
        api_key: API key if required
        api_port: API port
        provider: LLM provider ('koboldcpp', 'openai')
        extra_params: Additional parameters
        
    Returns:
        List of analysis results
    """
    if not chunks:
        return []
    
    # Default configuration
    if api_base is None:
        api_base = "http://localhost"
    if api_port is None:
        api_port = 5001
    if model is None:
        model = "hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Phi-4-reasoning-plus-Q5_K_M.gguf"
    
    results = []
    
    for chunk in chunks:
        try:
            if provider == 'koboldcpp':
                result = _analyze_with_koboldcpp(chunk, api_base, api_port, model, extra_params)
            elif provider == 'openai':
                result = _analyze_with_openai(chunk, api_base, api_key, model, extra_params)
            else:
                result = {'output': f"Analyzed chunk: {chunk[:100]}...", 'provider': provider}
            
            results.append(result)
        except Exception as e:
            logging.error(f"Error analyzing chunk: {e}")
            results.append({'output': f"Error: {str(e)}", 'error': True})
    
    return results


def _analyze_with_koboldcpp(chunk: str, api_base: str, api_port: int, model: str, 
                           extra_params: dict = None) -> Dict[str, Any]:
    """Analyze chunk using KoboldCpp API"""
    url = f"{api_base}:{api_port}/api/v1/generate"
    
    payload = {
        "prompt": chunk,
        "max_length": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    if extra_params:
        payload.update(extra_params)
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and data['results']:
            output = data['results'][0].get('text', '')
        else:
            output = data.get('text', str(data))
            
        return {
            'output': output,
            'provider': 'koboldcpp',
            'model': model,
            'success': True
        }
    except Exception as e:
        return {
            'output': f"KoboldCpp error: {str(e)}",
            'provider': 'koboldcpp',
            'error': True,
            'success': False
        }


def _analyze_with_openai(chunk: str, api_base: str, api_key: str, model: str,
                        extra_params: dict = None) -> Dict[str, Any]:
    """Analyze chunk using OpenAI API"""
    url = f"{api_base}/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": chunk}],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    if extra_params:
        payload.update(extra_params)
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        output = data['choices'][0]['message']['content']
        
        return {
            'output': output,
            'provider': 'openai',
            'model': model,
            'success': True
        }
    except Exception as e:
        return {
            'output': f"OpenAI error: {str(e)}",
            'provider': 'openai',
            'error': True,
            'success': False
        }


def rag_llm_pipeline(document_path: str, module_key: str, vars_json_path: str = None,
                    policy_prompt_path: str = None, model: str = None, 
                    api_base: str = None, api_key: str = None, api_port: int = None,
                    provider: str = None, output_path: str = None) -> Dict[str, Any]:
    """
    Full RAG-LLM pipeline for document analysis.
    
    Args:
        document_path: Path to document to analyze
        module_key: Interview module key
        vars_json_path: Path to variables JSON
        policy_prompt_path: Path to policy prompt
        model: LLM model name
        api_base: API base URL
        api_key: API key
        api_port: API port
        provider: LLM provider
        output_path: Output file path
        
    Returns:
        Analysis results
    """
    from .rag import chunk_document, load_policy_prompt
    import json
    import os
    import sys
    from datetime import datetime
    
    # Chunk document
    chunks = chunk_document(document_path)
    if not chunks or not any(c.strip() for c in chunks):
        return {'error': f"No usable text could be extracted from {document_path}"}
    
    # Load policy prompt
    if policy_prompt_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
        policy_prompt_path = os.path.join(base_dir, 'policy_prompt.yaml')
    
    policy_prompt = load_policy_prompt(policy_prompt_path)
    if not policy_prompt or not policy_prompt.strip():
        return {'error': "No usable policy prompt found"}
    
    # Load variable config
    if vars_json_path is None:
        vars_json_path = os.path.join(os.path.dirname(__file__), 'modules', f'{module_key}_vars.json')
    
    if not os.path.exists(vars_json_path):
        return {'error': f"Variable config file not found: {vars_json_path}"}
    
    with open(vars_json_path, 'r', encoding='utf-8') as f:
        try:
            logic_tree = json.load(f)
        except Exception:
            return {'error': f"Could not parse variable config: {vars_json_path}"}
    
    if not logic_tree or not isinstance(logic_tree, dict):
        return {'error': f"No usable variable schema found in {vars_json_path}"}
    
    var_list = list(logic_tree.keys())
    if not var_list:
        return {'error': "No variables defined in schema"}
    
    # LLM analysis
    extraction_prompt = (f"{policy_prompt}\n\nExtract as many of the following variables as possible "
                        f"from the document below. Output as a JSON object with variable names as keys.\n"
                        f"Variables: {var_list}\nDocument: {''.join(chunks)}")
    
    llm_outputs = analyze_chunks([extraction_prompt], model=model, api_base=api_base, 
                                api_key=api_key, api_port=api_port, provider=provider)
    
    if not llm_outputs or not llm_outputs[0]:
        return {'error': "LLM did not return usable output"}
    
    llm_json = llm_outputs[0]['output'] if isinstance(llm_outputs[0], dict) else llm_outputs[0]
    
    try:
        llm_vars = json.loads(llm_json)
    except Exception:
        llm_vars = {}
    
    if not llm_vars or not isinstance(llm_vars, dict):
        return {'error': "LLM did not extract any variables"}
    
    # Generate output
    result = {
        'success': True,
        'extracted_variables': llm_vars,
        'chunks': chunks,
        'module_key': module_key,
        'document_path': document_path,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file if output_path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result['output_path'] = output_path
    
    return result
