"""
INTV RAG System - Embedded and External RAG Support

This module provides a unified interface for RAG operations with support for:
- Embedded RAG using local models
- External Tika API for document parsing
- External Haystack API for advanced RAG
- Automatic model selection based on system capabilities
- Progressive model downloading with status indicators
"""

import os
import sys
import platform
import psutil
import logging
import requests
import time
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING

# Check for optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if TYPE_CHECKING or HAS_NUMPY:
    try:
        import numpy as np
    except ImportError:
        pass
from pathlib import Path
import json
import tempfile
import shutil

# GPU detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
    HAS_NUMPY = True
except ImportError:
    HAS_SKLEARN = False
    HAS_NUMPY = False
    # Fallback imports
    try:
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False

# Setup logging
logger = logging.getLogger(__name__)

class SystemCapabilities:
    """Detect system capabilities for automatic model selection"""
    
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
    def get_default_model(system_type: str) -> str:
        """Get default embedding model based on system capabilities"""
        models = {
            'raspberry_pi': 'hf.co/sentence-transformers/all-MiniLM-L6-v2',
            'arm_low_end': 'hf.co/sentence-transformers/all-MiniLM-L6-v2', 
            'cpu_minimal': 'hf.co/sentence-transformers/all-MiniLM-L6-v2',
            'cpu_low': 'hf.co/sentence-transformers/all-MiniLM-L6-v2',
            'cpu_medium': 'hf.co/sentence-transformers/all-mpnet-base-v2',
            'cpu_high': 'hf.co/sentence-transformers/all-mpnet-base-v2',
            'gpu_low': 'hf.co/sentence-transformers/all-mpnet-base-v2',
            'gpu_medium': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'gpu_high': 'hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1',
            # Legacy support
            'cpu_low_end': 'hf.co/sentence-transformers/all-MiniLM-L6-v2',
            'cuda_high_end': 'hf.co/sentence-transformers/all-mpnet-base-v2'
        }
        return models.get(system_type, 'hf.co/sentence-transformers/all-MiniLM-L6-v2')

class ModelDownloader:
    """Handle HuggingFace model downloading with progress indicators"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def parse_model_string(self, model_string: str) -> tuple:
        """Parse model string to extract repo_id and filename if specified"""
        if model_string.startswith('hf.co/'):
            model_string = model_string[6:]  # Remove 'hf.co/' prefix
            
        if ':' in model_string:
            repo_id, filename = model_string.split(':', 1)
            return repo_id, filename
        else:
            return model_string, None
    
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
        
        # For sentence transformers, check if directory exists
        if not filename:
            model_path = self.model_dir / repo_id.replace('/', '--')
            return model_path.exists() and any(model_path.iterdir())
        
        # For specific files, check if file exists
        model_path = self.model_dir / repo_id.replace('/', '--') / filename
        return model_path.exists()
    
    def download_model(self, model_string: str, force: bool = False) -> Path:
        """Download model from HuggingFace with progress indicator"""
        if not force and self.is_model_downloaded(model_string):
            logger.info(f"Model {model_string} already downloaded")
            repo_id, _ = self.parse_model_string(model_string)
            return self.model_dir / repo_id.replace('/', '--')
        
        repo_id, filename = self.parse_model_string(model_string)
        local_dir = self.model_dir / repo_id.replace('/', '--')
        local_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading model: {model_string}")
        
        try:
            if HAS_SENTENCE_TRANSFORMERS and not filename:
                # Use sentence-transformers for full model download
                print(f"⚡ Using sentence-transformers for {repo_id}")
                model = SentenceTransformer(repo_id, cache_folder=str(self.model_dir))
                # Move to expected location
                cache_dir = Path(model._modules['0'].auto_model.config._name_or_path)
                if cache_dir != local_dir:
                    if local_dir.exists():
                        shutil.rmtree(local_dir)
                    shutil.move(str(cache_dir), str(local_dir))
                
            elif HAS_TRANSFORMERS:
                # Use transformers for model download
                print(f"🤗 Using transformers for {repo_id}")
                if filename:
                    # Download specific file
                    from huggingface_hub import hf_hub_download
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(self.model_dir),
                        local_dir=str(local_dir)
                    )
                else:
                    # Download full model
                    AutoTokenizer.from_pretrained(repo_id, cache_dir=str(local_dir))
                    AutoModel.from_pretrained(repo_id, cache_dir=str(local_dir))
            else:
                raise ImportError("Neither sentence-transformers nor transformers available")
                
            print(f"✅ Model downloaded successfully: {model_string}")
            return local_dir
            
        except Exception as e:
            logger.error(f"Failed to download model {model_string}: {e}")
            print(f"❌ Failed to download model {model_string}: {e}")
            raise

class EmbeddedRAG:
    """Embedded RAG using local models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.downloader = ModelDownloader(config.get('model_dir', 'models'))
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        rag_config = self.config.get('rag', {}).get('embedded', {})
        model_string = rag_config.get('model', 'auto')
        
        if model_string == 'auto':
            system_type = SystemCapabilities.detect_system_type()
            model_string = SystemCapabilities.get_default_model(system_type)
            print(f"🎯 Auto-selected model for {system_type}: {model_string}")
        
        # Download model if needed
        try:
            model_path = self.downloader.download_model(model_string)
            
            # Load model
            if HAS_SENTENCE_TRANSFORMERS:
                repo_id, _ = self.downloader.parse_model_string(model_string)
                self.model = SentenceTransformer(str(model_path))
                print(f"✅ Loaded embedding model: {model_string}")
            else:
                logger.warning("sentence-transformers not available, using fallback")
                self.model = None
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
    
    def encode_texts(self, texts: List[str]):
        """Encode texts into embeddings"""
        if self.model is None:
            # Fallback: simple keyword matching
            logger.warning("No embedding model available, using keyword fallback")
            return None
        
        if not HAS_NUMPY:
            logger.warning("NumPy not available, using fallback encoding")
            return None
            
        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return None
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        """Retrieve most relevant chunks for query"""
        if not chunks:
            return []
            
        # Try embedding-based retrieval
        if self.model is not None:
            try:
                query_embedding = self.model.encode([query])
                chunk_embeddings = self.model.encode(chunks)
                
                if HAS_SKLEARN:
                    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                    top_indices = similarities.argsort()[-top_k:][::-1]
                    return [chunks[i] for i in top_indices if similarities[i] > 0.1]
                else:
                    # Fallback without sklearn
                    logger.warning("sklearn not available, using keyword fallback")
            except Exception as e:
                logger.error(f"Error in embedding retrieval: {e}")
        
        # Fallback: keyword-based retrieval
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

class ExternalTikaRAG:
    """External RAG using Apache Tika API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('rag', {}).get('external_tika', {})
        self.api_url = self.config.get('api_url', 'http://localhost:9998')
        self.timeout = self.config.get('timeout', 30)
    
    def extract_text(self, file_path: str) -> str:
        """Extract text using Tika API"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{self.api_url}/tika",
                    files={'file': f},
                    headers={'Accept': 'text/plain'},
                    timeout=self.timeout
                )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Tika extraction failed: {e}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        """Simple keyword-based retrieval for external Tika"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words))
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

class ExternalHaystackRAG:
    """External RAG using Haystack API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('rag', {}).get('external_haystack', {})
        self.api_url = self.config.get('api_url', 'http://localhost:8000')
        self.api_key = self.config.get('api_key', '')
        self.timeout = self.config.get('timeout', 30)
    
    def query_haystack(self, query: str, context: str) -> str:
        """Query Haystack API"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        payload = {
            'query': query,
            'context': context
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/query",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get('answer', '')
        except Exception as e:
            logger.error(f"Haystack query failed: {e}")
            raise

class RAGSystem:
    """Unified RAG system with embedded and external support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get('rag', {}).get('mode', 'embedded')
        
        # Initialize appropriate RAG backend
        if self.mode == 'embedded':
            self.backend = EmbeddedRAG(config)
        elif self.mode == 'external_tika':
            self.backend = ExternalTikaRAG(config)
        elif self.mode == 'external_haystack':
            self.backend = ExternalHaystackRAG(config)
        else:
            logger.warning(f"Unknown RAG mode: {self.mode}, falling back to embedded")
            self.backend = EmbeddedRAG(config)
            self.mode = 'embedded'
        
        print(f"🔍 RAG System initialized in {self.mode} mode")
    
    def process_query(self, query: str, chunks: List[str]) -> Dict[str, Any]:
        """Process query against chunks and return results"""
        rag_config = self.config.get('rag', {}).get('embedded', {})
        top_k = rag_config.get('top_k', 5)
        
        if self.mode == 'external_haystack':
            # For Haystack, send full context
            context = "\n---\n".join(chunks)
            try:
                answer = self.backend.query_haystack(query, context)
                return {
                    'relevant_chunks': chunks[:top_k],
                    'context': context,
                    'answer': answer,
                    'mode': self.mode
                }
            except Exception as e:
                logger.error(f"Haystack query failed, falling back to embedded: {e}")
                # Fallback to embedded mode
                fallback_backend = EmbeddedRAG(self.config)
                relevant_chunks = fallback_backend.retrieve_relevant_chunks(query, chunks, top_k)
        else:
            # For embedded and Tika, retrieve relevant chunks
            relevant_chunks = self.backend.retrieve_relevant_chunks(query, chunks, top_k)
        
        context = "\n---\n".join(relevant_chunks)
        
        return {
            'relevant_chunks': relevant_chunks,
            'context': context,
            'answer': None,  # To be filled by LLM
            'mode': self.mode
        }

def test_rag_system():
    """Test the RAG system"""
    config = {
        'rag': {
            'mode': 'embedded',
            'embedded': {
                'model': 'auto',
                'chunk_size': 1000,
                'chunk_overlap': 100,
                'top_k': 3
            }
        },
        'model_dir': 'models'
    }
    
    rag = RAGSystem(config)
    
    chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language.",
        "Machine learning models can process text.",
        "The weather is sunny today."
    ]
    
    result = rag.process_query("What is Python?", chunks)
    print("Test result:", result)

if __name__ == '__main__':
    test_rag_system()
