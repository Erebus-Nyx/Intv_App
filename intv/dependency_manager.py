#!/usr/bin/env python3
"""
INTV Dependency Manager

This module helps manage optional dependencies and provides installation guidance
for pipx-installed applications.
"""

import importlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import platform
import os

class DependencyManager:
    """Manages optional dependencies and provides installation guidance."""
    
    # Core dependency groups with installation commands
    DEPENDENCY_GROUPS = {
        'core': {
            'packages': ['PyPDF2', 'python-docx', 'requests', 'pyyaml', 'psutil', 'click', 'tqdm'],
            'install_cmd': 'pipx inject intv PyPDF2 python-docx requests pyyaml psutil click tqdm',
            'description': 'Core document processing and utilities'
        },
        'ml': {
            'packages': ['torch', 'transformers', 'sentence-transformers', 'numpy'],
            'install_cmd': 'pipx inject intv torch transformers sentence-transformers numpy',
            'description': 'Machine learning and embeddings'
        },
        'ocr': {
            'packages': ['pytesseract', 'Pillow', 'pdf2image'],
            'install_cmd': 'pipx inject intv pytesseract Pillow pdf2image',
            'description': 'OCR and image processing'
        },
        'audio': {
            'packages': ['faster-whisper', 'sounddevice', 'soundfile'],
            'install_cmd': 'pipx inject intv faster-whisper sounddevice soundfile',
            'description': 'Audio transcription and processing'
        },
        'rag': {
            'packages': ['faiss-cpu', 'chromadb'],
            'install_cmd': 'pipx inject intv faiss-cpu chromadb',
            'description': 'RAG and vector search'
        },
        'gpu': {
            'packages': ['torch'],  # We'll check for CUDA separately
            'install_cmd': 'pipx inject intv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
            'description': 'GPU acceleration with CUDA'
        }
    }
    
    def __init__(self):
        self.status = {}
        self.check_all_dependencies()
    
    def check_package(self, package_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a package is available and return version if found."""
        try:
            module = importlib.import_module(package_name.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, None
    
    def check_cuda(self) -> Tuple[bool, str]:
        """Check for CUDA availability."""
        try:
            import torch
            if torch.cuda.is_available():
                return True, f"CUDA {torch.version.cuda}, GPU count: {torch.cuda.device_count()}"
            else:
                return False, "PyTorch available but CUDA not detected"
        except ImportError:
            return False, "PyTorch not installed"
    
    def check_all_dependencies(self):
        """Check status of all dependency groups."""
        for group_name, group_info in self.DEPENDENCY_GROUPS.items():
            group_status = {
                'available': [],
                'missing': [],
                'description': group_info['description'],
                'install_cmd': group_info['install_cmd']
            }
            
            for package in group_info['packages']:
                is_available, version = self.check_package(package)
                if is_available:
                    group_status['available'].append(f"{package} ({version})")
                else:
                    group_status['missing'].append(package)
            
            self.status[group_name] = group_status
        
        # Special handling for GPU/CUDA
        cuda_available, cuda_info = self.check_cuda()
        self.status['gpu']['cuda_status'] = (cuda_available, cuda_info)
    
    def get_installation_guide(self, missing_only: bool = True) -> str:
        """Generate installation guide for missing dependencies."""
        guide = []
        guide.append("=== INTV DEPENDENCY INSTALLATION GUIDE ===")
        guide.append("Since INTV is installed with pipx, use 'pipx inject' to add dependencies:\n")
        
        for group_name, status in self.status.items():
            if missing_only and not status['missing']:
                continue
                
            guide.append(f"📦 {group_name.upper()}: {status['description']}")
            
            if status['available']:
                guide.append(f"   ✅ Available: {', '.join(status['available'])}")
            
            if status['missing']:
                guide.append(f"   ❌ Missing: {', '.join(status['missing'])}")
                guide.append(f"   💡 Install: {status['install_cmd']}")
            
            if group_name == 'gpu':
                cuda_available, cuda_info = status.get('cuda_status', (False, 'Unknown'))
                if cuda_available:
                    guide.append(f"   🚀 CUDA Status: {cuda_info}")
                else:
                    guide.append(f"   ⚠️  CUDA Status: {cuda_info}")
            
            guide.append("")
        
        # Add system-specific recommendations
        guide.append("=== SYSTEM-SPECIFIC RECOMMENDATIONS ===")
        
        # Detect system type and recommend appropriate packages
        try:
            import subprocess
            nvidia_available = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
        except:
            nvidia_available = False
        
        if nvidia_available:
            guide.append("🎯 NVIDIA GPU detected! Recommended full installation:")
            guide.append("   pipx inject intv torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            guide.append("   pipx inject intv transformers sentence-transformers")
            guide.append("   pipx inject intv PyPDF2 python-docx pytesseract Pillow pdf2image")
            guide.append("   pipx inject intv faiss-cpu chromadb")
        else:
            guide.append("💻 CPU-only system detected. Recommended installation:")
            guide.append("   pipx inject intv torch transformers sentence-transformers")
            guide.append("   pipx inject intv PyPDF2 python-docx pytesseract Pillow pdf2image")
        
        guide.append("")
        guide.append("=== VERIFY INSTALLATION ===")
        guide.append("After installing dependencies, verify with:")
        guide.append("   intv-platform  # Shows system capabilities")
        guide.append("   intv --help     # Shows available commands")
        
        return "\n".join(guide)
    
    def print_status(self):
        """Print current dependency status."""
        print(self.get_installation_guide(missing_only=False))
    
    def has_group(self, group_name: str) -> bool:
        """Check if all packages in a group are available."""
        if group_name not in self.status:
            return False
        return len(self.status[group_name]['missing']) == 0
    
    def get_missing_for_feature(self, feature: str) -> List[str]:
        """Get missing packages for a specific feature."""
        feature_groups = {
            'rag': ['core', 'ml', 'rag'],
            'ocr': ['core', 'ocr'],
            'audio': ['core', 'audio', 'ml'],
            'gpu': ['core', 'ml', 'gpu'],
            'document': ['core', 'ocr']
        }
        
        missing = []
        for group in feature_groups.get(feature, []):
            if group in self.status:
                missing.extend(self.status[group]['missing'])
        
        return list(set(missing))  # Remove duplicates

# Global instance
_dependency_manager = None

def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager

def check_feature_dependencies(feature: str) -> Tuple[bool, List[str]]:
    """Check if dependencies for a feature are available."""
    dm = get_dependency_manager()
    missing = dm.get_missing_for_feature(feature)
    return len(missing) == 0, missing

def print_installation_guide():
    """Print installation guide for missing dependencies."""
    dm = get_dependency_manager()
    dm.print_status()

if __name__ == "__main__":
    print_installation_guide()
