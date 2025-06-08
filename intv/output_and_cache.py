"""
INTV Output and Cache Management
"""
import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

logger = logging.getLogger(__name__)

def ensure_output_dir(output_path: str = None) -> str:
    """Ensure output directory exists and return the path"""
    if output_path is None:
        # Default to output directory in project root
        project_root = Path(__file__).parent.parent
        output_path = project_root / "output"
    else:
        output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure it's writable
    if not os.access(output_path, os.W_OK):
        logger.warning(f"Output directory {output_path} is not writable")
        # Fallback to temp directory
        output_path = Path(tempfile.gettempdir()) / "intv_output"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using fallback output directory: {output_path}")
    
    return str(output_path)

def ensure_cache_dir(cache_path: str = None) -> str:
    """Ensure cache directory exists and return the path"""
    if cache_path is None:
        # Default to cache directory in project root
        project_root = Path(__file__).parent.parent
        cache_path = project_root / ".cache"
    else:
        cache_path = Path(cache_path)
    
    # Create directory if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)
    
    return str(cache_path)

def purge_cache(cache_path: str = None, older_than_days: int = 7) -> Dict[str, Any]:
    """Purge cache files older than specified days"""
    cache_dir = Path(ensure_cache_dir(cache_path))
    
    if not cache_dir.exists():
        return {'success': True, 'message': 'Cache directory does not exist', 'files_removed': 0}
    
    cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
    files_removed = 0
    bytes_freed = 0
    errors = []
    
    try:
        for item in cache_dir.rglob('*'):
            if item.is_file():
                try:
                    # Check file modification time
                    if item.stat().st_mtime < cutoff_time:
                        file_size = item.stat().st_size
                        item.unlink()
                        files_removed += 1
                        bytes_freed += file_size
                        logger.debug(f"Removed cache file: {item}")
                except Exception as e:
                    errors.append(f"Failed to remove {item}: {str(e)}")
                    logger.warning(f"Failed to remove cache file {item}: {e}")
        
        # Remove empty directories
        for item in cache_dir.rglob('*'):
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                    logger.debug(f"Removed empty cache directory: {item}")
                except Exception as e:
                    logger.debug(f"Could not remove empty directory {item}: {e}")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'bytes_freed': bytes_freed,
            'errors': errors,
            'message': f'Removed {files_removed} files, freed {bytes_freed} bytes'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'files_removed': files_removed,
            'bytes_freed': bytes_freed
        }

def save_output_file(data: Any, filename: str, output_dir: str = None, format: str = 'json') -> str:
    """Save data to output file"""
    output_dir = ensure_output_dir(output_dir)
    output_path = Path(output_dir) / filename
    
    try:
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Saved output to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to save output file {output_path}: {e}")
        raise

def load_output_file(filename: str, output_dir: str = None, format: str = 'json') -> Any:
    """Load data from output file"""
    output_dir = ensure_output_dir(output_dir)
    output_path = Path(output_dir) / filename
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    
    try:
        if format.lower() == 'json':
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format.lower() == 'txt':
            with open(output_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported input format: {format}")
            
    except Exception as e:
        logger.error(f"Failed to load output file {output_path}: {e}")
        raise

def list_output_files(output_dir: str = None, pattern: str = '*') -> List[str]:
    """List files in output directory"""
    output_dir = ensure_output_dir(output_dir)
    output_path = Path(output_dir)
    
    try:
        files = [str(f.name) for f in output_path.glob(pattern) if f.is_file()]
        return sorted(files)
    except Exception as e:
        logger.error(f"Failed to list output files: {e}")
        return []

def cleanup_output_dir(output_dir: str = None, older_than_days: int = 30) -> Dict[str, Any]:
    """Clean up old output files"""
    output_dir = ensure_output_dir(output_dir)
    output_path = Path(output_dir)
    
    cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
    files_removed = 0
    bytes_freed = 0
    errors = []
    
    try:
        for item in output_path.iterdir():
            if item.is_file():
                try:
                    if item.stat().st_mtime < cutoff_time:
                        file_size = item.stat().st_size
                        item.unlink()
                        files_removed += 1
                        bytes_freed += file_size
                        logger.debug(f"Removed output file: {item}")
                except Exception as e:
                    errors.append(f"Failed to remove {item}: {str(e)}")
                    logger.warning(f"Failed to remove output file {item}: {e}")
        
        return {
            'success': True,
            'files_removed': files_removed,
            'bytes_freed': bytes_freed,
            'errors': errors,
            'message': f'Removed {files_removed} old files, freed {bytes_freed} bytes'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'files_removed': files_removed,
            'bytes_freed': bytes_freed
        }

def get_storage_info(output_dir: str = None, cache_dir: str = None) -> Dict[str, Any]:
    """Get storage information for output and cache directories"""
    output_dir = ensure_output_dir(output_dir)
    cache_dir = ensure_cache_dir(cache_dir)
    
    def get_dir_size(path: Path) -> int:
        """Get total size of directory"""
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size for {path}: {e}")
        return total
    
    output_path = Path(output_dir)
    cache_path = Path(cache_dir)
    
    return {
        'output_dir': {
            'path': str(output_path),
            'exists': output_path.exists(),
            'size_bytes': get_dir_size(output_path) if output_path.exists() else 0,
            'file_count': len(list(output_path.iterdir())) if output_path.exists() else 0
        },
        'cache_dir': {
            'path': str(cache_path),
            'exists': cache_path.exists(),
            'size_bytes': get_dir_size(cache_path) if cache_path.exists() else 0,
            'file_count': len(list(cache_path.rglob('*'))) if cache_path.exists() else 0
        }
    }
