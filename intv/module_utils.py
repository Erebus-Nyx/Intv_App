"""
Module utilities for INTV - provides functions to discover and work with interview modules
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_available_interview_types() -> List[Dict[str, Any]]:
    """
    Scan for available interview module types by looking at *_vars.json files
    in the modules directory.
    
    Returns:
        List of dictionaries with module information:
        - key: module key/type identifier
        - name: display name
        - description: module description
        - display: formatted display name (for backward compatibility)
    """
    modules = []
    
    # Get the modules directory path
    modules_dir = Path(__file__).parent / "modules"
    
    if not modules_dir.exists():
        return modules
    
    # Scan for *_vars.json files
    for json_file in modules_dir.glob("*_vars.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract header information
            header = data.get('_header', {})
            if header:
                module_key = header.get('type', json_file.stem.replace('_vars', ''))
                module_name = header.get('label', module_key.title())
                module_description = header.get('description', f'Module for {module_name} interviews')
                
                modules.append({
                    'key': module_key,
                    'name': module_name,
                    'description': module_description,
                    'display': module_name  # For backward compatibility
                })
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Skip invalid or malformed JSON files
            continue
    
    # Sort by key for consistent ordering
    modules.sort(key=lambda x: x['key'])
    
    return modules


def detect_filetype_from_extension(filepath: str) -> Optional[str]:
    """
    Detect file type from extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File type string or None if unknown
    """
    filepath = Path(filepath)
    extension = filepath.suffix.lower()
    
    # Audio formats
    if extension in ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']:
        return 'audio'
    
    # Document formats
    elif extension in ['.txt', '.rtf', '.docx', '.doc']:
        return 'document'
    
    # PDF
    elif extension == '.pdf':
        return 'pdf'
    
    # Image formats (for OCR)
    elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return 'image'
    
    return None


def get_module_vars_path(module_key: str) -> Optional[Path]:
    """
    Get the path to the vars.json file for a given module.
    
    Args:
        module_key: The module key/type identifier
        
    Returns:
        Path to the vars.json file or None if not found
    """
    modules_dir = Path(__file__).parent / "modules"
    
    # Try different naming patterns
    possible_names = [
        f"{module_key}_vars.json",
        f"intv_{module_key}_vars.json"
    ]
    
    for name in possible_names:
        vars_file = modules_dir / name
        if vars_file.exists():
            return vars_file
    
    return None


def load_module_vars(module_key: str) -> Optional[Dict[str, Any]]:
    """
    Load the variables configuration for a given module.
    
    Args:
        module_key: The module key/type identifier
        
    Returns:
        Dictionary with module variables or None if not found
    """
    vars_path = get_module_vars_path(module_key)
    if not vars_path:
        return None
    
    try:
        with open(vars_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
