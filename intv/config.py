"""
Configuration module for INTV.
Provides configuration loading and management functionality.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Config file paths
CONFIG_YAML_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'
CONFIG_JSON_PATH = Path(__file__).parent.parent / 'src' / 'config.json'
SETTINGS_JSON_PATH = Path(__file__).parent.parent / 'settings.json'

DEFAULT_CONFIG = {
    "llm_api_base": "http://localhost",
    "llm_api_key": None,
    "llm_api_port": 5001,  # Default to koboldcpp port
    "llm_provider": "koboldcpp",  # Default to koboldcpp
    "model": "hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M",
    "external_rag": False,
    "purge_variables": False,
    "name": "User"
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from config.yaml, then populate settings.json and environment variables.
    
    Args:
        config_path: Optional path to config file (for compatibility)
    """
    config = DEFAULT_CONFIG.copy()
    
    # First, try to load from config.yaml
    if CONFIG_YAML_PATH.exists() and HAS_YAML:
        try:
            with open(CONFIG_YAML_PATH, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
            config.update(yaml_config)
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
    
    # Then try to load from config.json
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, 'r') as f:
                json_config = json.load(f)
            config.update(json_config)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    
    # Save to settings.json for runtime use
    try:
        with open(SETTINGS_JSON_PATH, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write settings.json: {e}")
    
    # Set environment variables
    for key, value in config.items():
        if value is not None:
            os.environ[key.upper()] = str(value)
    
    return config


def get_config_value(key: str, default=None):
    """Get a configuration value by key."""
    config = load_config()
    return config.get(key, default)


def update_config(updates: Dict[str, Any]):
    """Update configuration with new values."""
    config = load_config()
    config.update(updates)
    
    # Save back to settings.json
    try:
        with open(SETTINGS_JSON_PATH, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update settings.json: {e}")
    
    return config
