import json
import yaml
import os
from pathlib import Path

# Config file paths
CONFIG_YAML_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'
CONFIG_JSON_PATH = Path(__file__).parent / 'config.json'
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

def load_config():
    """Load configuration from config.yaml, then populate settings.json and environment variables."""
    config = DEFAULT_CONFIG.copy()
    
    # First, try to load from config.yaml
    if CONFIG_YAML_PATH.exists():
        try:
            with CONFIG_YAML_PATH.open('r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            # Merge YAML config with defaults
            config.update(yaml_config)
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
    
    # Fallback to config.json for backward compatibility
    elif CONFIG_JSON_PATH.exists():
        try:
            with CONFIG_JSON_PATH.open('r', encoding='utf-8') as f:
                json_config = json.load(f)
            config.update(json_config)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    
    # Populate environment variables
    _populate_environment_variables(config)
    
    # Save to settings.json for runtime use
    _save_settings_json(config)
    
    return config

def _populate_environment_variables(config):
    """Populate environment variables from config."""
    env_mapping = {
        'llm_api_base': 'LLM_API_BASE',
        'llm_api_key': 'LLM_API_KEY', 
        'llm_api_port': 'LLM_API_PORT',
        'llm_provider': 'LLM_PROVIDER',
        'model': 'MODEL',
        'external_rag': 'EXTERNAL_RAG',
        'purge_variables': 'PURGE_VARIABLES'
    }
    
    for config_key, env_key in env_mapping.items():
        if config_key in config and config[config_key] is not None:
            os.environ[env_key] = str(config[config_key])

def _save_settings_json(config):
    """Save config to settings.json for runtime access."""
    try:
        with SETTINGS_JSON_PATH.open('w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save settings.json: {e}")

def save_config(config: dict):
    """Save config to both JSON formats for backward compatibility."""
    # Save to config.json
    with CONFIG_JSON_PATH.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Also save to settings.json
    _save_settings_json(config)
