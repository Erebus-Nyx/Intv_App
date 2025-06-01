import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'config.json'

DEFAULT_CONFIG = {
    "llm_api_base": "http://localhost",
    "llm_api_key": None,
    "llm_api_port": 11434,
    "llm_provider": "ollama",
    "model": "hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M",
    "external_rag": False,
    "purge_variables": False,
    "name": "User"
}

def load_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open('r', encoding='utf-8') as f:
            config = json.load(f)
        # Fill in any missing keys with defaults
        for k, v in DEFAULT_CONFIG.items():
            config.setdefault(k, v)
        return config
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    with CONFIG_PATH.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
