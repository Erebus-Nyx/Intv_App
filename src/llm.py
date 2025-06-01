# llm.py - LLM interface

import os
import requests
import yaml

# Helper to load structured system/policy prompt from config/policy_prompt.yaml
_DEF_POLICY_PROMPT = "You are a professional, neutral, and privacy-compliant assistant. Always follow organizational policy, never provide legal/medical advice, and always clarify when uncertain."
def load_policy_prompt():
    from src.config import load_config
    config = load_config()
    # Try config.yaml first
    policy = config.get("system_prompt")
    if policy:
        return policy
    # Try structured YAML in config/policy_prompt.yaml
    import os
    policy_path = os.path.join(os.path.dirname(__file__), "..", "config", "policy_prompt.yaml")
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            policy_yaml = yaml.safe_load(f)
        sections = []
        for key in ["behavior", "writing_guidelines", "compliance"]:
            if key in policy_yaml:
                sections.append(policy_yaml[key])
        if sections:
            return "\n\n".join(sections)
    # Try policy_prompt.txt in project root (legacy)
    txt_path = os.path.join(os.path.dirname(__file__), "..", "policy_prompt.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return _DEF_POLICY_PROMPT

def ensure_ollama_model(model: str, api_base: str = 'http://localhost', api_port: int = 11434):
    """
    Ensure the Ollama model exists locally, download if not present.
    """
    import requests
    url = f"{api_base}:{api_port}/api/tags"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        tags = resp.json().get('models', [])
        if any(m.get('name', '').lower() == model.lower() for m in tags):
            return True
    except Exception:
        pass
    # If not found, try to pull the model
    pull_url = f"{api_base}:{api_port}/api/pull"
    resp = requests.post(pull_url, json={"name": model})
    resp.raise_for_status()
    return resp.ok

def analyze_chunks(rag_results, model, api_base=None, api_key=None, api_port=None, provider='openai', extra_params=None):
    """
    Analyze document chunks using an LLM API (OpenAI or compatible, e.g., Ollama, KoboldCpp).
    Args:
        rag_results: List of text chunks or RAG-processed data.
        model: Model name or ID.
        api_base: Base URL for the API (e.g., 'http://localhost').
        api_key: API key if required.
        api_port: Port for the API (e.g., 11434 for Ollama).
        provider: 'openai', 'ollama', 'koboldcpp', etc.
        extra_params: Dict of extra params for the API call.
    Returns:
        LLM response as string or dict.
    """
    if isinstance(rag_results, list):
        prompt = '\n---\n'.join(rag_results)
    else:
        prompt = str(rag_results)
    if not api_base:
        api_base = os.environ.get('LLM_API_BASE', 'https://api.openai.com')
    if not api_port:
        api_port = os.environ.get('LLM_API_PORT', None)
    if not api_key:
        api_key = os.environ.get('LLM_API_KEY', None)
    if provider == 'openai':
        url = f"{api_base}/v1/chat/completions"
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': policy},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7
        }
        if extra_params:
            data.update(extra_params)
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    elif provider == 'ollama':
        # Ollama local API (default port 11434)
        port = api_port or 11434
        url = f"{api_base}:{port}/api/chat"
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': policy},
                {'role': 'user', 'content': prompt}
            ]
        }
        if extra_params:
            data.update(extra_params)
        # If CUDA requested, add 'options' for Ollama
        if extra_params and extra_params.get('cuda'):
            data.setdefault('options', {})['numa'] = False  # Example: set NUMA off for CUDA, or add other CUDA flags
            data['options']['cuda'] = True
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        return resp.json().get('message', resp.text)
    elif provider == 'koboldcpp':
        # KoboldCpp API (default port 5001)
        port = api_port or 5001
        url = f"{api_base}:{port}/api/v1/generate"
        # For text-only models, prepend policy to prompt
        full_prompt = f"{policy}\n\n{prompt}"
        data = {
            'prompt': full_prompt,
            'max_new_tokens': 512,
            'mode': 'chat',
            'character': 'User',
            'context': '',
            'model': model
        }
        if extra_params:
            data.update(extra_params)
        resp = requests.post(url, json=data)
        resp.raise_for_status()
        return resp.json().get('results', [{}])[0].get('text', resp.text)
    else:
        raise NotImplementedError(f"Provider '{provider}' not supported.")
