"""
LLM module for INTV - handles Large Language Model interactions
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional


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
        model = "hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q5_K_M"
    
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
