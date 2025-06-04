def analyze_chunks(chunks, model=None, api_base=None, api_key=None, api_port=None, provider=None, extra_params=None):
    """
    Minimal stub for analyze_chunks. Replace with actual LLM integration as needed.
    Returns a list of dicts with 'output' for each chunk.
    """
    # This stub just echoes the input chunks for now
    return [{'output': str(chunk)} for chunk in chunks]


def rag_llm_pipeline(
    document_path,
    module_key,
    vars_json_path=None,
    policy_prompt_path=None,
    model=None,
    api_base=None,
    api_key=None,
    api_port=None,
    provider=None,
    output_path=None
):
    """
    Full pipeline: chunk document, pass to LLM, extract/interview variables, prompt user for missing/uncertain info, and output JSON blocks as specified.
    If output_path is not provided, open a 'save as' dialog for the user to select the output file location.
    Exits with code 0 and explanation if any step produces no usable data.
    """
    from .rag import chunk_document, load_policy_prompt
    import json
    import os
    import sys
    from datetime import datetime
    # 1. Chunk document (OCR, audio, etc. handled in chunk_document)
    chunks = chunk_document(document_path)
    if not chunks or not any(c.strip() for c in chunks):
        print(f"[INFO] No usable text could be extracted from {document_path}. Exiting.")
        sys.exit(0)
    # 2. Load policy prompt
    if policy_prompt_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config'))
        policy_prompt_path = os.path.join(base_dir, 'policy_prompt.yaml')
    policy_prompt = load_policy_prompt(policy_prompt_path)
    if not policy_prompt or not policy_prompt.strip():
        print("[INFO] No usable policy prompt found. Exiting.")
        sys.exit(0)
    # 3. Load variable config for selected module
    if vars_json_path is None:
        vars_json_path = os.path.join(os.path.dirname(__file__), f'../modules/{module_key}_vars.json')
    if not os.path.exists(vars_json_path):
        print(f"[INFO] Variable config file not found: {vars_json_path}. Exiting.")
        sys.exit(0)
    with open(vars_json_path, 'r', encoding='utf-8') as f:
        try:
            logic_tree = json.load(f)
        except Exception:
            print(f"[INFO] Could not parse variable config: {vars_json_path}. Exiting.")
            sys.exit(0)
    if not logic_tree or not isinstance(logic_tree, dict):
        print(f"[INFO] No usable variable schema found in {vars_json_path}. Exiting.")
        sys.exit(0)
    var_list = list(logic_tree.keys())
    if not var_list:
        print(f"[INFO] No variables defined in schema. Exiting.")
        sys.exit(0)
    # 4. LLM: Evaluate document for variables, summary, and extra info
    extraction_prompt = f"""{policy_prompt}\n\nExtract as many of the following variables as possible from the document below. Output as a JSON object with variable names as keys.\nVariables: {var_list}\nDocument: {''.join(chunks)}"""
    llm_outputs = analyze_chunks([extraction_prompt], model=model, api_base=api_base, api_key=api_key, api_port=api_port, provider=provider)
    if not llm_outputs or not isinstance(llm_outputs, list) or not llm_outputs[0]:
        print("[INFO] LLM did not return usable output. Exiting.")
        sys.exit(0)
    llm_json = llm_outputs[0]['output'] if isinstance(llm_outputs[0], dict) else llm_outputs[0]
    try:
        llm_vars = json.loads(llm_json)
    except Exception:
        llm_vars = {}
    if not llm_vars or not isinstance(llm_vars, dict):
        print("[INFO] LLM did not extract any variables. Exiting.")
        sys.exit(0)
    # 5. Evaluate for extra (non-variable) important info
    extra_info_prompt = f"""{policy_prompt}\n\nBased on the document, list any additional information not classified into a variable but important for context."""
    extra_outputs = analyze_chunks([extra_info_prompt], model=model, api_base=api_base, api_key=api_key, api_port=api_port, provider=provider)
    extra_info = extra_outputs[0]['output'] if extra_outputs and isinstance(extra_outputs[0], dict) else (extra_outputs[0] if extra_outputs else None)
    if not extra_info or not str(extra_info).strip():
        print("[INFO] No extra information found by LLM. Exiting.")
        sys.exit(0)
    # 6. Determine missing/insufficient variables and prompt user for clarification
    missing_vars = []
    clarified_vars = {}
    for var, meta in logic_tree.items():
        val = llm_vars.get(var, "")
        default = meta.get('default', "")
        # If value is missing or matches default, prompt user with prefilled value
        if not val or val == default:
            prompt_val = default
            if val and val != default:
                prompt_val = val
            try:
                import readline
                def prefill():
                    readline.insert_text(str(prompt_val))
                    readline.redisplay()
                readline.set_startup_hook(prefill)
                user_input = input(f"{var} [{prompt_val}]: ")
                readline.set_startup_hook()
            except Exception:
                user_input = input(f"{var} [{prompt_val}]: ")
            user_input = user_input.strip()
            clarified_vars[var] = user_input if user_input else prompt_val
            missing_vars.append(var)
        else:
            clarified_vars[var] = val
    if not clarified_vars:
        print("[INFO] No variables could be clarified or provided. Exiting.")
        sys.exit(0)
    # 7. Save output JSON blocks
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H%M')
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.cache'))
    os.makedirs(cache_dir, exist_ok=True)
    default_name = f"{date_str}_{time_str}_{module_key}.json"
    if output_path is None:
        # Open a save as dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            output_path = filedialog.asksaveasfilename(
                title="Save output as...",
                defaultextension=".json",
                initialfile=default_name,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not output_path:
                print("[INFO] Save cancelled by user. Output not saved.")
                sys.exit(0)
        except Exception as e:
            # Fallback to .cache if dialog fails
            print(f"[WARNING] Could not open save dialog: {e}. Saving to .cache.")
            output_path = os.path.join(cache_dir, default_name)
    # Block 1: LLM summary/evaluation
    block1 = {'llm_summary': llm_json, 'extra_info': extra_info}
    # Block 2: Variable values
    block2 = clarified_vars
    # Block 3: Formatted narrative (LLM proofed)
    narrative_prompt = f"""{policy_prompt}\n\nUsing only the following variable values, write a single flowing narrative in first person, in compliance with all policy.\nVariables: {json.dumps(clarified_vars, ensure_ascii=False)}"""
    narrative_outputs = analyze_chunks([narrative_prompt], model=model, api_base=api_base, api_key=api_key, api_port=api_port, provider=provider)
    narrative = narrative_outputs[0]['output'] if narrative_outputs and isinstance(narrative_outputs[0], dict) else (narrative_outputs[0] if narrative_outputs else None)
    if not narrative or not str(narrative).strip():
        print("[INFO] LLM did not generate a narrative. Exiting.")
        sys.exit(0)
    # Proof narrative
    proof_prompt = f"Proofread and improve the following narrative for syntax, readability, spelling, and content.\n\n{narrative}"
    proof_outputs = analyze_chunks([proof_prompt], model=model, api_base=api_base, api_key=api_key, api_port=api_port, provider=provider)
    proofed_narrative = proof_outputs[0]['output'] if proof_outputs and isinstance(proof_outputs[0], dict) else (proof_outputs[0] if proof_outputs else None)
    if not proofed_narrative or not str(proofed_narrative).strip():
        print("[INFO] LLM did not generate a proofed narrative. Exiting.")
        sys.exit(0)
    # Write JSON with three blocks
    output = {
        'block1': block1,
        'block2': block2,
        'block3': proofed_narrative
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Output saved to {output_path}")
    return output
