import json
"""
Generic dynamic module processor for any _vars.json config.
Handles all variable resolution and output logic for interview, closing, and assessment modules.
"""

import os
from pathlib import Path
from llm_db import get_llm_variable, set_llm_variable
from config import load_config

# Load abbreviation reference from JSON so it can be modified at runtime
ABBREVIATION_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "abbreviation_reference.json")
if os.path.exists(ABBREVIATION_REFERENCE_PATH):
    with open(ABBREVIATION_REFERENCE_PATH, "r", encoding="utf-8") as f:
        ABBREVIATION_REFERENCE = json.load(f)
else:
    ABBREVIATION_REFERENCE = {}

def resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=None):
    import sys
    resolved = {}
    provided = provided or {}
    for var, meta in logic_tree.items():
        val = None
        if var in provided and provided[var] not in (None, ""):
            val = provided[var]
        if not val:
            val = db_lookup(str(lookup_id), var)
        if not val:
            # Prompt user with hint and autofill with default, using readline for editable autofill if available
            hint = meta.get("hint", "")
            default = meta.get("default", "")
            prompt_str = f"{var} ({hint}): "
            user_input = None
            try:
                import readline
                def prefill():
                    readline.insert_text(str(default))
                    readline.redisplay()
                readline.set_startup_hook(prefill)
                try:
                    user_input = input(prompt_str)
                finally:
                    readline.set_startup_hook()
            except Exception:
                # Fallback if readline is not available
                user_input = input(f"{prompt_str}{default if default else ''}")
            user_input = user_input.strip()
            val = user_input if user_input else default
        resolved[var] = val
    return resolved

def dynamic_module_output(lookup_id=None, output_path=None, module_key=None, provided_data=None):
    """
    Generic output function for any module type (e.g., intv_adult, close_dispo, homeassess, etc.)
    module_key: e.g. 'intv_adult', 'close_dispo', etc.
    """
    if not module_key:
        raise ValueError("module_key is required (e.g., 'intv_adult', 'close_dispo', etc.)")
    config_path = os.path.join(os.path.dirname(__file__), f"{module_key}_vars.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        logic_tree = json.load(f)
    def db_lookup(lookup_id, var):
        return get_llm_variable(lookup_id, var)
    def user_prompt(var, hint):
        default = logic_tree.get(var, {}).get('default', '')
        display_val = f" [{default}]" if default else ""
        prompt = f"No value found for '{var}'{f' ({hint})' if hint else ''}.{display_val} Enter value (or press Enter to keep {default!r}): " if default else f"No value found for '{var}'{f' ({hint})' if hint else ''}. Enter value: "
        val = input(prompt)
        if val.strip() == '' and default:
            return default
        return val
    resolved = resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=provided_data)
    for k, v in resolved.items():
        set_llm_variable(str(lookup_id), k, v)
    # Add abbreviation reference to the top of the narrative for LLM context, only if not empty
    abbrev_lines = [f"{abbr} = {meaning}" for abbr, meaning in ABBREVIATION_REFERENCE.items()]
    abbrev_reference_text = "Abbreviation Reference: " + "; ".join(abbrev_lines) if abbrev_lines else ""
    # Build narrative (simple concatenation of all resolved values with section headers, no hints, no curly braces, and use real line breaks)
    Tmp = ""
    clarification_needed = False
    pending_questions = []
    for var, meta in logic_tree.items():
        assessed = resolved.get(var)
        section = meta.get("hint", var)
        # Remove curly braces from variable values
        if isinstance(assessed, str):
            assessed = assessed.replace("{", "").replace("}", "")
        # Replace literal '\n' and HTML <br> with real line breaks, and remove stray //a> artifacts
        if isinstance(assessed, str):
            assessed = assessed.replace("\\n", "\n").replace("<br>", "\n").replace("//a>", "")
        if not assessed:
            clarification_needed = True
            pending_questions.append(f"Please provide a value for {section}.")
        else:
            Tmp += f"{var}: {assessed}\n"
    # Compose the narrative with abbreviation reference at the top, no extra separating lines
    if abbrev_reference_text:
        narrative = f"{abbrev_reference_text}\n" + Tmp.strip()
    else:
        narrative = Tmp.strip()
    # Remove duplicate prompts (e.g., 'Name: [No Name]Name: [No Name]')
    import re
    narrative = re.sub(r'(\b\w+: [^\n]+)\1+', r'\1', narrative)
    result = {
        "status": "pending" if clarification_needed else "success",
        # Use ensure_ascii=False and indent=2 for pretty output, and keep real line breaks
        "narrative": narrative,
        "clarification_needed": clarification_needed,
        "pending_questions": pending_questions,
        "is_final": not clarification_needed
    }
    # Store initial output to .cache directory with a custom name: {yyyy-mm-dd}_{hhmm}_{interview_type}_{name}.json
    from datetime import datetime
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')  # yyyy-mm-dd
    time_str = now.strftime('%H%M')    # hhmm 24hr
    interview_type = module_key
    # Try to extract the 'Name' field from the resolved variables (if present)
    name_val = resolved.get('Name') or resolved.get('name') or 'noname'
    # Clean name for filename (remove spaces, special chars)
    import re
    name_val = re.sub(r'[^A-Za-z0-9_-]', '', str(name_val).replace(' ', '_'))
    cache_dir = os.path.join(os.path.dirname(__file__), '../../.cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{date_str}_{time_str}_{interview_type}_{name_val}.json"
    cache_path = os.path.join(cache_dir, cache_filename)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Output also cached to {cache_path}")
    # Prompt user to save output if not running in GUI mode
    import sys
    if not getattr(sys, 'ps1', False):  # Not interactive shell
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            save_path = filedialog.asksaveasfilename(
                title="Save output as...",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if save_path:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"[INFO] Output saved to {save_path}")
        except Exception as e:
            print(f"[WARNING] Could not open save dialog: {e}")
    return result
