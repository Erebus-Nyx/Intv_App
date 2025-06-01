# doc_close_dispo.py
"""
Python equivalent of Doc_Close_Dispo.bas
Implements Compile_Dispo logic as a function (stub for now, expand as needed).
"""

import json
import os
from pathlib import Path
from src.llm_db import get_llm_variable, set_llm_variable
from src.config import load_config

def resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=None):
    resolved = {}
    provided = provided or {}
    for var, meta in logic_tree.items():
        val = None
        if var in provided and provided[var] not in (None, ""):
            val = provided[var]
        if not val:
            val = db_lookup(str(lookup_id), var)
        if not val:
            val = meta.get("default", "")
        if not val:
            val = user_prompt(var, meta.get("hint", ""))
        resolved[var] = val
    return resolved

def close_dispo_output(lookup_id=None, output_path=None, dispo_data=None):
    with open(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "close_dispo_vars.json"), "r", encoding="utf-8"
    ) as f:
        logic_tree = json.load(f)
    def db_lookup(lookup_id, var):
        return get_llm_variable(lookup_id, var)
    def user_prompt(var, hint):
        return input(f"No value found for '{var}' ({hint}). Please provide a value: ")
    resolved = resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=dispo_data)
    for k, v in resolved.items():
        set_llm_variable(str(lookup_id), k, v)
    def get_personal_reference(resolved, default="the worker"):
        first = resolved.get("Worker", "").strip()
        last = resolved.get("WorkerLastName", "").strip()
        if first and last:
            return f"{first} {last}"
        elif first:
            return first
        elif last:
            return f"Mr./Ms. {last}"
        return default
    person_ref = get_personal_reference(resolved, default="the worker")
    Tmp = ""
    def prompt_section(section_key, default_text):
        print(f"\nSection: {section_key}")
        print(f"Default: {default_text}")
        user_input = input(f"Enter narrative for {section_key} (leave blank to skip, Enter for default): ")
        if user_input.strip() == "":
            return default_text if default_text else None
        return user_input.strip()
    def interpolate(text, resolved):
        if not isinstance(text, str):
            return text
        for key in resolved:
            placeholder = f"{{{key.lower()}}}"
            if placeholder in text.lower():
                text = text.replace(f"{{{key}}}", str(resolved[key]))
                text = text.replace(f"{{{key.lower()}}}", str(resolved[key]))
                text = text.replace(f"{{{key.upper()}}}", str(resolved[key]))
        return text
    # Remove hardcoded section configs; build from logic_tree instead
    section_configs = []
    for var, meta in logic_tree.items():
        section_configs.append({
            "section": meta.get("hint", var),
            "assess_key": var,
            "default": meta.get("default", "")
        })
    clarification_needed = False
    pending_questions = []
    for config in section_configs:
        assessed = resolved.get(config['assess_key'])
        default_text = interpolate(config['default'], resolved)
        if not assessed:
            clarification_needed = True
            pending_questions.append(f"Please provide a value for {config['section']}.")
        else:
            narrative = prompt_section(config['section'], default_text)
            if narrative:
                Tmp += narrative + "  "
    result = {
        "status": "pending" if clarification_needed else "success",
        "narrative": Tmp.strip(),
        "clarification_needed": clarification_needed,
        "pending_questions": pending_questions,
        "is_final": not clarification_needed
    }
    if output_path:
        output_path = Path(output_path)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result
