# doc_allegations.py
"""
Python equivalent of Allegation array logic from Def_GlobalVars.bas and related forms.
Provides helper functions for allegation text lookup and formatting.
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

def allegations_output(lookup_id=None, output_path=None, allegations_data=None):
    with open(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "allegations_vars.json"), "r", encoding="utf-8"
    ) as f:
        logic_tree = json.load(f)
    def db_lookup(lookup_id, var):
        return get_llm_variable(lookup_id, var)
    def user_prompt(var, hint):
        return input(f"No value found for '{var}' ({hint}). Please provide a value: ")
    resolved = resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=allegations_data)
    for k, v in resolved.items():
        set_llm_variable(str(lookup_id), k, v)
    def get_personal_reference(resolved, default="the individual"):
        first = resolved.get("Victim", "").strip()
        last = resolved.get("VictimLastName", "").strip()
        if first and last:
            return f"{first} {last}"
        elif first:
            return first
        elif last:
            return f"Mr./Ms. {last}"
        return default
    person_ref = get_personal_reference(resolved, default="the individual")
    Tmp = ""
    def prompt_section(section_key, default_text):
        print(f"\nSection: {section_key}")
        print(f"Default: {default_text}")
        user_input = input(f"Enter narrative for {section_key} (leave blank to skip, Enter for default): ")
        if user_input.strip() == "":
            return default_text if default_text else None
        return user_input.strip()
    section_configs = [
        {"section": "Allegation Type", "assess_key": "AllegationType", "default": logic_tree.get("AllegationType", {}).get("default", "")},
        {"section": "Allegation Details", "assess_key": "AllegationDetails", "default": logic_tree.get("AllegationDetails", {}).get("default", "")},
        {"section": "Date", "assess_key": "Date", "default": logic_tree.get("Date", {}).get("default", "")},
        {"section": "Reporter", "assess_key": "Reporter", "default": logic_tree.get("Reporter", {}).get("default", "")},
        {"section": "Victim", "assess_key": "Victim", "default": logic_tree.get("Victim", {}).get("default", "")},
        {"section": "Perpetrator", "assess_key": "Perpetrator", "default": logic_tree.get("Perpetrator", {}).get("default", "")},
        {"section": "Location", "assess_key": "Location", "default": logic_tree.get("Location", {}).get("default", "")},
        {"section": "Action Taken", "assess_key": "ActionTaken", "default": logic_tree.get("ActionTaken", {}).get("default", "")}
    ]
    clarification_needed = False
    pending_questions = []
    for config in section_configs:
        assessed = resolved.get(config['assess_key'])
        if not assessed:
            clarification_needed = True
            pending_questions.append(f"Please provide a value for {config['section']}.")
        else:
            narrative = prompt_section(config['section'], config['default'])
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
