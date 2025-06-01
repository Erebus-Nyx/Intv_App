# doc_homeassess.py
"""
Python equivalent of Doc_HomeAssess.bas
Implements Generate_HomeAssess logic as a function (stub for now, expand as needed).
"""

import json
import os
from pathlib import Path
from src.llm_db import get_llm_variable, set_llm_variable
from src.config import load_config

def resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=None):
    """Resolve all variables for a module using DB, provided data, defaults, and user prompt."""
    resolved = {}
    provided = provided or {}
    for var, meta in logic_tree.items():
        val = None
        # 1. Provided data (e.g., from CLI or API)
        if var in provided and provided[var] not in (None, ""):
            val = provided[var]
        # 2. DB lookup
        if not val:
            val = db_lookup(str(lookup_id), var)
        # 3. Default from logic tree
        if not val:
            val = meta.get("default", "")
        # 4. Prompt user if still missing
        if not val:
            val = user_prompt(var, meta.get("hint", ""))
        resolved[var] = val
    return resolved

def homeassess_output(lookup_id=None, output_path=None, assess_data=None):
    # Load logic tree from JSON
    with open(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "homeassess_vars.json"), "r", encoding="utf-8"
    ) as f:
        logic_tree = json.load(f)
    def db_lookup(lookup_id, var):
        return get_llm_variable(lookup_id, var)
    def user_prompt(var, hint):
        return input(f"No value found for '{var}' ({hint}). Please provide a value: ")
    # Resolve all variables
    resolved = resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=assess_data)
    # Save all resolved variables to DB
    for k, v in resolved.items():
        set_llm_variable(str(lookup_id), k, v)
    Tmp = ""
    # Home assessment logic (unchanged, but now uses resolved[])
    def get_personal_reference(resolved, default="the resident"):
        """Return a personal reference string based on available name fields."""
        first = resolved.get("txt_Name", "").strip()
        last = resolved.get("txt_LastName", "").strip()
        if first and last:
            return f"{first} {last}"
        elif first:
            return first
        elif last:
            return f"Mr./Ms. {last}"
        return default

    person_ref = get_personal_reference(resolved, default="the resident")
    child_ref = resolved.get("txt_ChildName", "the child")
    if resolved.get("cb_AssessHome") == "Consented":
        Tmp = f"<br><br> House Walkthrough <br> Permission was granted during the home visit to see the residence "
        if resolved.get("cb_PhotographHome") == "Consented":
            Tmp += "and to take photos. The walkthrough and photos are summarized as follows:  <br><br>"
        elif resolved.get("cb_PhotographHome") == "Refused":
            Tmp += "but was not permitted to take photos. The observations during the walkthrough are summarized as follows:  <br><br>"
    elif resolved.get("cb_PhotoHome") == "Refused":
        Tmp = f"<Br><br>House Walkthrough <Br> Permission to see the residence was refused during the home visit. Any observations made during the interview are summarized as follows:  <br><br>"
    # Room-by-room assessment
    def prompt_section(section_key, default_text):
        """Prompt user for a section narrative, prepopulated with default. Blank = not assessed."""
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
    HomeAssessNarrative = Tmp
    result = {
        "status": "success",
        "narrative": HomeAssessNarrative.strip(),
        # Pronouns and name logic removed; LLM will handle pronouns contextually
    }
    if output_path:
        output_path = Path(output_path)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result
