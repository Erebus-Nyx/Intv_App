"""
compliance_check.py - Automated checklist for module/config compliance

This script checks all modules and config JSONs in the project for compliance with the standards in the README checklist.
Run with: python src/compliance_check.py
"""
import os
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / 'config'
MODULES_DIR = Path(__file__).parent / 'modules'

# Required keys/sections for all config JSONs (customize as needed)
REQUIRED_KEYS = [
    'Name', 'Role', 'Location', 'Rapport', 'Observation', 'Family', 'Chores',
    'Allegation', 'Screening', 'Summary'
]

# Narrative requirements
NARRATIVE_RULES = [
    'first-person', 'professional', 'paragraph', 'no bullet points', 'no headers',
    'clear party identification', 'demographic info included',
    'no speculation', 'concise', 'formal'
]

def check_config_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    for k, v in data.items():
        if not isinstance(v, dict) or 'hint' not in v:
            print(f"[WARN] {path.name}: Variable '{k}' missing 'hint' or not a dict.")
    if missing:
        print(f"[FAIL] {path.name}: Missing required keys: {missing}")
    else:
        print(f"[OK]   {path.name}: All required keys present.")

def main():
    print("=== Compliance Check: Config JSONs ===")
    for f in CONFIG_DIR.glob('*.json'):
        check_config_json(f)
    print("\n=== Compliance Check: Narrative/Module Structure (manual review recommended) ===")
    print("- Ensure all modules output 'clarification_needed', 'pending_questions', and 'is_final' flags.")
    print("- Ensure narrative output is first-person, professional, paragraph/story form.")
    print("- Check for LLM policy prompt enforcement in all LLM calls.")
    print("- No pronoun/name logic in code; LLM handles context.")
    print("- All configs/modules are user-editable and reloadable at runtime.")
    print("- See README checklist for full requirements.")

if __name__ == '__main__':
    main()
