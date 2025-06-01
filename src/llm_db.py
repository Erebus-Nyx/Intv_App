import sqlite3
from pathlib import Path
from typing import Any, Optional

# YAML support removed; all config is now JSON only

DB_PATH = Path(__file__).parent / 'llm_vars.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS llm_variables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lookup_id TEXT,
            var_name TEXT,
            var_value TEXT,
            UNIQUE(lookup_id, var_name)
        )
    ''')
    conn.commit()
    conn.close()

def set_llm_variable(lookup_id: str, var_name: str, var_value: Any):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO llm_variables (lookup_id, var_name, var_value)
        VALUES (?, ?, ?)
        ON CONFLICT(lookup_id, var_name) DO UPDATE SET var_value=excluded.var_value
    ''', (lookup_id, var_name, str(var_value)))
    conn.commit()
    conn.close()

def get_llm_variable(lookup_id: str, var_name: str) -> Optional[str]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT var_value FROM llm_variables WHERE lookup_id=? AND var_name=?
    ''', (lookup_id, var_name))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def get_all_llm_variables(lookup_id: str) -> dict:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT var_name, var_value FROM llm_variables WHERE lookup_id=?
    ''', (lookup_id,))
    rows = c.fetchall()
    conn.close()
    return {name: value for name, value in rows}

def get_module_variable_hints() -> dict:
    """
    Returns a dictionary mapping module names to the variables they expect, with hints and default values for LLM reference.
    """
    # Example: Extend this as you add more modules and variables
    return {
        'doc_intv_child': {
            'lookup_id': {'hint': 'Unique identifier for the interview/case', 'default': None},
            'narrative': {'hint': 'Full narrative output for the child interview', 'default': ''},
            'NarrativeIntro': {'hint': 'HTML-formatted introduction section', 'default': ''},
            'NarrativeObservation': {'hint': 'Observation section of the interview', 'default': ''},
            'NarrativeRapport': {'hint': 'Rapport-building section', 'default': ''},
            'NarrativeScreen': {'hint': 'Screening and background section', 'default': ''},
            'NarrativeAllegation': {'hint': 'Allegation details section', 'default': ''},
            'ChildFirst': {'hint': 'First name of the child', 'default': '[No Name]'},
            'g1': {'hint': 'Subjective pronoun (he/she/they)', 'default': 'They'},
            'g1a': {'hint': 'Subjective pronoun, lowercase', 'default': 'they'},
            'g2': {'hint': 'Possessive pronoun (his/her/their)', 'default': 'Their'},
            'g2a': {'hint': 'Possessive pronoun, lowercase', 'default': 'their'},
            'g3a': {'hint': 'Age-appropriate pronoun', 'default': 'their'},
        },
        # Add other modules and their variable hints/defaults here
    }

def get_needed_variables_for_module(module_name: str) -> dict:
    """
    Returns a dict of variable names, hints, and defaults for a given module.
    """
    hints = get_module_variable_hints()
    return hints.get(module_name, {})

def get_all_needed_variables() -> dict:
    """
    Returns a dictionary of all modules and their needed variables with hints.
    """
    return get_module_variable_hints()

# Deprecated: YAML defaults are no longer supported. All config defaults must be in JSON.

# For LLM: Utility to show missing variables for a module/lookup_id

def get_missing_variables_for_module(module_name: str, lookup_id: str) -> dict:
    """
    Returns a dict of variable names, hints, and defaults that are needed for a module but not yet set for a given lookup_id.
    """
    needed = get_needed_variables_for_module(module_name)
    existing = get_all_llm_variables(lookup_id)
    missing = {k: v for k, v in needed.items() if k not in existing}
    return missing

init_db()
