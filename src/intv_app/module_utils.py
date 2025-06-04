import os
import json

def get_available_interview_types(modules_dir=None):
    """
    List available interview/module types by scanning for _vars.json files in the modules directory.
    Returns a list of dicts with 'key' and 'display' for each module.
    """
    if modules_dir is None:
        modules_dir = os.path.join(os.path.dirname(__file__), '../modules')
    types = []
    for fname in os.listdir(modules_dir):
        if fname.endswith('_vars.json'):
            key = fname.replace('_vars.json', '')
            with open(os.path.join(modules_dir, fname), 'r', encoding='utf-8') as f:
                try:
                    schema = json.load(f)
                except Exception:
                    schema = {}
            display = schema.get('display', key)
            types.append({'key': key, 'display': display})
    return types

def detect_filetype_from_extension(filepath):
    """
    Infer file type from extension. Returns 'pdf', 'docx', 'txt', or 'unknown'.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.pdf', '.docx', '.txt']:
        return ext[1:]
    # Add more logic as needed
    return 'unknown'

def load_variable_schema(module_key, modules_dir=None):
    """
    Load and validate variable schema for a given module key from the modules directory.
    Returns the loaded JSON schema as a dict.
    """
    if modules_dir is None:
        modules_dir = os.path.join(os.path.dirname(__file__), '../modules')
    path = os.path.join(modules_dir, f'{module_key}_vars.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_default_value(var, schema):
    """
    Get default value for a variable from schema dict.
    Returns the default value or empty string if not set.
    """
    return schema.get(var, {}).get('default', '')

def prompt_for_variable(var, default_val):
    """
    Prompt user for a variable, prefilled with default value if available.
    Uses readline for prefill if available.
    """
    try:
        import readline
        def prefill():
            readline.insert_text(str(default_val))
            readline.redisplay()
        readline.set_startup_hook(prefill)
        user_input = input(f"{var} [{default_val}]: ")
        readline.set_startup_hook()
    except Exception:
        user_input = input(f"{var} [{default_val}]: ")
    return user_input.strip() if user_input.strip() else default_val
