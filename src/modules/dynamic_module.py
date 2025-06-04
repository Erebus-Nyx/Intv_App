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

    # --- LLM Variable Extraction Step (with timeout) ---
    llm_vars = {}
    try:
        # Try to load document text from lookup_id (file path)
        doc_text = None
        if lookup_id and os.path.exists(lookup_id):
            ext = os.path.splitext(lookup_id)[1].lower()
            if ext == '.txt':
                with open(lookup_id, 'r', encoding='utf-8') as f:
                    doc_text = f.read()
            elif ext == '.docx':
                try:
                    import docx
                    doc = docx.Document(lookup_id)
                    doc_text = '\n'.join([p.text for p in doc.paragraphs])
                except Exception as e:
                    print(f"[WARNING] Could not read DOCX: {e}")
            elif ext == '.pdf':
                try:
                    import PyPDF2
                    with open(lookup_id, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        doc_text = '\n'.join(page.extract_text() or '' for page in reader.pages)
                except Exception as e:
                    print(f"[WARNING] Could not read PDF: {e}")
        # Only proceed if we have document text
        if doc_text:
            from intv_app.llm import analyze_chunks
            var_list = list(logic_tree.keys())
            extraction_prompt = (
                f"Extract as many of the following variables as possible from the document below. "
                f"Output as a JSON object with variable names as keys.\n"
                f"Variables: {var_list}\n"
                f"Document: {doc_text}"
            )
            # Use model and API info from environment or defaults
            model = os.environ.get('LLM_MODEL', 'hf.co/unsloth/Phi-4-reasoning-plus-GGUF:Q6_K_XL')
            api_base = os.environ.get('LLM_API_BASE', 'http://localhost')
            api_port = int(os.environ.get('LLM_API_PORT', '5001'))
            # Always use koboldcpp as provider
            provider = 'koboldcpp'
            print(f"[DEBUG] LLM extraction config: provider={provider}, api_base={api_base}, api_port={api_port}, model={model}")
            # Force api_port to 5001 if provider is koboldcpp and port is 5001
            if provider == 'koboldcpp' and str(api_port) == '5001':
                api_port = 5001
                print(f"[DEBUG] Overriding api_port to 5001 for koboldcpp backend.")
            import signal
            class TimeoutException(Exception):
                pass
            def handler(signum, frame):
                raise TimeoutException()
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60)  # 60 second timeout
            try:
                llm_output = analyze_chunks([extraction_prompt], model=model, api_base=api_base, api_port=api_port, provider=provider)
                signal.alarm(0)
            except TimeoutException:
                print("[ERROR] LLM extraction timed out after 60 seconds.")
                llm_output = None
            except Exception as e:
                print(f"[ERROR] LLM extraction failed: {e}")
                llm_output = None
            finally:
                signal.alarm(0)
            if llm_output:
                print("[DEBUG] Raw LLM extraction output:\n", llm_output)
                try:
                    import json as _json
                    llm_vars = _json.loads(llm_output)
                    print("[INFO] LLM-extracted variable values:")
                    for k, v in llm_vars.items():
                        print(f"  {k}: {v}")
                    # Print the first 5 variables and their values
                    first5 = list(llm_vars.keys())[:5]
                    print("[DEBUG] First 5 variables and their values from LLM:")
                    for k in first5:
                        print(f"  {k}: {llm_vars[k]}")
                except Exception as e:
                    print(f"[WARNING] Could not parse LLM output as JSON: {e}")
                    llm_vars = {}
    except Exception as e:
        print(f"[WARNING] LLM variable extraction failed: {e}")
        llm_vars = {}
    # Merge with provided_data if any
    if provided_data:
        llm_vars.update(provided_data)
    resolved = resolve_variables(lookup_id, logic_tree, db_lookup, user_prompt, provided=llm_vars)
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
            # Only add the variable label and value once, and avoid repeating if label and value are the same
            label = section.strip()
            if label.lower() == var.replace('_', ' ').strip().lower():
                Tmp += f"{label}: {assessed}\n"
            else:
                Tmp += f"{label}: {assessed}\n"
    # Compose the narrative with abbreviation reference at the top, no extra separating lines
    if abbrev_reference_text:
        narrative = f"{abbrev_reference_text}\n" + Tmp.strip()
    else:
        narrative = Tmp.strip()
    # Remove duplicate lines (e.g., 'Name: [No Name]\nName: [No Name]')
    import re
    lines = narrative.splitlines()
    seen = set()
    deduped_lines = []
    for line in lines:
        if line not in seen:
            deduped_lines.append(line)
            seen.add(line)
    narrative = "\n".join(deduped_lines)

    # --- Post-processing: Suppress prompts for variables present in narrative (fuzzy match) ---
    suppressed_questions = []
    filtered_pending_questions = []
    for q in pending_questions:
        # Try to extract the variable/section name from the question
        # Example: "Please provide a value for Name." -> "Name"
        match = re.search(r"for ([^\.]+)", q)
        if match:
            var_label = match.group(1).strip().lower()
            # Fuzzy match: check if var_label or a close variant appears in the narrative
            found = False
            for line in deduped_lines:
                if var_label in line.lower() or (var_label.replace(' ', '') in line.lower().replace(' ', '')):
                    found = True
                    break
            if found:
                suppressed_questions.append(q)
                continue
        filtered_pending_questions.append(q)
    if suppressed_questions:
        print(f"[INFO] Suppressed clarification prompts for variables already present: {suppressed_questions}")
    pending_questions = filtered_pending_questions
    clarification_needed = bool(pending_questions)
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
        sys.exit(0)
    # After result is built, check for new abbreviations from LLM output
    # If the LLM output (result) contains a key 'new_abbreviations', append them to abbreviation_reference.json
    new_abbrevs = result.get('new_abbreviations') if isinstance(result, dict) else None
    if new_abbrevs and isinstance(new_abbrevs, dict):
        abbrev_path = os.path.join(os.path.dirname(__file__), "abbreviation_reference.json")
        try:
            with open(abbrev_path, "r", encoding="utf-8") as f:
                abbrevs = json.load(f)
        except Exception:
            abbrevs = {}
        updated = False
        for k, v in new_abbrevs.items():
            if k not in abbrevs:
                abbrevs[k] = v
                updated = True
        if updated:
            with open(abbrev_path, "w", encoding="utf-8") as f:
                json.dump(abbrevs, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Appended new abbreviations to abbreviation_reference.json: {list(new_abbrevs.keys())}")
    return result
