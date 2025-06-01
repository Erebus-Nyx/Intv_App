import argparse
from pathlib import Path
import datetime
import os

def read_data(file_path):
    """Read data from a text file."""
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def process_data(data):
    """Process the data and format it."""
    formatted_data = []
    for line in data:
        # Example processing: strip whitespace and format
        formatted_line = line.strip().title()  # Capitalize each word
        formatted_data.append(formatted_line)
    return formatted_data

def save_output(output, output_file):
    """Save the formatted output to a file."""
    with open(output_file, 'w') as file:
        for line in output:
            file.write(f"{line}\n")

def get_gender_pronouns(cb_gender: str):
    """Return tuple of pronouns based on gender string."""
    cb_gender = str(cb_gender).strip().capitalize()
    if cb_gender == "Male":
        return ("He", "he", "His", "him", "his")
    elif cb_gender == "Female":
        return ("She", "she", "Her", "her", "her")
    else:
        return ("They", "they", "Their", "them", "their")

def get_first_name(full_name: str) -> str:
    """Extract first (and possibly third) name from full name string."""
    parts = str(full_name).strip().split()
    if len(parts) > 2:
        return f"{parts[0]} {parts[2]}"
    elif len(parts) > 0:
        return parts[0]
    else:
        return "[No Name]"

def is_valid_filetype(file_path: Path, filetype: str) -> bool:
    return file_path.suffix.lower().lstrip('.') == filetype.lower()

def chunk_document(file_path: Path, filetype: str):
    # Example chunking logic for demonstration
    if filetype == 'txt':
        with file_path.open('r', encoding='utf-8') as f:
            text = f.read()
        # Simple paragraph chunking
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    elif filetype == 'pdf':
        try:
            import PyPDF2
        except ImportError:
            raise ImportError('PyPDF2 is required for PDF chunking. Please install it.')
        with file_path.open('rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or '' for page in reader.pages)
        # If text is empty or too short, try OCR
        if not text.strip() or len(text.strip()) < 50:
            try:
                from pdf2image import convert_from_path
                import pytesseract
                from src.llm_db import get_module_variable_hints
            except ImportError:
                raise ImportError('pdf2image and pytesseract are required for OCR. Please install them.')
            images = convert_from_path(str(file_path))
            ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
            # Dynamically build relevant keywords from all variable hints (for logging/diagnostics only)
            hints = get_module_variable_hints()
            keywords = set()
            for module_vars in hints.values():
                for var, meta in module_vars.items():
                    if isinstance(meta, dict) and 'hint' in meta:
                        for word in meta['hint'].split():
                            if len(word) > 3:
                                keywords.add(word.lower().strip('.,:;()'))
            found_keywords = [kw for kw in keywords if kw in ocr_text.lower()]
            if found_keywords:
                print(f"[INFO] OCR detected possible relevant keywords: {found_keywords}")
            else:
                print("[INFO] OCR did not match any known variable hints, but content will still be passed to the reasoning model.")
            text = ocr_text
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    elif filetype == 'docx':
        try:
            import docx
        except ImportError:
            raise ImportError('python-docx is required for DOCX chunking. Please install it.')
        doc = docx.Document(str(file_path))
        text = "\n".join([para.text for para in doc.paragraphs])
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    raise NotImplementedError(f"Chunking for {filetype} not implemented.")

def get_default_transcribed_filename(extension="txt"):
    """Generate a safe default filename for transcribed output in the output directory."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    filename = f"transcribed - {now}.{extension}"
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.abspath(os.path.join(output_dir, filename))

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process and format text data.')
    parser.add_argument('input_file', type=str, help='Path to the input text file')
    parser.add_argument('output_file', type=str, help='Path to save the formatted output')
    
    args = parser.parse_args()
    
    # Read data from the input file
    data = read_data(args.input_file)
    
    # Process the data
    formatted_data = process_data(data)
    
    # Save the output to the specified file
    save_output(formatted_data, args.output_file)
    
    print(f"Formatted data saved to {args.output_file}")

if __name__ == "__main__":
    main()
