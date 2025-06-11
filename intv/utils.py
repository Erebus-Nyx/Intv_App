import os
import mimetypes
from typing import List
from PyPDF2 import PdfReader
import logging

try:
    import docx
except ImportError:
    docx = None

# --- Logging Setup ---
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.logs'))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'rag_pipeline.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Configurable Parameters ---
VALID_FILETYPES = {'.pdf', '.docx', '.txt', '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.mp4', '.m4v', '.mov', '.avi', '.wmv', '.flv', '.webm'}

# --- File Validation ---
def is_valid_filetype(file_path: str) -> bool:
    """
    Check if the file type is valid for processing.
    Accepts PDF, DOCX, TXT, audio, and video files for different processing pipelines.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in VALID_FILETYPES:
        return True
    mime, _ = mimetypes.guess_type(file_path)
    if mime and (mime.startswith('application/pdf') or 
                 mime.startswith('application/msword') or 
                 mime.startswith('text/') or 
                 mime.startswith('audio/') or
                 mime.startswith('video/')):
        return True
    return False

def validate_file(file_path: str, max_size_mb: int = 1000) -> None:
    """
    Raise if file is invalid or too large.
    Logs errors for missing, invalid, or oversized files.
    """
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    if not is_valid_filetype(file_path):
        logging.error(f"Invalid file type: {file_path}")
        raise ValueError(f"Invalid file type: {file_path}")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        logging.error(f"File too large: {file_path} ({size_mb:.2f} MB)")
        raise ValueError(f"File too large: {file_path} ({size_mb:.2f} MB)")

# --- Chunking ---
def chunk_text(text: str, chunk_size: int = None, overlap: int = None, config=None) -> List[str]:
    """
    Chunk text with overlap for RAG processing.
    Returns a list of text chunks, each of size chunk_size with overlap.
    """
    # Use config-driven settings
    if config is None:
        try:
            import sys
            import os
            # Add src directory to path for config import
            src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from config import load_config
            config = load_config()
        except Exception as e:
            # Fallback to default values if config loading fails
            config = {
                'chunk_size': 1000,
                'chunk_overlap': 100
            }
    if chunk_size is None:
        chunk_size = config.get('chunk_size', 1000)
    if overlap is None:
        overlap = config.get('chunk_overlap', 100)

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def chunk_document(document_path: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split a document (PDF, DOCX, TXT) into text chunks for LLM evaluation.
    Returns a list of text chunks.
    """
    ext = os.path.splitext(document_path)[1].lower()
    text = ""
    if ext == '.pdf':
        reader = PdfReader(document_path)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
    elif ext == '.docx' and docx is not None:
        doc = docx.Document(document_path)
        text = "\n".join(para.text for para in doc.paragraphs)
    elif ext == '.txt':
        with open(document_path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type for chunking: {ext}")
    return chunk_text(text, chunk_size, overlap)

# --- Example Test Function ---
def test_rag_pipeline():
    """Test the RAG pipeline with a sample document and query."""
    import tempfile
    sample_text = "This is a test document.\nIt has several lines.\nThe quick brown fox jumps over the lazy dog."
    with tempfile.NamedTemporaryFile('w+', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        f.flush()
        path = f.name
    try:
        validate_file(path)
        chunks = chunk_document(path)
        assert len(chunks) > 0
    finally:
        os.remove(path)

# --- Documentation Example ---
"""
Usage:
    from intv.utils import validate_file, chunk_document
    validate_file('mydoc.pdf')
    chunks = chunk_document('mydoc.pdf')
"""
