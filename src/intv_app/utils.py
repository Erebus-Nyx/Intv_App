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
VALID_FILETYPES = {'.pdf', '.docx', '.txt'}
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

# --- File Validation ---
def is_valid_filetype(file_path: str) -> bool:
    """Check if the file type is valid for RAG processing."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in VALID_FILETYPES:
        return True
    mime, _ = mimetypes.guess_type(file_path)
    if mime and (mime.startswith('application/pdf') or mime.startswith('application/msword') or mime.startswith('text/')):
        return True
    return False

def validate_file(file_path: str, max_size_mb: int = 50) -> None:
    """Raise if file is invalid or too large."""
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
def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Chunk text with overlap."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def chunk_document(document_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
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
    from intv_app.utils import validate_file, chunk_document
    validate_file('mydoc.pdf')
    chunks = chunk_document('mydoc.pdf')
"""
