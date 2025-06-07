from .utils import is_valid_filetype, validate_file
from PyPDF2 import PdfReader
import os
from typing import List, Optional, Callable
from PIL import Image
import pytesseract
import logging
import yaml
from .ocr import ocr_file, ocr_pdf_page
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

try:
    import docx
except ImportError:
    docx = None

# --- RAG Pipeline Core Functions ---
# These functions handle document chunking, prompt loading, and RAG orchestration.

# 2. Set default chunk size/overlap and logging config
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.logs'))
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.logs'))
        os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'rag_pipeline.log')

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 3. Default policy prompt
DEFAULT_POLICY_PROMPT = """You are a helpful assistant. Provide a concise and accurate response based on the context.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"""

# 4. Load policy prompt from YAML (already present, just ensure import and fallback)
def load_policy_prompt(yaml_path=None):
    """
    Load the policy prompt from policy_prompt.yaml for RAG/LLM continuity.
    If not found or invalid, falls back to DEFAULT_POLICY_PROMPT.
    """
    if yaml_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config'))
        yaml_path = os.path.join(base_dir, 'policy_prompt.yaml')
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'prompt' in data:
                return data['prompt']
            elif isinstance(data, str):
                return data
            else:
                raise ValueError('policy_prompt.yaml must contain a string or a dict with a "prompt" key.')
    except Exception as e:
        logging.warning(f"Could not load policy prompt from {yaml_path}: {e}. Using default.")
        return DEFAULT_POLICY_PROMPT

# 5. Chunking with overlap

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

# 6. Document chunking with OCR for images and image-based PDFs

def chunk_document(document_path: str, chunk_size: int = None, overlap: int = None, config=None) -> List[str]:
    """
    Split a document (PDF, DOCX, TXT, or image) into text chunks for LLM evaluation.
    Returns a list of text chunks.
    Automatically applies OCR for image files and image-based PDFs.
    """
    # Use config-driven settings
    if config is None:
        try:
            import sys
            # Add src directory to path for config import
            src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from config import load_config
            config = load_config()
        except Exception as e:
            # Fallback to default values if config loading fails
            config = {
                'chunk_size': DEFAULT_CHUNK_SIZE,
                'chunk_overlap': DEFAULT_CHUNK_OVERLAP
            }
    if chunk_size is None:
        chunk_size = config.get('chunk_size', 500)
    if overlap is None:
        overlap = config.get('chunk_overlap', 50)

    ext = os.path.splitext(document_path)[1].lower()
    text = ""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    if ext == '.pdf':
        reader = PdfReader(document_path)
        page_texts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            if not page_text.strip():
                # OCR fallback for image-based PDF page
                try:
                    page_text = ocr_pdf_page(document_path, i)
                except Exception as e:
                    logging.warning(f"OCR failed for PDF page {i+1}: {e}")
            page_texts.append(page_text)
        text = "\n".join(page_texts)
    elif ext == '.docx' and docx is not None:
        doc = docx.Document(document_path)
        text = "\n".join(para.text for para in doc.paragraphs)
    elif ext == '.txt':
        with open(document_path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif ext in image_exts:
        text = ocr_file(document_path)
    else:
        raise ValueError(f"Unsupported file type for chunking: {ext}")
    return chunk_text(text, chunk_size, overlap)

# 7. Batch processing for chunking

def batch_chunk_documents(file_paths: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[List[str]]:
    """Batch process multiple documents for chunking."""
    all_chunks = []
    for path in file_paths:
        try:
            validate_file(path)
            chunks = chunk_document(path, chunk_size, overlap)
            all_chunks.append(chunks)
        except Exception as e:
            pass  # Logging can be added if needed
    return all_chunks

# --- Extensibility: Pluggable Retriever/LLM ---
def process_with_retriever_and_llm(
    chunks: List[str],
    query: str,
    retriever: Optional[Callable[[str, List[str]], List[str]]] = None,
    llm: Optional[Callable[[str, str], str]] = None,
    policy_prompt: Optional[str] = None,
    top_k: int = None,
    all_chunks: bool = False,
    config=None
) -> str:
    """
    Retrieve relevant chunks and pass to LLM for categorization/summarization.
    If all_chunks is True, process all chunks regardless of query relevance.
    """
    # Use config-driven settings
    if config is None:
        from config import load_config
        config = load_config()
    if top_k is None:
        top_k = config.get('rag_top_k', 5)
    if policy_prompt is None:
        policy_prompt = load_policy_prompt()
    if all_chunks:
        relevant_chunks = chunks
    else:
        # Retrieval (simple keyword search if no retriever)
        if retriever:
            relevant_chunks = retriever(query, chunks)
        else:
            words = set(query.lower().split())
            scored = [(sum(w in c.lower() for w in words), c) for c in chunks]
            scored.sort(reverse=True)
            relevant_chunks = [c for score, c in scored[:top_k] if score > 0]
            if not relevant_chunks:
                relevant_chunks = chunks[:top_k]
    context = "\n---\n".join(relevant_chunks)
    prompt = policy_prompt.format(context=context, query=query)
    # LLM call
    if llm:
        result = llm(prompt, context)
    else:
        result = f"[LLM output would go here for prompt: {prompt[:200]}...]"
    return result

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
        from .utils import validate_file
        validate_file(path)
        chunks = chunk_document(path)
        assert len(chunks) > 0
        result = process_with_retriever_and_llm(chunks, "What does the document say?", policy_prompt="Summarize:")
        print("Test result:", result)
    finally:
        os.remove(path)

def ensure_hf_prefix(model_name):
    """Ensure the model string starts with 'hf.co/' if it looks like a HuggingFace repo."""
    if not isinstance(model_name, str):
        return model_name
    if model_name.startswith('hf.co/'):
        return model_name
    if '/' in model_name:
        return f"hf.co/{model_name}"
    return model_name

def download_with_progress(url, dest_path):
    import requests
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = int(downloaded * 100 / total) if total else 0
                bar = ('#' * (percent // 2)).ljust(50)
                print(f"\r[DOWNLOAD] |{bar}| {percent}%", end='')
    print(f"\n[INFO] Download complete: {dest_path}")

def ensure_rag_model_downloaded(rag_model, model_dir):
    import os
    if not rag_model:
        return
    # Accept both 'hf.co/author/repo' and 'author/repo' formats
    if rag_model.startswith('hf.co/'):
        model_str = rag_model[len('hf.co/'):]
    else:
        model_str = rag_model
    # Only handle HuggingFace models
    if '/' not in model_str:
        return
    repo_id = model_str
    local_dir = model_dir
    os.makedirs(local_dir, exist_ok=True)
    # ONNX model support: use SentenceTransformer for loading
    if 'onnx' in rag_model.lower():
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Downloading/loading ONNX RAG model: {repo_id}")
            model = SentenceTransformer(rag_model)
            print(f"[INFO] Downloaded and loaded ONNX RAG model: {repo_id}")
            return model
        except Exception as e:
            print(f"[ERROR] Could not download or load ONNX RAG model {repo_id}: {e}")
            raise
    # Default: PyTorch model
    try:
        from transformers import AutoModel, AutoTokenizer
        AutoModel.from_pretrained(repo_id, cache_dir=local_dir)
        AutoTokenizer.from_pretrained(repo_id, cache_dir=local_dir)
        print(f"[INFO] Downloaded RAG model: {repo_id}")
    except Exception as e:
        print(f"[ERROR] Could not download RAG model {repo_id}: {e}")
        raise

def get_rag_model(config):
    # Use rag_model if set, otherwise fall back to llm_model
    rag_model = config.get('rag_model', '')
    if not rag_model:
        rag_model = config.get('llm_model', '')
    rag_model = ensure_hf_prefix(rag_model)
    model_dir = config.get('model_dir', 'models')
    ensure_rag_model_downloaded(rag_model, model_dir)
    return rag_model
