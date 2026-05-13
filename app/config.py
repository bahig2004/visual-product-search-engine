from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = DATA_DIR / "vectors"

PRODUCTS_FILE = DATA_DIR / "products.json"
FAISS_INDEX_FILE = VECTOR_DIR / "products_text.index"
METADATA_FILE = VECTOR_DIR / "products_metadata.json"

RAG_DOCUMENTS_FILE = BASE_DIR / "processed" / "rag" / "documents.json"
IMAGES_DIR = BASE_DIR / "images"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
FUZZY_THRESHOLD = 75
