import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# Hauptverzeichnis des Projekts
BASE_DIR = Path(__file__).resolve().parent.parent

# Datenverzeichnisse
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"
EVALUATION_DIR = DATA_DIR / "evaluation"

# ChromaDB
CHROMA_COLLECTION_NAME = "rag_documents"

# Lokales Sprachmodell
OLLAMA_MODEL = "qwen3:4b"

# Einstellungen für die Antwortgenerierung
LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 4096
OLLAMA_KEEP_ALIVE = "10m"

# Optionales vollständiges GPU-Offloading
OLLAMA_NUM_GPU_LAYERS = 36

# Embedding-Modell
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_BATCH_SIZE = 32

# Textaufteilung
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 3
MIN_SIMILARITY = 0.25
MAX_CONTEXT_CHARS = 4000

# Datei-Upload
MAX_UPLOAD_SIZE_MB = 20

# Technische Fehlerdetails
DEBUG_MODE = (
    os.getenv("RAG_DEBUG", "false")
    .strip()
    .casefold()
    in {"1", "true", "yes", "ja"}
)


def create_data_directories() -> None:
    """Erstellt die benötigten Datenordner."""

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )