from pathlib import Path


# Hauptverzeichnis des Projekts
BASE_DIR = Path(__file__).resolve().parent.parent

# Datenverzeichnisse
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"

# ChromaDB
CHROMA_COLLECTION_NAME = "rag_documents"

# Lokales Sprachmodell
OLLAMA_MODEL = "qwen3:4b"

# Einstellungen für die Antwortgenerierung
LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 4096

# Embedding-Modell
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Textaufteilung
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4
MIN_SIMILARITY = 0.25
MAX_CONTEXT_CHARS = 4000

# Datei-Upload
MAX_UPLOAD_SIZE_MB = 20


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