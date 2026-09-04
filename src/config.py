from pathlib import Path


# Hauptverzeichnis des Projekts
BASE_DIR = Path(__file__).resolve().parent.parent

# Datenverzeichnisse
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"

# Sprachmodell
OLLAMA_MODEL = "qwen3:4b"

# Embedding-Modell
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Einstellungen für die Textaufteilung
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Anzahl der Suchergebnisse
TOP_K = 4


def create_data_directories() -> None:
    """Erstellt die benötigten Datenordner, falls sie nicht existieren."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)