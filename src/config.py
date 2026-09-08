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

# Auswahl des Sprachmodell-Anbieters:
# ollama = lokaler Betrieb
# gemini = Cloud-Betrieb
LLM_PROVIDER = (
    os.getenv("LLM_PROVIDER", "ollama")
    .strip()
    .casefold()
)

SUPPORTED_LLM_PROVIDERS = {
    "ollama",
    "gemini",
}

# Lokales Sprachmodell
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b",
).strip()

# Cloud-Sprachmodell
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
).strip()

# Optionaler lokaler Gemini-Schlüssel.
# In der öffentlichen App wird der vom jeweiligen
# Besucher eingegebene Schlüssel verwendet.
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

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


def validate_configuration() -> None:
    """Prüft die Anwendungskonfiguration."""

    if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
        supported_values = ", ".join(
            sorted(SUPPORTED_LLM_PROVIDERS)
        )

        raise ValueError(
            "Unbekannter LLM_PROVIDER "
            f"'{LLM_PROVIDER}'. Erlaubt sind: "
            f"{supported_values}."
        )

    if not OLLAMA_MODEL:
        raise ValueError(
            "OLLAMA_MODEL darf nicht leer sein."
        )

    if not GEMINI_MODEL:
        raise ValueError(
            "GEMINI_MODEL darf nicht leer sein."
        )


def is_cloud_mode() -> bool:
    """Gibt an, ob der Gemini-Cloudbetrieb aktiv ist."""

    return LLM_PROVIDER == "gemini"


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