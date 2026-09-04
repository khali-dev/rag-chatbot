from src.config import(
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    OLLAMA_MODEL,
    TOP_K,
    create_data_directories
)

def test_chunk_configuration()-> None:
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert CHUNK_OVERLAP < CHUNK_SIZE

def test_retrieval_configuration() -> None:
    assert TOP_K


def test_ollama_model_configuration() -> None:
    assert OLLAMA_MODEL == "qwen3:4b"


def test_data_directories() -> None:
    create_data_directories

    assert DOCUMENTS_DIR.exists()
    assert DOCUMENTS_DIR.is_dir()
    assert CHROMA_DIR.exists()
    assert CHROMA_DIR.is_dir()
    