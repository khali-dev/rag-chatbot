from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.text_splitter import DocumentChunk


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Lädt das Embedding-Modell und hält es im Arbeitsspeicher."""

    return SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(
    texts: Sequence[str],
) -> list[list[float]]:
    """Erzeugt normalisierte Embeddings für mehrere Texte."""

    if not texts:
        return []

    cleaned_texts = [text.strip() for text in texts]

    if any(not text for text in cleaned_texts):
        raise ValueError("Leere Texte können nicht eingebettet werden.")

    model = get_embedding_model()

    embedding_matrix = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding_matrix.tolist()


def create_query_embedding(query: str) -> list[float]:
    """Erzeugt ein Embedding für eine einzelne Nutzerfrage."""

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Die Frage darf nicht leer sein.")

    return create_embeddings([cleaned_query])[0]


def create_chunk_embeddings(
    chunks: Sequence[DocumentChunk],
) -> list[list[float]]:
    """Erzeugt für jeden Dokument-Chunk ein Embedding."""

    chunk_texts = [chunk.text for chunk in chunks]
    return create_embeddings(chunk_texts)


def cosine_similarity(
    first_embedding: Sequence[float],
    second_embedding: Sequence[float],
) -> float:
    """Berechnet die Kosinus-Ähnlichkeit zweier Embeddings."""

    first_vector = np.asarray(first_embedding, dtype=np.float32)
    second_vector = np.asarray(second_embedding, dtype=np.float32)

    if first_vector.ndim != 1 or second_vector.ndim != 1:
        raise ValueError("Beide Embeddings müssen eindimensional sein.")

    if first_vector.size == 0 or second_vector.size == 0:
        raise ValueError("Embeddings dürfen nicht leer sein.")

    if first_vector.shape != second_vector.shape:
        raise ValueError(
            "Die Embeddings müssen dieselbe Dimension besitzen."
        )

    first_norm = np.linalg.norm(first_vector)
    second_norm = np.linalg.norm(second_vector)

    if first_norm == 0 or second_norm == 0:
        raise ValueError("Ein Nullvektor besitzt keine definierte Ähnlichkeit.")

    similarity = np.dot(first_vector, second_vector)
    similarity /= first_norm * second_norm

    return float(similarity)