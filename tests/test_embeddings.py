import numpy as np
import pytest

from src.embeddings import (
    cosine_similarity,
    create_chunk_embeddings,
    create_embeddings,
    create_query_embedding,
)
from src.text_splitter import DocumentChunk


def test_create_single_embedding() -> None:
    embedding = create_query_embedding("Was ist RAG?")

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


def test_embedding_is_normalized() -> None:
    embedding = create_query_embedding("Ein Test für ein Embedding.")

    vector_length = np.linalg.norm(embedding)

    assert vector_length == pytest.approx(1.0, abs=0.001)


def test_create_multiple_embeddings() -> None:
    texts = [
        "Das ist der erste Text.",
        "Das ist der zweite Text.",
        "Das ist der dritte Text.",
    ]

    embeddings = create_embeddings(texts)

    assert len(embeddings) == 3
    assert all(len(embedding) == 384 for embedding in embeddings)


def test_similar_texts_are_closer() -> None:
    texts = [
        "Das Unternehmen verwendet Cloud Computing.",
        "Die Firma nutzt Dienste aus der Cloud.",
        "Der Kuchen wird im Backofen gebacken.",
    ]

    embeddings = create_embeddings(texts)

    similar_score = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    unrelated_score = cosine_similarity(
        embeddings[0],
        embeddings[2],
    )

    assert similar_score > unrelated_score


def test_create_chunk_embeddings() -> None:
    chunks = [
        DocumentChunk(
            text="Der erste Dokumentabschnitt.",
            source="example.pdf",
            page=1,
            chunk_index=0,
            chunk_id="chunk-1",
        ),
        DocumentChunk(
            text="Der zweite Dokumentabschnitt.",
            source="example.pdf",
            page=1,
            chunk_index=1,
            chunk_id="chunk-2",
        ),
    ]

    embeddings = create_chunk_embeddings(chunks)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_empty_text_list_returns_empty_list() -> None:
    embeddings = create_embeddings([])

    assert embeddings == []


def test_empty_query_raises_error() -> None:
    with pytest.raises(ValueError, match="nicht leer"):
        create_query_embedding("   ")


def test_different_dimensions_raise_error() -> None:
    with pytest.raises(ValueError, match="dieselbe Dimension"):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )