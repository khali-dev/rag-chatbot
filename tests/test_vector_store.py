from uuid import uuid4

import chromadb
import pytest

from src.text_splitter import DocumentChunk
from src.vector_store import VectorStore


@pytest.fixture
def vector_store() -> VectorStore:
    client = chromadb.EphemeralClient()

    return VectorStore(
        client=client,
        collection_name=f"test_{uuid4().hex}",
    )


def create_test_chunk(
    chunk_id: str,
    text: str,
    source: str = "example.pdf",
    page: int | None = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        source=source,
        page=page,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
    )


def test_new_vector_store_is_empty(
    vector_store: VectorStore,
) -> None:
    assert vector_store.count() == 0


def test_add_chunk(
    vector_store: VectorStore,
) -> None:
    chunk = create_test_chunk(
        chunk_id="chunk-1",
        text="Cloud Computing stellt IT-Ressourcen bereit.",
    )

    vector_store.add_chunks(
        chunks=[chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    assert vector_store.count() == 1


def test_upsert_does_not_create_duplicate(
    vector_store: VectorStore,
) -> None:
    chunk = create_test_chunk(
        chunk_id="chunk-1",
        text="Ein Dokumenttext.",
    )

    vector_store.add_chunks(
        chunks=[chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    vector_store.add_chunks(
        chunks=[chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    assert vector_store.count() == 1


def test_search_returns_most_similar_chunk(
    vector_store: VectorStore,
) -> None:
    cloud_chunk = create_test_chunk(
        chunk_id="cloud",
        text="Cloud Computing stellt Server bereit.",
        source="cloud.pdf",
    )

    cooking_chunk = create_test_chunk(
        chunk_id="cooking",
        text="Ein Kuchen wird im Backofen gebacken.",
        source="cooking.pdf",
    )

    vector_store.add_chunks(
        chunks=[cloud_chunk, cooking_chunk],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = vector_store.search(
        query_embedding=[0.9, 0.1, 0.0],
        number_of_results=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "cloud"
    assert results[0].source == "cloud.pdf"


def test_text_file_has_no_page_number(
    vector_store: VectorStore,
) -> None:
    chunk = create_test_chunk(
        chunk_id="text-file",
        text="Inhalt einer Textdatei.",
        source="notes.txt",
        page=None,
    )

    vector_store.add_chunks(
        chunks=[chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        number_of_results=1,
    )

    assert results[0].page is None


def test_different_lengths_raise_error(
    vector_store: VectorStore,
) -> None:
    chunk = create_test_chunk(
        chunk_id="chunk-1",
        text="Ein Dokumenttext.",
    )

    with pytest.raises(ValueError, match="gleich groß"):
        vector_store.add_chunks(
            chunks=[chunk],
            embeddings=[],
        )


def test_empty_store_returns_no_results(
    vector_store: VectorStore,
) -> None:
    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0]
    )

    assert results == []


def test_delete_document(
    vector_store: VectorStore,
) -> None:
    first_chunk = create_test_chunk(
        chunk_id="first",
        text="Inhalt des ersten Dokuments.",
        source="first.pdf",
    )

    second_chunk = create_test_chunk(
        chunk_id="second",
        text="Inhalt des zweiten Dokuments.",
        source="second.pdf",
    )

    vector_store.add_chunks(
        chunks=[first_chunk, second_chunk],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    vector_store.delete_document("first.pdf")

    assert vector_store.count() == 1


def test_reset_removes_all_chunks(
    vector_store: VectorStore,
) -> None:
    chunk = create_test_chunk(
        chunk_id="chunk-1",
        text="Ein Dokumenttext.",
    )

    vector_store.add_chunks(
        chunks=[chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    vector_store.reset()

    assert vector_store.count() == 0