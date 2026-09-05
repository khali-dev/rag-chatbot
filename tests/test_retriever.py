from unittest.mock import MagicMock, patch

import pytest

from src.retriever import Retriever
from src.vector_store import SearchResult


def create_search_result(
    chunk_id: str,
    text: str,
    similarity: float,
    source: str = "example.pdf",
    page: int | None = 1,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page=page,
        chunk_index=0,
        distance=1.0 - similarity,
        similarity=similarity,
    )


def test_empty_query_raises_error() -> None:
    vector_store = MagicMock()
    retriever = Retriever(vector_store=vector_store)

    with pytest.raises(ValueError, match="nicht leer"):
        retriever.retrieve("   ")


@patch(
    "src.retriever.create_query_embedding",
    return_value=[1.0, 0.0, 0.0],
)
def test_retrieve_searches_vector_store(
    mock_embedding: MagicMock,
) -> None:
    vector_store = MagicMock()
    expected_result = create_search_result(
        chunk_id="chunk-1",
        text="Cloud Computing stellt Ressourcen bereit.",
        similarity=0.8,
    )

    vector_store.search.return_value = [expected_result]

    retriever = Retriever(
        vector_store=vector_store,
        top_k=4,
        min_similarity=0.25,
    )

    results = retriever.retrieve("Was ist Cloud Computing?")

    assert results == [expected_result]

    mock_embedding.assert_called_once_with(
        "Was ist Cloud Computing?"
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[1.0, 0.0, 0.0],
        number_of_results=4,
    )


@patch(
    "src.retriever.create_query_embedding",
    return_value=[1.0, 0.0, 0.0],
)
def test_low_similarity_results_are_removed(
    mock_embedding: MagicMock,
) -> None:
    vector_store = MagicMock()

    relevant_result = create_search_result(
        chunk_id="relevant",
        text="Passender Dokumenttext.",
        similarity=0.75,
    )

    irrelevant_result = create_search_result(
        chunk_id="irrelevant",
        text="Unpassender Dokumenttext.",
        similarity=0.10,
    )

    vector_store.search.return_value = [
        relevant_result,
        irrelevant_result,
    ]

    retriever = Retriever(
        vector_store=vector_store,
        min_similarity=0.25,
    )

    results = retriever.retrieve("Eine Frage")

    assert results == [relevant_result]


def test_context_contains_source_and_page() -> None:
    retriever = Retriever(vector_store=MagicMock())

    result = create_search_result(
        chunk_id="chunk-1",
        text="Ein relevanter Dokumenttext.",
        similarity=0.8,
        source="cloud.pdf",
        page=7,
    )

    context = retriever.build_context([result])

    assert "[Quelle 1: cloud.pdf, Seite 7]" in context
    assert "Ein relevanter Dokumenttext." in context


def test_text_file_context_has_no_page() -> None:
    retriever = Retriever(vector_store=MagicMock())

    result = create_search_result(
        chunk_id="chunk-1",
        text="Inhalt einer Textdatei.",
        similarity=0.8,
        source="notes.txt",
        page=None,
    )

    context = retriever.build_context([result])

    assert "[Quelle 1: notes.txt]" in context
    assert "Seite" not in context


def test_empty_results_create_empty_context() -> None:
    retriever = Retriever(vector_store=MagicMock())

    context = retriever.build_context([])

    assert context == ""


def test_context_respects_maximum_length() -> None:
    retriever = Retriever(
        vector_store=MagicMock(),
        max_context_chars=100,
    )

    result = create_search_result(
        chunk_id="chunk-1",
        text="A" * 500,
        similarity=0.8,
    )

    context = retriever.build_context([result])

    assert len(context) <= 100


def test_invalid_top_k_raises_error() -> None:
    with pytest.raises(ValueError, match="top_k"):
        Retriever(
            vector_store=MagicMock(),
            top_k=0,
        )