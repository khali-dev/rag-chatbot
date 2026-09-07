from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.document_loader import DocumentPage
from src.rag_pipeline import (
    NO_RELEVANT_CONTEXT_MESSAGE,
    RAGPipeline,
)
from src.retriever import RetrievalResponse
from src.text_splitter import DocumentChunk
from src.vector_store import SearchResult


def create_pipeline(
    vector_store: MagicMock | None = None,
    retriever: MagicMock | None = None,
    answer_generator: MagicMock | None = None,
) -> tuple[
    RAGPipeline,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    actual_vector_store = (
        vector_store
        if vector_store is not None
        else MagicMock()
    )

    actual_retriever = (
        retriever
        if retriever is not None
        else MagicMock()
    )

    actual_answer_generator = (
        answer_generator
        if answer_generator is not None
        else MagicMock()
    )

    pipeline = RAGPipeline(
        vector_store=actual_vector_store,
        retriever=actual_retriever,
        answer_generator=actual_answer_generator,
    )

    return (
        pipeline,
        actual_vector_store,
        actual_retriever,
        actual_answer_generator,
    )


def test_empty_file_list_raises_error() -> None:
    pipeline, _, _, _ = create_pipeline()

    with pytest.raises(ValueError, match="mindestens"):
        pipeline.index_documents([])


@patch("src.rag_pipeline.create_chunk_embeddings")
@patch("src.rag_pipeline.split_documents")
@patch("src.rag_pipeline.load_documents")
def test_index_documents(
    mock_load_documents: MagicMock,
    mock_split_documents: MagicMock,
    mock_create_embeddings: MagicMock,
) -> None:
    document_page = DocumentPage(
        text="Ein Dokumenttext.",
        source="example.pdf",
        page=1,
    )

    chunk = DocumentChunk(
        text="Ein Dokumenttext.",
        source="example.pdf",
        page=1,
        chunk_index=0,
        chunk_id="chunk-1",
    )

    embedding = [1.0, 0.0, 0.0]

    mock_load_documents.return_value = [document_page]
    mock_split_documents.return_value = [chunk]
    mock_create_embeddings.return_value = [embedding]

    pipeline, vector_store, _, _ = create_pipeline()
    vector_store.count.return_value = 1

    result = pipeline.index_documents(
        [Path("example.pdf")]
    )

    assert result.indexed_documents == 1
    assert result.loaded_pages == 1
    assert result.created_chunks == 1
    assert result.stored_chunks == 1
    assert result.sources == ("example.pdf",)

    vector_store.delete_document.assert_called_once_with(
        "example.pdf"
    )

    vector_store.add_chunks.assert_called_once_with(
        chunks=[chunk],
        embeddings=[embedding],
    )


def test_empty_question_raises_error() -> None:
    pipeline, _, _, _ = create_pipeline()

    with pytest.raises(ValueError, match="nicht leer"):
        pipeline.answer_question("   ")


def test_question_without_results_does_not_call_llm() -> None:
    pipeline, _, retriever, answer_generator = (
        create_pipeline()
    )

    retriever.retrieve_with_context.return_value = (
        RetrievalResponse(
            query="Unbekannte Frage",
            results=(),
            context="",
        )
    )

    response = pipeline.answer_question(
        "Unbekannte Frage"
    )

    assert response.answer == NO_RELEVANT_CONTEXT_MESSAGE
    assert response.sources == ()
    assert response.context == ""

    answer_generator.assert_not_called()


def test_question_with_result_calls_llm() -> None:
    pipeline, _, retriever, answer_generator = (
        create_pipeline()
    )

    search_result = SearchResult(
        chunk_id="chunk-1",
        text="RAG verbindet Suche und Sprachmodell.",
        source="rag.pdf",
        page=3,
        chunk_index=0,
        distance=0.2,
        similarity=0.8,
    )

    context = (
        "[Quelle 1: rag.pdf, Seite 3]\n"
        "RAG verbindet Suche und Sprachmodell."
    )

    retriever.retrieve_with_context.return_value = (
        RetrievalResponse(
            query="Was ist RAG?",
            results=(search_result,),
            context=context,
        )
    )

    answer_generator.return_value = (
        "RAG verbindet eine Suche mit einem "
        "Sprachmodell [Quelle 1]."
    )

    response = pipeline.answer_question(
        "Was ist RAG?"
    )

    assert response.answer == (
        "RAG verbindet eine Suche mit einem "
        "Sprachmodell [Quelle 1]."
    )

    assert len(response.sources) == 1
    assert response.sources[0].number == 1
    assert response.sources[0].source == "rag.pdf"
    assert response.sources[0].page == 3
    assert response.sources[0].similarity == 0.8
    assert response.context == context

    answer_generator.assert_called_once_with(
        question="Was ist RAG?",
        context=context,
    )


def test_multiple_sources_are_numbered() -> None:
    pipeline, _, retriever, answer_generator = (
        create_pipeline()
    )

    first_result = SearchResult(
        chunk_id="first",
        text="Erster Text.",
        source="first.pdf",
        page=1,
        chunk_index=0,
        distance=0.1,
        similarity=0.9,
    )

    second_result = SearchResult(
        chunk_id="second",
        text="Zweiter Text.",
        source="second.pdf",
        page=2,
        chunk_index=0,
        distance=0.2,
        similarity=0.8,
    )

    retriever.retrieve_with_context.return_value = (
        RetrievalResponse(
            query="Eine Frage",
            results=(first_result, second_result),
            context="Kontext",
        )
    )

    answer_generator.return_value = "Eine Antwort."

    response = pipeline.answer_question("Eine Frage")

    assert len(response.sources) == 2
    assert response.sources[0].number == 1
    assert response.sources[1].number == 2