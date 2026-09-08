from types import SimpleNamespace
from unittest.mock import Mock

from src.rag_pipeline import (
    NO_RELEVANT_CONTEXT_MESSAGE,
    RAGPipeline,
)


def create_pipeline(
    retrieval_results: list,
    context: str = "",
) -> tuple[RAGPipeline, Mock, Mock]:
    """Erstellt eine Pipeline mit Test-Doubles."""

    vector_store = Mock()
    retriever = Mock()

    retriever.retrieve_with_context.return_value = (
        SimpleNamespace(
            results=retrieval_results,
            context=context,
        )
    )

    answer_generator = Mock(
        return_value="Dokumentantwort"
    )

    fallback_generator = Mock(
        return_value="Allgemeine Antwort"
    )

    pipeline = RAGPipeline(
        vector_store=vector_store,
        retriever=retriever,
        answer_generator=answer_generator,
        fallback_answer_generator=fallback_generator,
    )

    return (
        pipeline,
        answer_generator,
        fallback_generator,
    )


def test_no_results_without_fallback() -> None:
    """Ohne Fallback bleibt die bisherige Meldung."""

    pipeline, answer_generator, fallback_generator = (
        create_pipeline([])
    )

    result = pipeline.answer_question(
        question="Testfrage",
        allow_general_knowledge=False,
    )

    assert result.answer == (
        NO_RELEVANT_CONTEXT_MESSAGE
    )
    assert result.answer_mode == "not_answered"
    assert result.sources == ()
    assert result.used_document_sources is False
    assert result.used_general_knowledge is False

    answer_generator.assert_not_called()
    fallback_generator.assert_not_called()


def test_no_results_with_fallback() -> None:
    """Ohne Quellen wird allgemeines Wissen verwendet."""

    pipeline, answer_generator, fallback_generator = (
        create_pipeline([])
    )

    result = pipeline.answer_question(
        question="Testfrage",
        allow_general_knowledge=True,
    )

    assert result.answer == "Allgemeine Antwort"
    assert result.answer_mode == "general_knowledge"
    assert result.sources == ()
    assert result.used_document_sources is False
    assert result.used_general_knowledge is True

    answer_generator.assert_not_called()

    fallback_generator.assert_called_once_with(
        question="Testfrage"
    )


def test_document_answer_has_sources() -> None:
    """Eine passende Dokumentantwort behält Quellen."""

    retrieval_result = SimpleNamespace(
        source="test.pdf",
        page=2,
        similarity=0.9,
        text="Passender Dokumenttext",
    )

    pipeline, answer_generator, fallback_generator = (
        create_pipeline(
            retrieval_results=[
                retrieval_result
            ],
            context=(
                "[Quelle 1] Passender Dokumenttext"
            ),
        )
    )

    result = pipeline.answer_question(
        question="Testfrage",
        allow_general_knowledge=True,
    )

    assert result.answer == "Dokumentantwort"
    assert result.answer_mode == "documents"
    assert len(result.sources) == 1
    assert result.sources[0].source == "test.pdf"
    assert result.used_document_sources is True
    assert result.used_general_knowledge is False

    answer_generator.assert_called_once()
    fallback_generator.assert_not_called()


def test_model_rejects_context_and_uses_fallback() -> None:
    """Eine abgelehnte Dokumentantwort löst Fallback aus."""

    retrieval_result = SimpleNamespace(
        source="test.pdf",
        page=1,
        similarity=0.7,
        text="Unpassender Dokumenttext",
    )

    pipeline, answer_generator, fallback_generator = (
        create_pipeline(
            retrieval_results=[
                retrieval_result
            ],
            context=(
                "[Quelle 1] Unpassender Dokumenttext"
            ),
        )
    )

    answer_generator.return_value = (
        NO_RELEVANT_CONTEXT_MESSAGE
    )

    result = pipeline.answer_question(
        question="Andere Testfrage",
        allow_general_knowledge=True,
    )

    assert result.answer == "Allgemeine Antwort"
    assert result.answer_mode == "general_knowledge"
    assert result.sources == ()

    fallback_generator.assert_called_once_with(
        question="Andere Testfrage"
    )