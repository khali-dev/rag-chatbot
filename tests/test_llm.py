from unittest.mock import MagicMock, patch

import pytest

from src.llm import (
    SYSTEM_PROMPT,
    build_messages,
    extract_response_content,
    generate_answer,
)


def test_build_messages_contains_question_and_context() -> None:
    messages = build_messages(
        question="Was ist RAG?",
        context="[Quelle 1] RAG verbindet Suche und LLM.",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "Was ist RAG?" in messages[1]["content"]
    assert "RAG verbindet Suche und LLM." in messages[1]["content"]


def test_empty_question_raises_error() -> None:
    with pytest.raises(ValueError, match="nicht leer"):
        build_messages(
            question="   ",
            context="Ein Dokumentkontext.",
        )


def test_empty_context_raises_error() -> None:
    with pytest.raises(ValueError, match="nicht leer"):
        build_messages(
            question="Eine Frage",
            context="   ",
        )


def test_extract_dictionary_response() -> None:
    response = {
        "message": {
            "content": "Eine Testantwort."
        }
    }

    result = extract_response_content(response)

    assert result == "Eine Testantwort."


def test_empty_response_raises_error() -> None:
    response = {
        "message": {
            "content": "   "
        }
    }

    with pytest.raises(RuntimeError, match="Ollama hat keinen endgültigen Antworttext zurückgegeben."):
        extract_response_content(response)


@patch("src.llm.ollama.chat")
def test_generate_answer_calls_ollama(
    mock_chat: MagicMock,
) -> None:
    mock_chat.return_value = {
        "message": {
            "content": "RAG verbindet Suche und LLM [Quelle 1]."
        }
    }

    answer = generate_answer(
        question="Was ist RAG?",
        context="[Quelle 1] RAG verbindet Suche und LLM.",
    )

    assert answer == (
        "RAG verbindet Suche und LLM [Quelle 1]."
    )

    mock_chat.assert_called_once()


@patch("src.llm.ollama.chat")
def test_connection_error_is_converted(
    mock_chat: MagicMock,
) -> None:
    mock_chat.side_effect = ConnectionError(
        "Ollama ist nicht erreichbar."
    )

    with pytest.raises(RuntimeError, match="nicht erreichbar"):
        generate_answer(
            question="Was ist RAG?",
            context="[Quelle 1] Ein Kontext.",
        )


def test_negative_temperature_raises_error() -> None:
    with pytest.raises(ValueError, match="nicht negativ"):
        generate_answer(
            question="Was ist RAG?",
            context="[Quelle 1] Ein Kontext.",
            temperature=-0.1,
        )


