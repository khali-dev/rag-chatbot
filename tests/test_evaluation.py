from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation import (
    EvaluationQuestion,
    calculate_keyword_score,
    contains_refusal,
    evaluate_question,
    load_evaluation_questions,
    summarize_results,
    write_evaluation_report,
)
from src.rag_pipeline import (
    NO_RELEVANT_CONTEXT_MESSAGE,
    RAGAnswer,
    RAGSource,
)


def create_source() -> RAGSource:
    return RAGSource(
        number=1,
        source="manual.pdf",
        page=5,
        similarity=0.8,
        text="Die Wandmontage ist möglich.",
    )


def test_keyword_score() -> None:
    score = calculate_keyword_score(
        answer=(
            "Die Wandmontage sollte durch "
            "einen Techniker erfolgen."
        ),
        expected_keywords=[
            "Wandmontage",
            "Techniker",
        ],
    )

    assert score == 1.0


def test_keyword_score_without_keywords() -> None:
    score = calculate_keyword_score(
        answer="Eine Antwort.",
        expected_keywords=[],
    )

    assert score is None


def test_refusal_is_detected() -> None:
    assert contains_refusal(
        NO_RELEVANT_CONTEXT_MESSAGE
    )


def test_answerable_question_passes() -> None:
    pipeline = MagicMock()

    pipeline.answer_question.return_value = RAGAnswer(
        question="Ist Wandmontage möglich?",
        answer=(
            "Ja, die Wandmontage ist möglich "
            "[Quelle 1]."
        ),
        sources=(create_source(),),
        context="Kontext",
    )

    question = EvaluationQuestion(
        question_id="test-1",
        question="Ist Wandmontage möglich?",
        answerable=True,
        expected_sources=("manual.pdf",),
        expected_pages=(5,),
        expected_keywords=("Wandmontage",),
    )

    result = evaluate_question(
        pipeline=pipeline,
        question=question,
    )

    assert result.retrieval_hit is True
    assert result.keyword_score == 1.0
    assert result.passed is True


def test_unanswerable_question_passes() -> None:
    pipeline = MagicMock()

    pipeline.answer_question.return_value = RAGAnswer(
        question="Welche Farbe hat das Auto?",
        answer=NO_RELEVANT_CONTEXT_MESSAGE,
        sources=(),
        context="",
    )

    question = EvaluationQuestion(
        question_id="test-2",
        question="Welche Farbe hat das Auto?",
        answerable=False,
        expected_sources=(),
        expected_pages=(),
        expected_keywords=(),
    )

    result = evaluate_question(
        pipeline=pipeline,
        question=question,
    )

    assert result.refusal_correct is True
    assert result.passed is True


def test_load_questions(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "questions.json"

    file_path.write_text(
        """
        [
          {
            "id": "test-1",
            "question": "Was ist RAG?",
            "answerable": true,
            "expected_sources": ["rag.pdf"],
            "expected_pages": [1],
            "expected_keywords": ["Retrieval"]
          }
        ]
        """,
        encoding="utf-8",
    )

    questions = load_evaluation_questions(
        file_path
    )

    assert len(questions) == 1
    assert questions[0].question_id == "test-1"
    assert questions[0].answerable is True


def test_empty_question_file_raises_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "questions.json"
    file_path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="keine Fragen",
    ):
        load_evaluation_questions(file_path)


def test_write_report(
    tmp_path: Path,
) -> None:
    pipeline = MagicMock()

    pipeline.answer_question.return_value = RAGAnswer(
        question="Ist Wandmontage möglich?",
        answer="Ja, Wandmontage ist möglich.",
        sources=(create_source(),),
        context="Kontext",
    )

    question = EvaluationQuestion(
        question_id="test-1",
        question="Ist Wandmontage möglich?",
        answerable=True,
        expected_sources=("manual.pdf",),
        expected_pages=(5,),
        expected_keywords=("Wandmontage",),
    )

    result = evaluate_question(
        pipeline=pipeline,
        question=question,
    )

    report_path = write_evaluation_report(
        results=[result],
        file_path=tmp_path / "report.csv",
    )

    assert report_path.exists()

    report_content = report_path.read_text(
        encoding="utf-8-sig"
    )

    assert "test-1" in report_content
    assert "manual_answer_score" in report_content


def test_summary() -> None:
    pipeline = MagicMock()

    pipeline.answer_question.return_value = RAGAnswer(
        question="Eine Frage",
        answer="Eine richtige Antwort.",
        sources=(create_source(),),
        context="Kontext",
    )

    question = EvaluationQuestion(
        question_id="test-1",
        question="Eine Frage",
        answerable=True,
        expected_sources=("manual.pdf",),
        expected_pages=(5,),
        expected_keywords=("richtige",),
    )

    result = evaluate_question(
        pipeline=pipeline,
        question=question,
    )

    summary = summarize_results([result])

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["success_rate"] == 1.0