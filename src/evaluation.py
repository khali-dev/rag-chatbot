import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.rag_pipeline import (
    NO_RELEVANT_CONTEXT_MESSAGE,
    RAGPipeline,
)


@dataclass(frozen=True)
class EvaluationQuestion:
    """Eine Frage mit erwarteten Ergebnissen."""

    question_id: str
    question: str
    answerable: bool
    expected_sources: tuple[str, ...]
    expected_pages: tuple[int, ...]
    expected_keywords: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Das Ergebnis einer einzelnen Testfrage."""

    question_id: str
    question: str
    answerable: bool
    answer: str
    retrieved_sources: tuple[str, ...]
    retrieved_pages: tuple[int, ...]
    retrieval_hit: bool | None
    keyword_score: float | None
    refusal_correct: bool | None
    duration_seconds: float
    passed: bool
    error: str | None


def load_evaluation_questions(
    file_path: str | Path,
) -> list[EvaluationQuestion]:
    """Lädt und validiert Testfragen aus einer JSON-Datei."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Die Evaluationsdatei wurde nicht gefunden: {path}"
        )

    raw_data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_data, list):
        raise ValueError(
            "Die Evaluationsdatei muss eine JSON-Liste enthalten."
        )

    questions: list[EvaluationQuestion] = []

    for position, entry in enumerate(
        raw_data,
        start=1,
    ):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Eintrag {position} ist kein JSON-Objekt."
            )

        question_id = str(
            entry.get("id", "")
        ).strip()

        question = str(
            entry.get("question", "")
        ).strip()

        answerable = entry.get("answerable")

        if not question_id:
            raise ValueError(
                f"Eintrag {position} besitzt keine ID."
            )

        if not question:
            raise ValueError(
                f"Eintrag {position} besitzt keine Frage."
            )

        if not isinstance(answerable, bool):
            raise ValueError(
                f"Eintrag {question_id}: "
                "'answerable' muss true oder false sein."
            )

        expected_sources = tuple(
            str(source).strip()
            for source in entry.get(
                "expected_sources",
                [],
            )
            if str(source).strip()
        )

        expected_pages = tuple(
            int(page)
            for page in entry.get(
                "expected_pages",
                [],
            )
        )

        expected_keywords = tuple(
            str(keyword).strip()
            for keyword in entry.get(
                "expected_keywords",
                [],
            )
            if str(keyword).strip()
        )

        questions.append(
            EvaluationQuestion(
                question_id=question_id,
                question=question,
                answerable=answerable,
                expected_sources=expected_sources,
                expected_pages=expected_pages,
                expected_keywords=expected_keywords,
            )
        )

    if not questions:
        raise ValueError(
            "Die Evaluationsdatei enthält keine Fragen."
        )

    return questions


def calculate_keyword_score(
    answer: str,
    expected_keywords: Sequence[str],
) -> float | None:
    """Berechnet den Anteil gefundener Schlüsselbegriffe."""

    if not expected_keywords:
        return None

    normalized_answer = answer.casefold()

    found_keywords = sum(
        1
        for keyword in expected_keywords
        if keyword.casefold() in normalized_answer
    )

    return found_keywords / len(expected_keywords)


def contains_refusal(answer: str) -> bool:
    """Prüft, ob das System eine Antwort ablehnt."""

    normalized_answer = answer.casefold()

    refusal_phrases = (
        NO_RELEVANT_CONTEXT_MESSAGE.casefold(),
        "kann anhand der bereitgestellten dokumente "
        "nicht beantwortet werden",
        "geht aus den bereitgestellten dokumenten "
        "nicht hervor",
        "ist im bereitgestellten kontext nicht enthalten",
    )

    return any(
        phrase in normalized_answer
        for phrase in refusal_phrases
    )


def evaluate_question(
    pipeline: RAGPipeline,
    question: EvaluationQuestion,
) -> EvaluationResult:
    """Führt eine einzelne Evaluationsfrage aus."""

    start_time = perf_counter()

    try:
        response = pipeline.answer_question(
            question.question
        )

        duration = perf_counter() - start_time

        retrieved_sources = tuple(
            source.source
            for source in response.sources
        )

        retrieved_pages = tuple(
            source.page
            for source in response.sources
            if source.page is not None
        )

        keyword_score = calculate_keyword_score(
            answer=response.answer,
            expected_keywords=question.expected_keywords,
        )

        retrieval_hit: bool | None = None
        refusal_correct: bool | None = None

        if question.answerable:
            source_match = (
                not question.expected_sources
                or any(
                    source in question.expected_sources
                    for source in retrieved_sources
                )
            )

            page_match = (
                not question.expected_pages
                or any(
                    page in question.expected_pages
                    for page in retrieved_pages
                )
            )

            retrieval_hit = (
                bool(response.sources)
                and source_match
                and page_match
            )

            keyword_requirement_met = (
                keyword_score is None
                or keyword_score >= 0.5
            )

            passed = (
                retrieval_hit
                and keyword_requirement_met
            )

        else:
            refusal_correct = contains_refusal(
                response.answer
            )

            passed = refusal_correct

        return EvaluationResult(
            question_id=question.question_id,
            question=question.question,
            answerable=question.answerable,
            answer=response.answer,
            retrieved_sources=retrieved_sources,
            retrieved_pages=retrieved_pages,
            retrieval_hit=retrieval_hit,
            keyword_score=keyword_score,
            refusal_correct=refusal_correct,
            duration_seconds=duration,
            passed=passed,
            error=None,
        )

    except Exception as exc:
        duration = perf_counter() - start_time

        return EvaluationResult(
            question_id=question.question_id,
            question=question.question,
            answerable=question.answerable,
            answer="",
            retrieved_sources=(),
            retrieved_pages=(),
            retrieval_hit=None,
            keyword_score=None,
            refusal_correct=None,
            duration_seconds=duration,
            passed=False,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


def evaluate_questions(
    pipeline: RAGPipeline,
    questions: Sequence[EvaluationQuestion],
) -> list[EvaluationResult]:
    """Führt mehrere Evaluationsfragen aus."""

    return [
        evaluate_question(
            pipeline=pipeline,
            question=question,
        )
        for question in questions
    ]


def summarize_results(
    results: Sequence[EvaluationResult],
) -> dict[str, float | int]:
    """Berechnet eine Zusammenfassung der Ergebnisse."""

    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": 0.0,
            "average_duration": 0.0,
        }

    passed = sum(
        1
        for result in results
        if result.passed
    )

    average_duration = sum(
        result.duration_seconds
        for result in results
    ) / total

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / total,
        "average_duration": average_duration,
    }


def write_evaluation_report(
    results: Sequence[EvaluationResult],
    file_path: str | Path,
) -> Path:
    """Speichert die Ergebnisse als CSV-Datei."""

    path = Path(file_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "question_id",
        "question",
        "answerable",
        "answer",
        "retrieved_sources",
        "retrieved_pages",
        "retrieval_hit",
        "keyword_score",
        "refusal_correct",
        "duration_seconds",
        "automatic_passed",
        "manual_answer_score",
        "manual_source_score",
        "notes",
        "error",
    ]

    with path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "answerable": result.answerable,
                    "answer": result.answer,
                    "retrieved_sources": " | ".join(
                        result.retrieved_sources
                    ),
                    "retrieved_pages": " | ".join(
                        str(page)
                        for page in result.retrieved_pages
                    ),
                    "retrieval_hit": result.retrieval_hit,
                    "keyword_score": result.keyword_score,
                    "refusal_correct": result.refusal_correct,
                    "duration_seconds": round(
                        result.duration_seconds,
                        3,
                    ),
                    "automatic_passed": result.passed,
                    "manual_answer_score": "",
                    "manual_source_score": "",
                    "notes": "",
                    "error": result.error or "",
                }
            )

    return path