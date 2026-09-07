from datetime import datetime

from src.config import (
    EVALUATION_DIR,
    create_data_directories,
)
from src.evaluation import (
    evaluate_questions,
    load_evaluation_questions,
    summarize_results,
    write_evaluation_report,
)
from src.rag_pipeline import RAGPipeline


def main() -> None:
    create_data_directories()

    questions_path = (
        EVALUATION_DIR / "questions.json"
    )

    questions = load_evaluation_questions(
        questions_path
    )

    print(
        f"Geladene Testfragen: {len(questions)}"
    )

    print(
        "Achtung: Jede Frage kann das lokale "
        "Sprachmodell aufrufen."
    )

    confirmation = input(
        "Evaluation starten? [j/N]: "
    ).strip().casefold()

    if confirmation not in {"j", "ja"}:
        print("Evaluation wurde abgebrochen.")
        return

    pipeline = RAGPipeline()

    print()
    print("Evaluation wird ausgeführt ...")
    print()

    results = evaluate_questions(
        pipeline=pipeline,
        questions=questions,
    )

    for result in results:
        status = (
            "BESTANDEN"
            if result.passed
            else "NICHT BESTANDEN"
        )

        print(
            f"[{status}] "
            f"{result.question_id}: "
            f"{result.question}"
        )

        if result.error:
            print(f"Fehler: {result.error}")
        else:
            print(f"Antwort: {result.answer}")

            print(
                "Quellen: "
                f"{', '.join(result.retrieved_sources) or '-'}"
            )

            print(
                "Seiten: "
                f"{result.retrieved_pages or '-'}"
            )

            if result.keyword_score is not None:
                print(
                    "Keyword Score: "
                    f"{result.keyword_score:.2f}"
                )

            if result.refusal_correct is not None:
                print(
                    "Korrekte Ablehnung: "
                    f"{result.refusal_correct}"
                )

        print(
            "Dauer: "
            f"{result.duration_seconds:.2f} Sekunden"
        )

        print("-" * 60)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = EVALUATION_DIR / (
        f"results_{timestamp}.csv"
    )

    write_evaluation_report(
        results=results,
        file_path=report_path,
    )

    summary = summarize_results(results)

    print()
    print("Zusammenfassung:")
    print(f"Fragen: {summary['total']}")
    print(f"Bestanden: {summary['passed']}")
    print(f"Fehlgeschlagen: {summary['failed']}")

    print(
        "Automatische Erfolgsrate: "
        f"{summary['success_rate']:.1%}"
    )

    print(
        "Durchschnittliche Dauer: "
        f"{summary['average_duration']:.2f} Sekunden"
    )

    print()
    print(f"CSV-Bericht: {report_path}")


if __name__ == "__main__":
    main()