from time import perf_counter

from src.embeddings import get_embedding_model
from src.performance import (
    format_benchmark_result,
    measure_operation,
)
from src.rag_pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()

    stored_chunks = (
        pipeline.vector_store.count()
    )

    print(
        f"Gespeicherte Chunks: {stored_chunks}"
    )

    if stored_chunks == 0:
        print(
            "Es sind keine Dokumente indexiert."
        )
        return

    embedding_model = get_embedding_model()

    print(
        "Embedding-Gerät: "
        f"{embedding_model.device}"
    )

    question = input(
        "Testfrage zum Dokument: "
    ).strip()

    if not question:
        print("Die Frage darf nicht leer sein.")
        return

    print()
    print("Erster Durchlauf wird gemessen ...")

    cold_start = perf_counter()

    first_response = (
        pipeline.answer_question(question)
    )

    cold_duration = (
        perf_counter() - cold_start
    )

    print(
        "Erster Durchlauf: "
        f"{cold_duration:.2f} Sekunden"
    )

    print()
    print("Antwort:")
    print(first_response.answer)

    print()
    print(
        "Drei weitere Durchläufe werden "
        "gemessen ..."
    )

    warm_result = measure_operation(
        name="Warme RAG-Anfrage",
        operation=lambda: (
            pipeline.answer_question(question)
        ),
        repetitions=3,
    )

    print()
    print(
        format_benchmark_result(
            warm_result
        )
    )

    print()
    print(
        "Der erste Durchlauf enthält möglicherweise "
        "Modellladezeit. Die weiteren Durchläufe "
        "verwenden bereits geladene Modelle."
    )


if __name__ == "__main__":
    main()