from src.rag_pipeline import RAGPipeline


def main() -> None:
    question = input(
        "Stelle eine Frage zum Dokument: "
    ).strip()

    if not question:
        print("Die Frage darf nicht leer sein.")
        return

    pipeline = RAGPipeline()

    print()
    print("Suche passende Dokumentstellen ...")

    response = pipeline.answer_question(question)

    print()
    print("Antwort:")
    print(response.answer)

    if not response.sources:
        return

    print()
    print("Verwendete Quellen:")

    for source in response.sources:
        if source.page is None:
            source_label = source.source
        else:
            source_label = (
                f"{source.source}, Seite {source.page}"
            )

        print(
            f"[Quelle {source.number}] "
            f"{source_label} "
            f"(Ähnlichkeit: {source.similarity:.3f})"
        )


if __name__ == "__main__":
    main()