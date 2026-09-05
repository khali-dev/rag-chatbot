from src.llm import generate_answer
from src.retriever import Retriever


def main() -> None:
    query = input("Stelle eine Frage zum Dokument: ").strip()

    if not query:
        print("Die Frage darf nicht leer sein.")
        return

    retriever = Retriever()
    retrieval_response = retriever.retrieve_with_context(query)

    if not retrieval_response.results:
        print()
        print(
            "Es wurden keine ausreichend passenden "
            "Dokumentstellen gefunden."
        )
        return

    print()
    print("Erzeuge Antwort mit dem lokalen Sprachmodell ...")

    answer = generate_answer(
        question=query,
        context=retrieval_response.context,
    )

    print()
    print("Antwort:")
    print(answer)

    print()
    print("Verwendete Quellen:")

    for source_number, result in enumerate(
        retrieval_response.results,
        start=1,
    ):
        if result.page is None:
            source_label = result.source
        else:
            source_label = (
                f"{result.source}, Seite {result.page}"
            )

        print(
            f"[Quelle {source_number}] "
            f"{source_label} "
            f"(Ähnlichkeit: {result.similarity:.3f})"
        )


if __name__ == "__main__":
    main()