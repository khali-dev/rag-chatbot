from src.retriever import Retriever


query = input("Stelle eine Frage zum Dokument: ").strip()

if not query:
    raise ValueError("Die Frage darf nicht leer sein.")

retriever = Retriever()
response = retriever.retrieve_with_context(query)

if not response.results:
    print(
        "Es wurden keine ausreichend passenden "
        "Dokumentstellen gefunden."
    )
else:
    print()
    print(f"Gefundene Ergebnisse: {len(response.results)}")
    print()

    for result in response.results:
        page_label = (
            f"Seite {result.page}"
            if result.page is not None
            else "keine Seitennummer"
        )

        print(
            f"{result.source} | "
            f"{page_label} | "
            f"Ähnlichkeit {result.similarity:.3f}"
        )
        print(result.text[:300])
        print()

    print("Zusammengestellter LLM-Kontext:")
    print("-" * 50)
    print(response.context)