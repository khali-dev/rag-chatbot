from src.embeddings import create_query_embedding
from src.vector_store import VectorStore


query = input("Stelle eine Frage zum PDF: ").strip()

if not query:
    raise ValueError("Die Frage darf nicht leer sein.")

store = VectorStore()
query_embedding = create_query_embedding(query)
results = store.search(query_embedding)

if not results:
    print("Es wurden keine Ergebnisse gefunden.")
else:
    for result in results:
        print(
            f"{result.source} | "
            f"Seite {result.page} | "
            f"Ähnlichkeit {result.similarity:.3f}"
        )
        print(result.text[:300])
        print()