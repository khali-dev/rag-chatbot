from pathlib import Path

from src.rag_pipeline import RAGPipeline


def main() -> None:
    user_input = input(
        "Pfad des zu indexierenden Dokuments: "
    ).strip()

    file_path_text = user_input.strip('"')

    if not file_path_text:
        print("Es wurde kein Dokument angegeben.")
        return

    file_path = Path(file_path_text)

    pipeline = RAGPipeline()

    print()
    print("Dokument wird verarbeitet ...")

    result = pipeline.index_documents([file_path])

    print()
    print("Indexierung erfolgreich:")
    print(
        f"Dokumente: {result.indexed_documents}"
    )
    print(f"Geladene Seiten: {result.loaded_pages}")
    print(f"Erstellte Chunks: {result.created_chunks}")
    print(
        f"Gespeicherte Chunks insgesamt: "
        f"{result.stored_chunks}"
    )
    print(f"Quellen: {', '.join(result.sources)}")


if __name__ == "__main__":
    main()