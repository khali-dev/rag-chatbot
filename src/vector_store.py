from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    TOP_K,
)
from src.text_splitter import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """Ein von ChromaDB gefundenes Suchergebnis."""

    chunk_id: str
    text: str
    source: str
    page: int | None
    chunk_index: int
    distance: float
    similarity: float


class VectorStore:
    """Verwaltet Dokument-Chunks und Embeddings in ChromaDB."""

    def __init__(
        self,
        storage_path: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        client: Any | None = None,
    ) -> None:
        self.collection_name = collection_name

        if client is None:
            self.client = chromadb.PersistentClient(
                path=str(storage_path)
            )
        else:
            self.client = client

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        """Lädt die Collection oder erstellt sie bei Bedarf."""

        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Gibt die Anzahl der gespeicherten Chunks zurück."""

        return self.collection.count()

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Speichert Chunks und ihre Embeddings in ChromaDB."""

        if not chunks and not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Die Anzahl der Chunks und Embeddings muss gleich groß sein."
            )

        if any(not embedding for embedding in embeddings):
            raise ValueError("Embeddings dürfen nicht leer sein.")

        ids: list[str] = []
        documents: list[str] = []
        metadata_list: list[dict[str, str | int]] = []
        embedding_list: list[list[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            metadata: dict[str, str | int] = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }

            if chunk.page is not None:
                metadata["page"] = chunk.page

            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadata_list.append(metadata)
            embedding_list.append(list(embedding))

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadata_list,
            embeddings=embedding_list,
        )

    def search(
        self,
        query_embedding: Sequence[float],
        number_of_results: int = TOP_K,
    ) -> list[SearchResult]:
        """Sucht die ähnlichsten Chunks zu einem Frage-Embedding."""

        if not query_embedding:
            raise ValueError("Das Frage-Embedding darf nicht leer sein.")

        if number_of_results <= 0:
            raise ValueError(
                "Die Anzahl der Suchergebnisse muss größer als 0 sein."
            )

        stored_count = self.count()

        if stored_count == 0:
            return []

        result_count = min(number_of_results, stored_count)

        query_result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )

        ids = query_result["ids"][0]
        documents = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]

        search_results: list[SearchResult] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            page_value = metadata.get("page")

            page = (
                int(page_value)
                if page_value is not None
                else None
            )

            numeric_distance = float(distance)

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    source=str(metadata["source"]),
                    page=page,
                    chunk_index=int(metadata["chunk_index"]),
                    distance=numeric_distance,
                    similarity=1.0 - numeric_distance,
                )
            )

        return search_results

    def delete_document(self, source: str) -> None:
        """Löscht alle Chunks eines bestimmten Dokuments."""

        cleaned_source = source.strip()

        if not cleaned_source:
            raise ValueError("Der Dateiname darf nicht leer sein.")

        self.collection.delete(
            where={"source": cleaned_source}
        )

    def reset(self) -> None:
        """Löscht die Collection und erstellt eine leere neu."""

        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = self._get_or_create_collection()