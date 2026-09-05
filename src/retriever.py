from collections.abc import Sequence
from dataclasses import dataclass

from src.config import (
    MAX_CONTEXT_CHARS,
    MIN_SIMILARITY,
    TOP_K,
)
from src.embeddings import create_query_embedding
from src.vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class RetrievalResponse:
    """Vollständiges Ergebnis eines Retrieval-Vorgangs."""

    query: str
    results: tuple[SearchResult, ...]
    context: str


class Retriever:
    """Sucht relevante Dokumentstellen zu einer Nutzerfrage."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k muss größer als 0 sein.")

        if max_context_chars <= 0:
            raise ValueError(
                "max_context_chars muss größer als 0 sein."
            )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

        self.top_k = top_k
        self.min_similarity = min_similarity
        self.max_context_chars = max_context_chars

    def retrieve(
        self,
        query: str,
    ) -> list[SearchResult]:
        """Sucht und filtert relevante Chunks."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Die Frage darf nicht leer sein.")

        query_embedding = create_query_embedding(cleaned_query)

        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            number_of_results=self.top_k,
        )

        return [
            result
            for result in search_results
            if result.similarity >= self.min_similarity
        ]

    def build_context(
        self,
        results: Sequence[SearchResult],
    ) -> str:
        """Formatiert Suchergebnisse als Kontext für das LLM."""

        context_sections: list[str] = []
        current_length = 0

        for source_number, result in enumerate(results, start=1):
            if result.page is None:
                source_label = result.source
            else:
                source_label = (
                    f"{result.source}, Seite {result.page}"
                )

            section = (
                f"[Quelle {source_number}: {source_label}]\n"
                f"{result.text}"
            )

            separator_length = 2 if context_sections else 0
            remaining_space = (
                self.max_context_chars
                - current_length
                - separator_length
            )

            if remaining_space <= 0:
                break

            if len(section) > remaining_space:
                shortened_section = section[:remaining_space].rstrip()

                if shortened_section:
                    context_sections.append(shortened_section)

                break

            context_sections.append(section)
            current_length += len(section) + separator_length

        return "\n\n".join(context_sections)

    def retrieve_with_context(
        self,
        query: str,
    ) -> RetrievalResponse:
        """Sucht relevante Chunks und baut den LLM-Kontext."""

        cleaned_query = query.strip()
        results = self.retrieve(cleaned_query)
        context = self.build_context(results)

        return RetrievalResponse(
            query=cleaned_query,
            results=tuple(results),
            context=context,
        )