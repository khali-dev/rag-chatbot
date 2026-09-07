from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.document_loader import load_documents
from src.embeddings import create_chunk_embeddings
from src.llm import generate_answer
from src.retriever import Retriever
from src.text_splitter import split_documents
from src.vector_store import VectorStore


NO_RELEVANT_CONTEXT_MESSAGE = (
    "Diese Frage kann anhand der bereitgestellten "
    "Dokumente nicht beantwortet werden."
)


@dataclass(frozen=True)
class IndexingResult:
    """Ergebnis einer Dokumentindexierung."""

    indexed_documents: int
    loaded_pages: int
    created_chunks: int
    stored_chunks: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class RAGSource:
    """Eine Quelle, die für eine Antwort verwendet wurde."""

    number: int
    source: str
    page: int | None
    similarity: float
    text: str


@dataclass(frozen=True)
class RAGAnswer:
    """Vollständige Antwort der RAG-Pipeline."""

    question: str
    answer: str
    sources: tuple[RAGSource, ...]
    context: str


class RAGPipeline:
    """Verbindet Indexierung, Retrieval und Antwortgenerierung."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        retriever: Retriever | None = None,
        answer_generator: Callable[..., str] = generate_answer,
    ) -> None:
        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore()
        )

        self.retriever = (
            retriever
            if retriever is not None
            else Retriever(vector_store=self.vector_store)
        )

        self.answer_generator = answer_generator

    def index_documents(
        self,
        file_paths: Sequence[str | Path],
    ) -> IndexingResult:
        """Lädt, zerlegt und indexiert mehrere Dokumente."""

        if not file_paths:
            raise ValueError(
                "Es muss mindestens ein Dokument angegeben werden."
            )

        document_pages = load_documents(file_paths)

        chunks = split_documents(document_pages)

        if not chunks:
            raise ValueError(
                "Aus den Dokumenten konnten keine Chunks erstellt werden."
            )

        embeddings = create_chunk_embeddings(chunks)

        if len(chunks) != len(embeddings):
            raise RuntimeError(
                "Die Anzahl der Chunks und Embeddings stimmt nicht überein."
            )

        source_names = tuple(
            dict.fromkeys(
                document_page.source
                for document_page in document_pages
            )
        )

        # Alte Chunks der erneut indexierten Dateien entfernen.
        for source_name in source_names:
            self.vector_store.delete_document(source_name)

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return IndexingResult(
            indexed_documents=len(source_names),
            loaded_pages=len(document_pages),
            created_chunks=len(chunks),
            stored_chunks=self.vector_store.count(),
            sources=source_names,
        )

    def answer_question(
        self,
        question: str,
    ) -> RAGAnswer:
        """Beantwortet eine Frage anhand indexierter Dokumente."""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Die Frage darf nicht leer sein.")

        retrieval_response = (
            self.retriever.retrieve_with_context(
                cleaned_question
            )
        )

        if not retrieval_response.results:
            return RAGAnswer(
                question=cleaned_question,
                answer=NO_RELEVANT_CONTEXT_MESSAGE,
                sources=(),
                context="",
            )

        answer = self.answer_generator(
            question=cleaned_question,
            context=retrieval_response.context,
        )

        sources = tuple(
            RAGSource(
                number=source_number,
                source=result.source,
                page=result.page,
                similarity=result.similarity,
                text=result.text,
            )
            for source_number, result in enumerate(
                retrieval_response.results,
                start=1,
            )
        )

        return RAGAnswer(
            question=cleaned_question,
            answer=answer,
            sources=sources,
            context=retrieval_response.context,
        )