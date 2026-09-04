import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.document_loader import DocumentPage


@dataclass(frozen=True)
class DocumentChunk:
    """Ein durchsuchbarer Textabschnitt mit Quelleninformationen."""

    text: str
    source: str
    page: int | None
    chunk_index: int
    chunk_id: str


def normalize_text(text: str) -> str:
    """Entfernt unnötige Zeilenumbrüche und mehrfache Leerzeichen."""

    return re.sub(r"\s+", " ", text).strip()


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Teilt einen Text in überlappende Abschnitte."""

    if chunk_size <= 0:
        raise ValueError("Die Chunk-Größe muss größer als 0 sein.")

    if chunk_overlap < 0:
        raise ValueError("Die Überlappung darf nicht negativ sein.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Die Überlappung muss kleiner als die Chunk-Größe sein."
        )

    normalized_text = normalize_text(text)

    if not normalized_text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            window = normalized_text[start:end]

            possible_boundaries = [
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
                window.rfind(" "),
            ]

            boundary = max(possible_boundaries)
            minimum_boundary = int(chunk_size * 0.6)

            if boundary >= minimum_boundary:
                end = start + boundary + 1

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def create_chunk_id(
    source: str,
    page: int | None,
    chunk_index: int,
    text: str,
) -> str:
    """Erzeugt eine reproduzierbare eindeutige ID für einen Chunk."""

    identity = f"{source}|{page}|{chunk_index}|{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def split_document_page(
    document_page: DocumentPage,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Teilt eine geladene Dokumentseite in DocumentChunk-Objekte."""

    text_chunks = split_text(
        text=document_page.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    document_chunks: list[DocumentChunk] = []

    for chunk_index, chunk_text in enumerate(text_chunks):
        chunk_id = create_chunk_id(
            source=document_page.source,
            page=document_page.page,
            chunk_index=chunk_index,
            text=chunk_text,
        )

        document_chunks.append(
            DocumentChunk(
                text=chunk_text,
                source=document_page.source,
                page=document_page.page,
                chunk_index=chunk_index,
                chunk_id=chunk_id,
            )
        )

    return document_chunks


def split_documents(
    document_pages: Iterable[DocumentPage],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Teilt mehrere Dokumentseiten in eine gemeinsame Chunk-Liste."""

    all_chunks: list[DocumentChunk] = []

    for document_page in document_pages:
        page_chunks = split_document_page(
            document_page=document_page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(page_chunks)

    return all_chunks