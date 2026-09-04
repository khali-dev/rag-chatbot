import pytest

from src.document_loader import DocumentPage
from src.text_splitter import (
    DocumentChunk,
    create_chunk_id,
    normalize_text,
    split_document_page,
    split_documents,
    split_text,
)


def test_normalize_text() -> None:
    text = "RAG\n\nverbindet   Suche\tund LLM."

    result = normalize_text(text)

    assert result == "RAG verbindet Suche und LLM."


def test_short_text_creates_one_chunk() -> None:
    chunks = split_text(
        text="Das ist ein kurzer Text.",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks == ["Das ist ein kurzer Text."]


def test_long_text_creates_multiple_chunks() -> None:
    text = "RAG ist ein hilfreiches Verfahren. " * 100

    chunks = split_text(
        text=text,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_empty_text_creates_no_chunks() -> None:
    chunks = split_text(
        text="   \n\t",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks == []


def test_invalid_chunk_size_raises_error() -> None:
    with pytest.raises(ValueError, match="größer als 0"):
        split_text(
            text="Test",
            chunk_size=0,
            chunk_overlap=0,
        )


def test_negative_overlap_raises_error() -> None:
    with pytest.raises(ValueError, match="nicht negativ"):
        split_text(
            text="Test",
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="kleiner"):
        split_text(
            text="Test",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_document_metadata_is_preserved() -> None:
    document_page = DocumentPage(
        text="Ein kurzer Dokumenttext.",
        source="example.pdf",
        page=4,
    )

    chunks = split_document_page(
        document_page=document_page,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].source == "example.pdf"
    assert chunks[0].page == 4
    assert chunks[0].chunk_index == 0


def test_chunk_ids_are_reproducible() -> None:
    first_id = create_chunk_id(
        source="example.pdf",
        page=2,
        chunk_index=0,
        text="Beispieltext",
    )

    second_id = create_chunk_id(
        source="example.pdf",
        page=2,
        chunk_index=0,
        text="Beispieltext",
    )

    assert first_id == second_id


def test_different_chunks_have_different_ids() -> None:
    first_id = create_chunk_id(
        source="example.pdf",
        page=2,
        chunk_index=0,
        text="Erster Text",
    )

    second_id = create_chunk_id(
        source="example.pdf",
        page=2,
        chunk_index=1,
        text="Zweiter Text",
    )

    assert first_id != second_id


def test_multiple_pages_are_split() -> None:
    document_pages = [
        DocumentPage(
            text="Text der ersten Seite.",
            source="example.pdf",
            page=1,
        ),
        DocumentPage(
            text="Text der zweiten Seite.",
            source="example.pdf",
            page=2,
        ),
    ]

    chunks = split_documents(
        document_pages=document_pages,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[1].page == 2