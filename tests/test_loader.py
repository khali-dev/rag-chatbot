from pathlib import Path

import pymupdf
import pytest

from src.document_loader import (
    DocumentPage,
    load_document,
    load_documents,
    load_pdf_file,
    load_text_file,
)


def test_load_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text(
        "RAG verbindet eine Suche mit einem Sprachmodell.",
        encoding="utf-8",
    )

    pages = load_text_file(file_path)

    assert len(pages) == 1
    assert isinstance(pages[0], DocumentPage)
    assert pages[0].text == (
        "RAG verbindet eine Suche mit einem Sprachmodell."
    )
    assert pages[0].source == "example.txt"
    assert pages[0].page is None


def test_load_pdf_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.pdf"

    pdf = pymupdf.open()
    pdf_page = pdf.new_page()
    pdf_page.insert_text(
        (72, 72),
        "This is a PDF test page.",
    )
    pdf.save(file_path)
    pdf.close()

    pages = load_pdf_file(file_path)

    assert len(pages) == 1
    assert pages[0].text == "This is a PDF test page."
    assert pages[0].source == "example.pdf"
    assert pages[0].page == 1


def test_load_document_selects_text_loader(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Test content", encoding="utf-8")

    pages = load_document(file_path)

    assert len(pages) == 1
    assert pages[0].source == "notes.txt"


def test_load_multiple_documents(tmp_path: Path) -> None:
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("First document", encoding="utf-8")
    second_file.write_text("Second document", encoding="utf-8")

    pages = load_documents([first_file, second_file])

    assert len(pages) == 2
    assert pages[0].source == "first.txt"
    assert pages[1].source == "second.txt"


def test_missing_file_raises_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(missing_file)


def test_empty_text_file_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="leer"):
        load_text_file(file_path)


def test_unsupported_file_type_raises_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "image.jpg"
    file_path.write_bytes(b"fake image data")

    with pytest.raises(ValueError, match="Nicht unterstützter"):
        load_document(file_path)


def test_empty_pdf_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.pdf"

    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(file_path)
    pdf.close()

    with pytest.raises(ValueError, match="keinen erkennbaren Text"):
        load_pdf_file(file_path)