import pytest

from src.file_security import (
    sanitize_filename,
    validate_uploaded_file,
)


def test_path_is_removed_from_filename() -> None:
    filename = sanitize_filename(
        "../../documents/manual.pdf"
    )

    assert filename == "manual.pdf"


def test_windows_path_is_removed() -> None:
    filename = sanitize_filename(
        r"C:\Users\Test\manual.pdf"
    )

    assert filename == "manual.pdf"


def test_invalid_characters_are_replaced() -> None:
    filename = sanitize_filename(
        'mein<tv>:"manual".pdf'
    )

    assert filename == (
        "mein_tv___manual_.pdf"
    )


def test_windows_reserved_name_is_prefixed() -> None:
    filename = sanitize_filename(
        "CON.txt"
    )

    assert filename == "_CON.txt"


def test_valid_pdf_is_accepted() -> None:
    filename = validate_uploaded_file(
        filename="manual.pdf",
        file_content=(
            b"%PDF-1.7\n"
            b"example content"
        ),
    )

    assert filename == "manual.pdf"


def test_fake_pdf_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="PDF-Dateikopf",
    ):
        validate_uploaded_file(
            filename="fake.pdf",
            file_content=b"This is not a PDF.",
        )


def test_valid_text_file_is_accepted() -> None:
    filename = validate_uploaded_file(
        filename="notes.txt",
        file_content=(
            "Ein gültiger Text."
            .encode("utf-8")
        ),
    )

    assert filename == "notes.txt"


def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="UTF-8",
    ):
        validate_uploaded_file(
            filename="notes.txt",
            file_content=b"\xff\xfe\xfa",
        )


def test_binary_text_file_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="binäre Daten",
    ):
        validate_uploaded_file(
            filename="binary.txt",
            file_content=b"Text\x00Binary",
        )


def test_empty_file_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="leer",
    ):
        validate_uploaded_file(
            filename="empty.txt",
            file_content=b"",
        )


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="nicht unterstützt",
    ):
        validate_uploaded_file(
            filename="program.exe",
            file_content=b"example",
        )


def test_large_file_is_rejected() -> None:
    file_content = b"A" * (
        1024 * 1024 + 1
    )

    with pytest.raises(
        ValueError,
        match="größer",
    ):
        validate_uploaded_file(
            filename="large.txt",
            file_content=file_content,
            max_size_mb=1,
        )


def test_invalid_maximum_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="größer als 0",
    ):
        validate_uploaded_file(
            filename="notes.txt",
            file_content=b"Text",
            max_size_mb=0,
        )