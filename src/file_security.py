import re

from src.config import MAX_UPLOAD_SIZE_MB


ALLOWED_EXTENSIONS = frozenset({
    ".pdf",
    ".txt",
})

WINDOWS_RESERVED_NAMES = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
})


def sanitize_filename(
    filename: str,
) -> str:
    """Entfernt Pfade und ungültige Zeichen."""

    normalized_name = (
        str(filename)
        .replace("\\", "/")
        .split("/")[-1]
        .strip()
    )

    sanitized_name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        normalized_name,
    )

    sanitized_name = sanitized_name.rstrip(
        ". "
    )

    if not sanitized_name:
        raise ValueError(
            "Der Dateiname ist ungültig."
        )

    filename_stem = sanitized_name.rsplit(
        ".",
        maxsplit=1,
    )[0].upper()

    if filename_stem in WINDOWS_RESERVED_NAMES:
        sanitized_name = f"_{sanitized_name}"

    return sanitized_name


def validate_uploaded_file(
    filename: str,
    file_content: bytes,
    max_size_mb: int = MAX_UPLOAD_SIZE_MB,
) -> str:
    """Prüft Dateiname, Größe und grundlegenden Inhalt."""

    if max_size_mb <= 0:
        raise ValueError(
            "Die maximale Uploadgröße "
            "muss größer als 0 sein."
        )

    safe_filename = sanitize_filename(
        filename
    )

    extension = (
        "." + safe_filename.rsplit(".", maxsplit=1)[-1].lower()
        if "." in safe_filename
        else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(
            sorted(ALLOWED_EXTENSIONS)
        )

        raise ValueError(
            f"Der Dateityp '{extension or 'ohne Endung'}' "
            f"wird nicht unterstützt. "
            f"Erlaubt sind: {supported}"
        )

    if not file_content:
        raise ValueError(
            f"Die Datei '{safe_filename}' ist leer."
        )

    maximum_bytes = (
        max_size_mb * 1024 * 1024
    )

    if len(file_content) > maximum_bytes:
        raise ValueError(
            f"Die Datei '{safe_filename}' ist größer "
            f"als {max_size_mb} MB."
        )

    if extension == ".pdf":
        first_bytes = file_content[:1024]

        if b"%PDF-" not in first_bytes:
            raise ValueError(
                f"Die Datei '{safe_filename}' besitzt "
                "keinen gültigen PDF-Dateikopf."
            )

    if extension == ".txt":
        try:
            decoded_text = file_content.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Die Textdatei '{safe_filename}' "
                "ist nicht als UTF-8 gespeichert."
            ) from exc

        if "\x00" in decoded_text:
            raise ValueError(
                f"Die Datei '{safe_filename}' enthält "
                "binäre Daten und ist keine gültige "
                "Textdatei."
            )

        if not decoded_text.strip():
            raise ValueError(
                f"Die Textdatei '{safe_filename}' "
                "enthält keinen Text."
            )

    return safe_filename