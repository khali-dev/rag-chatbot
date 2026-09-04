from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


@dataclass(frozen=True)
class DocumentPage:
    """Ein geladener Textabschnitt mit Quelleninformationen."""

    text: str
    source: str
    page: int | None


def _require_file(file_path: str | Path) -> Path:
    """Prüft, ob der übergebene Pfad auf eine vorhandene Datei zeigt."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Die Datei wurde nicht gefunden: {path}")

    if not path.is_file():
        raise ValueError(f"Der Pfad zeigt nicht auf eine Datei: {path}")

    return path


def load_text_file(file_path: str | Path) -> list[DocumentPage]:
    """Lädt eine UTF-8-Textdatei."""

    path = _require_file(file_path)

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Es wurde keine Textdatei übergeben: {path.name}")

    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Die Textdatei ist nicht als UTF-8 gespeichert: {path.name}"
        ) from exc

    if not text:
        raise ValueError(f"Die Textdatei ist leer: {path.name}")

    return [
        DocumentPage(
            text=text,
            source=path.name,
            page=None,
        )
    ]


def load_pdf_file(file_path: str | Path) -> list[DocumentPage]:
    """Extrahiert den Text einer PDF seitenweise."""

    path = _require_file(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Es wurde keine PDF-Datei übergeben: {path.name}")

    pages: list[DocumentPage] = []

    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(
                    f"Die PDF ist passwortgeschützt: {path.name}"
                )

            for page_index, pdf_page in enumerate(pdf):
                text = pdf_page.get_text("text", sort=True).strip()

                if not text:
                    continue

                pages.append(
                    DocumentPage(
                        text=text,
                        source=path.name,
                        page=page_index + 1,
                    )
                )

    except pymupdf.FileDataError as exc:
        raise ValueError(
            f"Die PDF ist beschädigt oder ungültig: {path.name}"
        ) from exc

    if not pages:
        raise ValueError(
            f"Die PDF enthält keinen erkennbaren Text: {path.name}. "
            "Möglicherweise handelt es sich um ein eingescanntes Dokument."
        )

    return pages


def load_document(file_path: str | Path) -> list[DocumentPage]:
    """Wählt anhand der Dateiendung den passenden Loader aus."""

    path = _require_file(file_path)
    extension = path.suffix.lower()

    if extension == ".txt":
        return load_text_file(path)

    if extension == ".pdf":
        return load_pdf_file(path)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))

    raise ValueError(
        f"Nicht unterstützter Dateityp '{extension}' bei {path.name}. "
        f"Unterstützt werden: {supported}"
    )


def load_documents(
    file_paths: Iterable[str | Path],
) -> list[DocumentPage]:
    """Lädt mehrere Dokumente und führt ihre Seiten zusammen."""

    loaded_pages: list[DocumentPage] = []

    for file_path in file_paths:
        loaded_pages.extend(load_document(file_path))

    return loaded_pages