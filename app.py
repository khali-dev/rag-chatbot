from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import chromadb
import streamlit as st

from src.config import (
    DEBUG_MODE,
    DOCUMENTS_DIR,
    GEMINI_MODEL,
    LLM_PROVIDER,
    MAX_QUESTIONS_PER_SESSION,
    MAX_UPLOAD_SIZE_MB,
    OLLAMA_MODEL,
    create_data_directories,
    is_cloud_mode,
    validate_configuration,
)
from src.file_security import validate_uploaded_file
from src.rag_pipeline import (
    IndexingResult,
    RAGPipeline,
    RAGSource,
)
from src.vector_store import VectorStore


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)


def initialize_application() -> None:
    """Prüft die Konfiguration und erstellt Datenordner."""

    try:
        validate_configuration()

        if not is_cloud_mode():
            create_data_directories()

    except Exception as exc:
        st.error(
            "Die Anwendung ist nicht korrekt "
            "konfiguriert."
        )

        if DEBUG_MODE:
            st.code(
                f"{type(exc).__name__}: {exc}"
            )

        st.stop()


def initialize_session_state() -> None:
    """Initialisiert den Zustand der aktuellen Sitzung."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_indexing_result" not in st.session_state:
        st.session_state.last_indexing_result = None

    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0

    if "session_identifier" not in st.session_state:
        st.session_state.session_identifier = (
            uuid4().hex
        )


@st.cache_resource(show_spinner=False)
def get_local_pipeline() -> RAGPipeline:
    """Erstellt die dauerhafte lokale RAG-Pipeline."""

    return RAGPipeline()


def get_cloud_pipeline() -> RAGPipeline:
    """Erstellt eine getrennte Pipeline pro Cloud-Sitzung."""

    if "cloud_pipeline" not in st.session_state:
        session_identifier = (
            st.session_state.session_identifier
        )

        chroma_client = chromadb.EphemeralClient()

        vector_store = VectorStore(
            client=chroma_client,
            collection_name=(
                f"rag_session_{session_identifier}"
            ),
        )

        st.session_state.cloud_pipeline = (
            RAGPipeline(
                vector_store=vector_store
            )
        )

    return st.session_state.cloud_pipeline


def get_pipeline() -> RAGPipeline:
    """Gibt die passende lokale oder Cloud-Pipeline zurück."""

    if is_cloud_mode():
        return get_cloud_pipeline()

    return get_local_pipeline()


def save_uploaded_file(
    uploaded_file: Any,
    destination_directory: Path,
) -> Path:
    """Prüft und speichert eine hochgeladene Datei."""

    file_content = bytes(
        uploaded_file.getbuffer()
    )

    safe_filename = validate_uploaded_file(
        filename=uploaded_file.name,
        file_content=file_content,
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        destination_directory / safe_filename
    )

    destination.write_bytes(file_content)

    return destination


def index_uploaded_files(
    pipeline: RAGPipeline,
    uploaded_files: list[Any],
) -> IndexingResult:
    """Indexiert Uploads dauerhaft lokal oder temporär in der Cloud."""

    if is_cloud_mode():
        with TemporaryDirectory(
            prefix="rag_upload_"
        ) as temporary_directory:
            upload_directory = Path(
                temporary_directory
            )

            saved_paths = [
                save_uploaded_file(
                    uploaded_file=uploaded_file,
                    destination_directory=upload_directory,
                )
                for uploaded_file in uploaded_files
            ]

            return pipeline.index_documents(
                saved_paths
            )

    saved_paths = [
        save_uploaded_file(
            uploaded_file=uploaded_file,
            destination_directory=DOCUMENTS_DIR,
        )
        for uploaded_file in uploaded_files
    ]

    return pipeline.index_documents(
        saved_paths
    )


def display_error(
    action: str,
    error: Exception,
) -> None:
    """Zeigt eine sichere Fehlermeldung."""

    if isinstance(error, ValueError):
        st.error(f"{action}: {error}")
    else:
        st.error(
            f"{action} ist fehlgeschlagen."
        )

    if DEBUG_MODE:
        with st.expander(
            "Technische Details"
        ):
            st.code(
                f"{type(error).__name__}: {error}"
            )


def render_sources(
    sources: tuple[RAGSource, ...],
) -> None:
    """Zeigt die verwendeten Quellen."""

    if not sources:
        return

    with st.expander(
        f"Verwendete Quellen ({len(sources)})"
    ):
        for position, source in enumerate(
            sources
        ):
            if source.page is None:
                source_label = source.source
            else:
                source_label = (
                    f"{source.source}, "
                    f"Seite {source.page}"
                )

            st.markdown(
                f"**[Quelle {source.number}] "
                f"{source_label}**"
            )

            st.caption(
                "Semantische Ähnlichkeit: "
                f"{source.similarity:.3f}"
            )

            st.write(source.text)

            if position < len(sources) - 1:
                st.divider()


def render_chat_history() -> None:
    """Zeigt den bisherigen Chatverlauf."""

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":
                render_sources(
                    message.get(
                        "sources",
                        (),
                    )
                )


def render_indexing_result(
    result: IndexingResult,
) -> None:
    """Zeigt das letzte Indexierungsergebnis."""

    st.success(
        f"{result.indexed_documents} "
        f"Dokument(e) wurden zuletzt indexiert."
    )

    st.caption(
        f"{result.loaded_pages} Seiten · "
        f"{result.created_chunks} Chunks"
    )


def render_cloud_notice() -> None:
    """Zeigt den Datenschutzhinweis der Cloud-Version."""

    if not is_cloud_mode():
        return

    st.info(
        "Öffentliche Demo: Hochgeladene Dokumente "
        "werden nur für diese Sitzung verarbeitet. "
        "Für die Antwort werden relevante "
        "Textausschnitte an die Gemini API "
        "übermittelt. Lade keine vertraulichen "
        "oder personenbezogenen Dokumente hoch."
    )


def reset_chat_history() -> None:
    """Löscht ausschließlich den Chatverlauf."""

    st.session_state.messages = []
    st.rerun()


initialize_application()
initialize_session_state()

pipeline = get_pipeline()

if LLM_PROVIDER == "gemini":
    active_model = GEMINI_MODEL
    provider_label = "Google Gemini"
else:
    active_model = OLLAMA_MODEL
    provider_label = "Lokales Ollama"


with st.sidebar:
    st.header("Dokumente")

    st.write(
        "Lade PDF- oder Textdateien hoch und "
        "indexiere sie für die Suche."
    )

    if is_cloud_mode():
        st.caption(
            "Die Dokumente werden nur innerhalb "
            "deiner aktuellen Sitzung indexiert."
        )

    st.caption(
        f"Maximale Dateigröße: "
        f"{MAX_UPLOAD_SIZE_MB} MB"
    )

    uploaded_files = st.file_uploader(
        "Dokumente auswählen",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Ausgewählte Dateien:")

        for uploaded_file in uploaded_files:
            st.write(
                f"• {uploaded_file.name}"
            )

    index_button = st.button(
        "Dokumente indexieren",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if index_button:
        st.session_state.last_indexing_result = None

        try:
            with st.spinner(
                "Dokumente werden geprüft "
                "und verarbeitet ..."
            ):
                indexing_result = (
                    index_uploaded_files(
                        pipeline=pipeline,
                        uploaded_files=uploaded_files,
                    )
                )

                st.session_state.last_indexing_result = (
                    indexing_result
                )

        except Exception as exc:
            display_error(
                "Die Indexierung",
                exc,
            )

    last_result = (
        st.session_state.last_indexing_result
    )

    if last_result is not None:
        render_indexing_result(
            last_result
        )

    st.divider()
    st.subheader("Indexierte Dokumente")

    indexed_sources = (
        pipeline.vector_store.list_sources()
    )

    if indexed_sources:
        for source_name in indexed_sources:
            st.write(f"• {source_name}")
    else:
        st.caption(
            "Noch keine Dokumente indexiert."
        )

    st.metric(
        "Gespeicherte Chunks",
        pipeline.vector_store.count(),
    )

    if is_cloud_mode():
        remaining_questions = max(
            0,
            MAX_QUESTIONS_PER_SESSION
            - st.session_state.questions_asked,
        )

        st.metric(
            "Verbleibende Fragen",
            remaining_questions,
        )

    st.divider()

    if st.button(
        "Chatverlauf löschen",
        use_container_width=True,
    ):
        reset_chat_history()

    st.caption(
        f"LLM-Anbieter: {provider_label}"
    )

    st.caption(
        f"Sprachmodell: {active_model}"
    )

    if DEBUG_MODE:
        st.caption(
            "Entwicklungsmodus ist aktiviert."
        )


st.title("📚 RAG Document Assistant")

st.write(
    "Stelle Fragen zu deinen indexierten "
    "PDF- und Textdokumenten."
)

render_cloud_notice()

stored_chunks = (
    pipeline.vector_store.count()
)

question_limit_reached = (
    is_cloud_mode()
    and st.session_state.questions_asked
    >= MAX_QUESTIONS_PER_SESSION
)

if stored_chunks == 0:
    st.warning(
        "Es sind noch keine Dokumente indexiert. "
        "Lade zuerst ein Dokument über die "
        "Seitenleiste hoch."
    )
else:
    st.success(
        f"Die Wissensdatenbank enthält "
        f"{stored_chunks} Chunks."
    )

if question_limit_reached:
    st.warning(
        "Das Fragenlimit dieser Sitzung wurde "
        "erreicht. Starte für weitere Fragen "
        "eine neue Browsersitzung."
    )


render_chat_history()


question = st.chat_input(
    "Stelle eine Frage zu deinen Dokumenten ...",
    disabled=(
        stored_chunks == 0
        or question_limit_reached
    ),
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if is_cloud_mode():
                st.session_state.questions_asked += 1

            with st.spinner(
                "Suche Dokumentstellen und "
                "erzeuge Antwort ..."
            ):
                response = (
                    pipeline.answer_question(
                        question
                    )
                )

            st.markdown(response.answer)
            render_sources(response.sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                }
            )

        except Exception as exc:
            display_error(
                "Die Beantwortung der Frage",
                exc,
            )