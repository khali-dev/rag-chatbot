from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    DEBUG_MODE,
    DOCUMENTS_DIR,
    MAX_UPLOAD_SIZE_MB,
    OLLAMA_MODEL,
    create_data_directories,
)
from src.file_security import (
    validate_uploaded_file,
)
from src.rag_pipeline import (
    IndexingResult,
    RAGPipeline,
    RAGSource,
)


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

create_data_directories()


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """Erstellt und speichert die RAG-Pipeline."""

    return RAGPipeline()


def save_uploaded_file(
    uploaded_file: Any,
) -> Path:
    """Prüft und speichert eine hochgeladene Datei."""

    file_content = bytes(
        uploaded_file.getbuffer()
    )

    safe_filename = validate_uploaded_file(
        filename=uploaded_file.name,
        file_content=file_content,
    )

    destination = (
        DOCUMENTS_DIR / safe_filename
    )

    destination.write_bytes(file_content)

    return destination


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
        for position, source in enumerate(sources):
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


def initialize_session_state() -> None:
    """Initialisiert den Anwendungszustand."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_indexing_result" not in st.session_state:
        st.session_state.last_indexing_result = None


def render_chat_history() -> None:
    """Zeigt den bisherigen Chatverlauf."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(
                    message.get("sources", ())
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


pipeline = get_pipeline()
initialize_session_state()


with st.sidebar:
    st.header("Dokumente")

    st.write(
        "Lade PDF- oder Textdateien hoch und "
        "indexiere sie für die Suche."
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
                saved_paths = [
                    save_uploaded_file(uploaded_file)
                    for uploaded_file in uploaded_files
                ]

                indexing_result = (
                    pipeline.index_documents(
                        saved_paths
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
        render_indexing_result(last_result)

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

    st.divider()

    if st.button(
        "Chatverlauf löschen",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        f"Lokales Sprachmodell: {OLLAMA_MODEL}"
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

stored_chunks = pipeline.vector_store.count()

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


render_chat_history()


question = st.chat_input(
    "Stelle eine Frage zu deinen Dokumenten ...",
    disabled=stored_chunks == 0,
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