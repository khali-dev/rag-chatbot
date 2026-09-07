from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    DOCUMENTS_DIR,
    OLLAMA_MODEL,
    create_data_directories,
)
from src.rag_pipeline import RAGPipeline, RAGSource


st.set_page_config(
    page_title="RAG Document Assistant",
    layout="wide",
)

create_data_directories()


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """Erstellt eine Pipeline und verwendet sie bei erneuten Aufrufen weiter."""

    return RAGPipeline()


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Speichert eine hochgeladene Datei im Dokumentordner."""

    safe_filename = Path(uploaded_file.name).name
    extension = Path(safe_filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise ValueError(
            f"Nicht unterstützter Dateityp: {extension}"
        )

    destination = DOCUMENTS_DIR / safe_filename
    destination.write_bytes(
        bytes(uploaded_file.getbuffer())
    )

    return destination


def render_sources(
    sources: tuple[RAGSource, ...],
) -> None:
    """Zeigt die Quellen einer Antwort an."""

    if not sources:
        return

    with st.expander(
        f"Verwendete Quellen ({len(sources)})"
    ):
        for source in sources:
            if source.page is None:
                source_label = source.source
            else:
                source_label = (
                    f"{source.source}, Seite {source.page}"
                )

            st.markdown(
                f"**[Quelle {source.number}] "
                f"{source_label}**"
            )

            st.caption(
                f"Ähnlichkeit: {source.similarity:.3f}"
            )

            st.write(source.text)

            if source.number < len(sources):
                st.divider()


def initialize_session_state() -> None:
    """Initialisiert den Chatverlauf."""

    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    """Zeigt alle bisherigen Chatnachrichten an."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(
                    message.get("sources", ())
                )


pipeline = get_pipeline()
initialize_session_state()


# Seitenleiste
with st.sidebar:
    st.header("Dokumente")

    st.write(
        "Lade PDF- oder Textdateien hoch und "
        "indexiere sie für die Suche."
    )

    uploaded_files = st.file_uploader(
        "Dokumente auswählen",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Ausgewählte Dateien:")

        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")

    index_button = st.button(
        "Dokumente indexieren",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if index_button:
        try:
            with st.spinner(
                "Dokumente werden verarbeitet ..."
            ):
                saved_paths = [
                    save_uploaded_file(uploaded_file)
                    for uploaded_file in uploaded_files
                ]

                indexing_result = (
                    pipeline.index_documents(saved_paths)
                )

            st.success(
                f"{indexing_result.indexed_documents} "
                f"Dokument(e) erfolgreich indexiert."
            )

            st.write(
                f"Geladene Seiten: "
                f"{indexing_result.loaded_pages}"
            )

            st.write(
                f"Erstellte Chunks: "
                f"{indexing_result.created_chunks}"
            )

        except Exception as exc:
            st.error(
                f"Die Indexierung ist fehlgeschlagen: {exc}"
            )

    st.divider()

    st.metric(
        "Gespeicherte Chunks",
        pipeline.vector_store.count(),
    )

    if st.button(
        "Chatverlauf löschen",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        f"Lokales Sprachmodell: {OLLAMA_MODEL}"
    )


# Hauptbereich
st.title("RAG Document Assistant")

st.write(
    "Stelle Fragen zu deinen indexierten "
    "PDF- und Textdokumenten."
)

stored_chunks = pipeline.vector_store.count()

if stored_chunks == 0:
    st.warning(
        "Es sind noch keine Dokumente indexiert. "
        "Lade zuerst ein Dokument über die Seitenleiste hoch."
    )
else:
    st.success(
        f"Die Wissensdatenbank enthält "
        f"{stored_chunks} Chunks."
    )


render_chat_history()


question = st.chat_input(
    "Stelle eine Frage zu deinen Dokumenten ..."
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
                "Suche Dokumentstellen und erzeuge Antwort ..."
            ):
                response = pipeline.answer_question(
                    question
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
            error_message = (
                f"Bei der Verarbeitung ist ein Fehler "
                f"aufgetreten: {exc}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": (),
                }
            )