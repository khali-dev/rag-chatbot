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
    MAX_UPLOAD_SIZE_MB,
    OLLAMA_MODEL,
    create_data_directories,
    is_cloud_mode,
    validate_configuration,
)
from src.file_security import validate_uploaded_file
from src.llm import (
    generate_answer,
    generate_general_answer,
)
from src.rag_pipeline import (
    GENERAL_KNOWLEDGE_WARNING,
    IndexingResult,
    RAGPipeline,
    RAGSource,
)
from src.vector_store import VectorStore


st.set_page_config(
    page_title="RAG Document Assistant",
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

    if "session_identifier" not in st.session_state:
        st.session_state.session_identifier = (
            uuid4().hex
        )

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    if "allow_general_knowledge" not in st.session_state:
        st.session_state.allow_general_knowledge = False


def get_session_api_key() -> str:
    """Gibt den API-Schlüssel der Sitzung zurück."""

    api_key = st.session_state.get(
        "gemini_api_key",
        "",
    )

    if not isinstance(api_key, str):
        return ""

    return api_key.strip()


def has_session_api_key() -> bool:
    """Prüft, ob ein Sitzungsschlüssel vorhanden ist."""

    return bool(get_session_api_key())


def general_knowledge_enabled() -> bool:
    """Prüft, ob allgemeines Wissen erlaubt ist."""

    return bool(
        st.session_state.get(
            "allow_general_knowledge",
            False,
        )
    )


def generate_session_answer(
    question: str,
    context: str,
) -> str:
    """Erzeugt eine Dokumentantwort für die Sitzung."""

    return generate_answer(
        question=question,
        context=context,
        api_key=get_session_api_key(),
    )


def generate_session_general_answer(
    question: str,
) -> str:
    """Erzeugt eine allgemeine Antwort für die Sitzung."""

    return generate_general_answer(
        question=question,
        api_key=get_session_api_key(),
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
                vector_store=vector_store,
                answer_generator=generate_session_answer,
                fallback_answer_generator=(
                    generate_session_general_answer
                ),
            )
        )

    return st.session_state.cloud_pipeline


def get_pipeline() -> RAGPipeline:
    """Gibt die passende Pipeline zurück."""

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
    """Indexiert Uploads lokal oder temporär."""

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


def render_answer(
    content: str,
    answer_mode: str,
    sources: tuple[RAGSource, ...],
) -> None:
    """Zeigt Antwort, Warnung und Quellen."""

    if answer_mode == "general_knowledge":
        st.warning(
            GENERAL_KNOWLEDGE_WARNING
        )

    st.markdown(content)

    if answer_mode == "documents":
        render_sources(sources)


def render_chat_history() -> None:
    """Zeigt den bisherigen Chatverlauf."""

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            if message["role"] == "assistant":
                render_answer(
                    content=message["content"],
                    answer_mode=message.get(
                        "answer_mode",
                        "documents",
                    ),
                    sources=message.get(
                        "sources",
                        (),
                    ),
                )
            else:
                st.markdown(
                    message["content"]
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
    """Zeigt den Hinweis für die öffentliche App."""

    if not is_cloud_mode():
        return

    st.info(
        "Öffentliche Demo: Dokumente und API-Schlüssel "
        "werden nur innerhalb der aktuellen Sitzung "
        "verwendet. Relevante Textausschnitte werden "
        "zur Beantwortung an die Gemini API übertragen. "
        "Lade keine vertraulichen oder "
        "personenbezogenen Dokumente hoch."
    )


def reset_chat_history() -> None:
    """Löscht ausschließlich den Chatverlauf."""

    st.session_state.messages = []
    st.rerun()


def reset_cloud_session() -> None:
    """Löscht Schlüssel, Dokumentindex und Chat."""

    st.session_state.gemini_api_key = ""
    st.session_state.allow_general_knowledge = False
    st.session_state.messages = []
    st.session_state.last_indexing_result = None
    st.session_state.session_identifier = uuid4().hex

    if "cloud_pipeline" in st.session_state:
        del st.session_state.cloud_pipeline


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
    if is_cloud_mode():
        st.header("Gemini-Zugang")

        st.write(
            "Gib deinen eigenen Gemini-API-Schlüssel "
            "ein. Er wird nur für deine aktuelle "
            "Sitzung verwendet."
        )

        st.markdown(
            "[API-Schlüssel in Google AI Studio "
            "erstellen](https://aistudio.google.com/apikey)"
        )

        st.text_input(
            "Gemini-API-Schlüssel",
            type="password",
            key="gemini_api_key",
            placeholder="API-Schlüssel eingeben",
            help=(
                "Der Schlüssel wird im Sitzungszustand "
                "verwendet und nicht in ChromaDB oder "
                "einer Datei gespeichert."
            ),
        )

        if has_session_api_key():
            st.success(
                "API-Schlüssel wurde eingegeben."
            )
        else:
            st.warning(
                "Für Antworten wird ein eigener "
                "Gemini-API-Schlüssel benötigt."
            )

        st.button(
            "Sitzung und API-Schlüssel löschen",
            on_click=reset_cloud_session,
            use_container_width=True,
        )

        st.divider()

    st.header("Antwortmodus")

    st.toggle(
        "Allgemeines Modellwissen zulassen",
        key="allow_general_knowledge",
        help=(
            "Wenn die Dokumente keine Antwort "
            "enthalten, darf das Modell anhand "
            "seines allgemeinen Wissens antworten. "
            "Solche Antworten werden als ohne "
            "Gewähr gekennzeichnet."
        ),
    )

    if general_knowledge_enabled():
        st.caption(
            "Allgemeines Wissen ist aktiviert. "
            "Antworten ohne Dokumentquellen werden "
            "deutlich gekennzeichnet."
        )
    else:
        st.caption(
            "Es werden ausschließlich Antworten "
            "aus Dokumentquellen erzeugt."
        )

    st.divider()
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


st.title("RAG Document Assistant")

st.write(
    "Stelle Fragen zu deinen indexierten "
    "PDF- und Textdokumenten."
)

render_cloud_notice()

stored_chunks = (
    pipeline.vector_store.count()
)

api_key_missing = (
    is_cloud_mode()
    and not has_session_api_key()
)

knowledge_available = (
    stored_chunks > 0
    or general_knowledge_enabled()
)

if stored_chunks == 0:
    if general_knowledge_enabled():
        st.info(
            "Es sind keine Dokumente indexiert. "
            "Fragen werden momentan anhand des "
            "allgemeinen Modellwissens beantwortet."
        )
    else:
        st.warning(
            "Es sind noch keine Dokumente indexiert. "
            "Lade zuerst ein Dokument hoch oder "
            "aktiviere allgemeines Modellwissen."
        )
else:
    st.success(
        f"Die Wissensdatenbank enthält "
        f"{stored_chunks} Chunks."
    )

if api_key_missing:
    st.warning(
        "Gib in der Seitenleiste deinen eigenen "
        "Gemini-API-Schlüssel ein, um Fragen "
        "stellen zu können."
    )


render_chat_history()


question = st.chat_input(
    "Stelle eine Frage ...",
    disabled=(
        not knowledge_available
        or api_key_missing
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
            with st.spinner(
                "Suche Dokumentstellen und "
                "erzeuge Antwort ..."
            ):
                response = (
                    pipeline.answer_question(
                        question=question,
                        allow_general_knowledge=(
                            general_knowledge_enabled()
                        ),
                    )
                )

            render_answer(
                content=response.answer,
                answer_mode=response.answer_mode,
                sources=response.sources,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "answer_mode": (
                        response.answer_mode
                    ),
                }
            )

        except Exception as exc:
            display_error(
                "Die Beantwortung der Frage",
                exc,
            )