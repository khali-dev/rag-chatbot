import streamlit as st

from src.config import OLLAMA_MODEL, create_data_directories

create_data_directories()

st.set_page_config(page_title="RAG Dokument-Assistent",
                   layout="centered",   
)

st.title("RAG Dokument-Assistent")

st.write("Dieser Chatbot beatnwortet später Fragen anhand von hochgeladenen Textdokumenten")

st.success("Die Projektumgebung wurde erfolgreich eingerichtet.")

st.info(f"Konfiguriertes Sprachmodell: {OLLAMA_MODEL}")