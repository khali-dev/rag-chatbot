## Anwendung starten

### Voraussetzungen

- Python 3.12
- Ollama
- Modell `qwen3:4b`

### Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt


ollama pull qwen3:4b
python -m streamlit run app.py

## Unterstützte Dokumente

Die Anwendung unterstützt:

- PDF-Dateien mit extrahierbarem Text
- UTF-8-Textdateien
- Dateien bis maximal 20 MB

Eingescannte PDFs ohne Textebene benötigen OCR und werden
in der aktuellen Version noch nicht unterstützt.

## Lokale Datenspeicherung

Hochgeladene Dokumente werden unter `data/documents`
gespeichert. Der Vektorindex von ChromaDB wird unter
`data/chroma` gespeichert.

Beide Verzeichnisse werden von Git ignoriert.