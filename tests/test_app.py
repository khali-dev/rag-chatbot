from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts() -> None:
    project_directory = (
        Path(__file__).resolve().parent.parent
    )

    app_path = project_directory / "app.py"

    app = AppTest.from_file(str(app_path))
    app.run(timeout=30)

    assert not app.exception
    assert len(app.title) == 1
    assert app.title[0].value == (
        "RAG Document Assistant"
    )
    assert len(app.chat_input) == 1