"""Test dymny aplikacji Streamlit: skrypt wykonuje się bez wyjątków na plikach z results/.

Dlaczego: aplikacja jest pokazywana na obronie; test wyłapuje błędy po zmianie formatu plików wyników
lub wersji Streamlit. Pomijany, gdy nie ma wyników (np. świeży klon bez uruchomionego eksperymentu).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (PROJECT_DIR / "results" / "predictions.csv").exists(),
    reason="brak wyników w results/ (uruchom run_experiment --quick)",
)
def test_app_runs_without_exceptions():
    """Pięć zakładek, tytuł zaczyna się od "Prognozowanie", brak wyjątków przy pierwszym renderowaniu."""
    app = AppTest.from_file(str(PROJECT_DIR / "app" / "app.py"), default_timeout=180).run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 5
    assert app.title[0].value.startswith("Prognozowanie")
