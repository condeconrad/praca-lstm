"""Zapis i wczytanie modeli: wczytany model daje identyczne prognozy jak oryginał.

Dlaczego: pliki w models/ mają pozwalać odtworzyć każdą prognozę bez ponownego uczenia; test sprawdza to
na małym modelu (LSTM i ARIMA) oraz, gdy są wyniki pełnego przebiegu, na zapisanym modelu 2025Q4 wobec
`results/predictions_seeds.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import ArimaModel, LstmModel
from tests.test_lstm import make_table

PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_lstm_save_and_load_gives_identical_predictions(tmp_path):
    """Sieć + skalery z dysku dają prognozy równe co do bitu i te same ustawienia."""
    table = make_table()
    train, test = table.iloc[:250], table.iloc[250:]
    model = LstmModel(config.FEATURES_A, window_size=10, units=8, n_layers=1, seed=0, max_epochs=2)
    model.fit(train)
    original = model.predict(test)

    model.save(tmp_path / "2019Q4_seed0")
    assert (tmp_path / "2019Q4_seed0.keras").exists()
    assert (tmp_path / "2019Q4_seed0_scalers.joblib").exists()

    loaded = LstmModel.load(tmp_path / "2019Q4_seed0")
    np.testing.assert_array_equal(loaded.predict(test), original)
    assert loaded.feature_names == model.feature_names
    assert loaded.window_size == model.window_size
    assert loaded.n_parameters() == model.n_parameters()


def test_arima_save_and_load_gives_identical_predictions(tmp_path):
    """Parametry z JSON + dane uczące (filter zamiast fit) dają te same prognozy co oryginał."""
    table = make_table()
    train, test = table.iloc[:250], table.iloc[250:]
    model = ArimaModel(order=(1, 0, 1))
    model.fit(train)
    original = model.predict(test)

    model.save(tmp_path / "2019Q4.json")
    loaded = ArimaModel.load(tmp_path / "2019Q4.json", train)
    np.testing.assert_allclose(loaded.predict(test), original, rtol=0, atol=1e-12)
    assert loaded.n_parameters() == model.n_parameters()


@pytest.mark.skipif(
    not (PROJECT_DIR / "models" / "lstm" / "A" / "2025Q4_seed0.keras").exists()
    or not (PROJECT_DIR / "results" / "predictions_seeds.csv").exists(),
    reason="brak zapisanych modeli z pełnego przebiegu",
)
def test_saved_last_window_model_reproduces_predictions_csv():
    """Model LSTM A (2025Q4, ziarno 0) z dysku odtwarza prognozy zapisane w predictions_seeds.csv."""
    from src.data import ensure_data
    from src.features import build_feature_table
    from src.walk_forward import make_windows

    prices = ensure_data()
    features = build_feature_table(prices[config.TICKERS[0]], prices[config.MARKET_TICKER])
    window = [w for w in make_windows(features) if w.name == "2025Q4"][0]
    model = LstmModel.load(PROJECT_DIR / "models" / "lstm" / "A" / "2025Q4_seed0")
    reproduced = model.predict(window.test)

    saved = pd.read_csv(PROJECT_DIR / "results" / "predictions_seeds.csv", parse_dates=["date"])
    saved = saved[(saved["model"] == "lstm_A") & (saved["seed"] == 0) & (saved["window"] == "2025Q4")]
    saved = saved.sort_values("date")
    assert list(saved["date"]) == list(window.test.index)
    # CSV ma 8 miejsc po przecinku, stąd tolerancja 1e-8.
    np.testing.assert_allclose(reproduced, saved["y_pred"].to_numpy(), atol=1e-8)
