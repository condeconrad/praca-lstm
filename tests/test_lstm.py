"""LSTM: skaler tylko na danych uczących, sekwencje bez przyszłości, strojenie z regułą odrzucenia.

Dlaczego: to w LSTM najłatwiej o wyciek (skaler na całości, okno sięgające w przód). Testy używają małej sieci
(8 neuronów, 2 epoki), żeby trwały sekundy, bo sprawdzają logikę przygotowania danych, nie jakość prognoz.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import LstmModel, make_sequences
from src.tuning import chosen_config_from_summary, summarize_tuning


def make_table(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Sztuczna tabela z cechami wariantu A i celem przesuniętym o dzień."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    table = pd.DataFrame(
        {
            "log_return": rng.normal(0, 0.02, n),
            "log_volume_change": rng.normal(0, 0.3, n),
        },
        index=dates,
    )
    table[config.TARGET] = table["log_return"].shift(-1)
    return table.dropna()


def test_make_sequences_shape_and_content():
    """Sekwencje mają kształt (liczba, okno, cechy), pierwsza zaczyna się od wiersza 0, ostatnia kończy na ostatnim."""
    features = np.arange(10).reshape(5, 2).astype(float)
    sequences = make_sequences(features, window_size=3)
    assert sequences.shape == (3, 3, 2)
    np.testing.assert_array_equal(sequences[0], features[0:3])
    np.testing.assert_array_equal(sequences[-1], features[2:5])


@pytest.fixture(scope="module")
def trained_model():
    """Jeden mały model uczony raz na moduł testowy (oszczędność czasu)."""
    table = make_table()
    train = table.iloc[:250]
    model = LstmModel(config.FEATURES_A, window_size=10, units=8, n_layers=1, seed=0, max_epochs=2)
    model.fit(train)
    return model, table


def test_scaler_fitted_only_on_training_part(trained_model):
    """Średnia skalera równa średniej z pierwszych 80% dni, a różna od średniej z całości (walidacja niewidziana)."""
    model, table = trained_model
    train = table.iloc[:250]
    n_fit_rows = int(round(250 * (1 - config.LSTM_VALIDATION_FRACTION)))
    expected_mean = train[config.FEATURES_A].iloc[:n_fit_rows].mean().to_numpy()
    np.testing.assert_allclose(model.feature_scaler.mean_, expected_mean)
    expected_target_mean = train[config.TARGET].iloc[:n_fit_rows].mean()
    assert model.target_scaler.mean_[0] == pytest.approx(expected_target_mean)
    # Średnia z całego zbioru uczącego jest inna, więc skaler nie widział walidacji.
    assert not np.allclose(model.feature_scaler.mean_, train[config.FEATURES_A].mean().to_numpy())


def test_predict_returns_one_value_per_test_day_and_ignores_future(trained_model):
    """Jedna prognoza na dzień; prognozy pierwszych 10 dni nie zależą od dni po nich."""
    model, table = trained_model
    test = table.iloc[250:]
    predictions = model.predict(test)
    assert predictions.shape == (len(test),)
    assert np.isfinite(predictions).all()
    # Prognozy dla pierwszych 10 dni nie zależą od dni późniejszych.
    predictions_short = model.predict(test.iloc[:10])
    np.testing.assert_allclose(predictions[:10], predictions_short, rtol=1e-5, atol=1e-7)
    assert model.n_parameters() > 0


def test_summarize_tuning_rejects_collapsed_models():
    """Kombinacja o najniższym MAE, ale zapadnięta (std 2% maksimum), odpada; wygrywa następna po MAE."""
    rows = []
    for window, mae, std in ((20, 0.010, 0.0001), (60, 0.012, 0.0050), (40, 0.011, 0.0040)):
        for seed in (0, 1):
            rows.append(
                {"variant": "A", "window": window, "units": 32, "layers": 1, "seed": seed, "mae": mae,
                 "direction_accuracy": 0.5, "pred_std": std, "epochs_trained": 10, "n_parameters": 100,
                 "fit_seconds": 1.0}
            )
    summary = summarize_tuning(pd.DataFrame(rows))
    # Najniższe MAE ma model zapadnięty (std = 2% maksimum), więc odpada; wygrywa okno 40.
    assert summary.loc[summary["window"] == 20, "collapsed"].item()
    chosen = chosen_config_from_summary(summary)
    assert chosen["A"]["window"] == 40
    assert chosen["A"]["all_collapsed"] is False
