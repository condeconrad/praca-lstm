"""Model naiwny, ARIMA i Isolation Forest na prostych danych.

Dlaczego: sprawdzamy kontrakt `fit`/`predict` (jedna prognoza na dzień testowy, brak zależności od dni późniejszych),
wybór rzędu ARIMA z siatki oraz to, że Isolation Forest wskazuje ewidentnie skrajny dzień.
"""

import numpy as np
import pandas as pd

from src import config
from src.anomalies import fit_and_flag
from src.models import ArimaModel, NaiveModel, select_arima_order


def make_table(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Sztuczna tabela z kolumnami używanymi przez ARIMA i Isolation Forest (zwroty z rozkładu normalnego)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    returns = rng.normal(0, 0.02, n)
    table = pd.DataFrame(
        {
            "log_return": returns,
            "abs_log_return": np.abs(returns),
            "log_volume_change": rng.normal(0, 0.3, n),
            "volatility20": pd.Series(returns).rolling(20).std().bfill().to_numpy(),
        },
        index=dates,
    )
    table[config.TARGET] = table["log_return"].shift(-1)
    return table.dropna()


def test_naive_predicts_zero():
    """Model naiwny zwraca same zera (kierunek: zawsze wzrost)."""
    table = make_table()
    model = NaiveModel()
    model.fit(table.iloc[:300])
    assert (model.predict(table.iloc[300:]) == 0).all()


def test_arima_predicts_one_value_per_test_day_and_uses_only_past():
    """ARIMA: jedna skończona prognoza na dzień; prognozy pierwszych 10 dni nie zależą od dni po nich."""
    table = make_table()
    train, test = table.iloc[:300], table.iloc[300:]
    model = ArimaModel(order=(1, 0, 0))
    model.fit(train)
    predictions = model.predict(test)
    assert len(predictions) == len(test)
    assert np.isfinite(predictions).all()
    assert model.n_parameters() == 3  # stała, AR(1), wariancja

    # Prognoza dla pierwszych 10 dni testowych nie może zależeć od dni po nich.
    predictions_short = model.predict(test.iloc[:10])
    np.testing.assert_allclose(predictions[:10], predictions_short)


def test_select_arima_order_returns_candidate_from_grid():
    """Wybór rzędu zwraca kombinację z siatki, d = 0, a tabela kandydatów jest posortowana po AIC."""
    chosen, table = select_arima_order(make_table()["log_return"], candidates=[0, 1])
    assert chosen["p"] in (0, 1) and chosen["q"] in (0, 1) and chosen["d"] == 0
    assert len(table) == 4
    assert table["aic"].is_monotonic_increasing


def test_isolation_forest_flags_extreme_day():
    """Dzień ze zwrotem 50% i skokiem wolumenu musi być oznaczony jako nietypowy, a większość dni jako zwykłe."""
    table = make_table()
    train = table.iloc[:300].copy()
    test = table.iloc[300:].copy()
    test.iloc[5, test.columns.get_loc("abs_log_return")] = 0.5  # ekstremalny dzień
    test.iloc[5, test.columns.get_loc("log_volume_change")] = 3.0
    flagged = fit_and_flag(train, test)
    assert len(flagged) == len(test)
    assert bool(flagged.iloc[5]["is_anomaly"])
    assert flagged["is_anomaly"].mean() < 0.5
