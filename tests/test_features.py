"""Testy cech: brak wycieku z przyszłości i poprawne wyrównanie celu.

Najważniejsze testy projektu: `test_target_is_next_day_log_return` (cel dnia t to zwrot z t na t+1)
i `test_no_leakage_from_future` (zmiana danych po dniu t nie zmienia cech dnia t). Bez nich cała ocena
modeli mogłaby być nieświadomym "podglądaniem przyszłości".
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_feature_table, log_returns, rsi_wilder


def make_prices(n_days: int, seed: int = 0) -> pd.DataFrame:
    """Losowy, ale powtarzalny szereg cen o `n_days` sesjach (dni robocze)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
    volume = rng.integers(1_000, 10_000, n_days).astype(float)
    frame = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume},
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def test_target_is_next_day_log_return():
    """Cel w dniu t musi być równy ln(Close_{t+1} / Close_t) policzonemu ręcznie z cen."""
    stock = make_prices(120)
    market = make_prices(120, seed=1)
    table = build_feature_table(stock, market)
    date = table.index[10]
    position = stock.index.get_loc(date)
    expected = np.log(stock["Close"].iloc[position + 1] / stock["Close"].iloc[position])
    assert table.loc[date, config.TARGET] == pytest.approx(expected)


def test_no_leakage_from_future():
    """Zmiana danych po dniu t nie może zmienić cech dnia t."""
    stock = make_prices(150)
    market = make_prices(150, seed=1)
    table_before = build_feature_table(stock, market)

    cut_date = table_before.index[60]
    cut_position = stock.index.get_loc(cut_date)
    stock_modified = stock.copy()
    market_modified = market.copy()
    stock_modified.iloc[cut_position + 1 :, :] *= 3.0  # zmieniamy całą przyszłość
    market_modified.iloc[cut_position + 1 :, :] *= 0.5
    table_after = build_feature_table(stock_modified, market_modified)

    feature_columns = config.FEATURES_B + ["abs_log_return"]
    for column in feature_columns:
        assert table_after.loc[cut_date, column] == pytest.approx(table_before.loc[cut_date, column])
    # Cel dnia t zależy od t+1, więc musi się zmienić (to jedyna kolumna patrząca w przyszłość).
    assert table_after.loc[cut_date, config.TARGET] != pytest.approx(table_before.loc[cut_date, config.TARGET])


def test_feature_table_has_no_missing_values_and_all_columns():
    """Po odrzuceniu rozbiegu tabela nie ma braków i zawiera wszystkie kolumny potrzebne modelom i Isolation Forest."""
    table = build_feature_table(make_prices(200), make_prices(200, seed=2))
    assert not table.isna().any().any()
    for column in config.FEATURES_B + [config.TARGET] + config.ANOMALY_FEATURES:
        assert column in table.columns


def test_warmup_rows_are_dropped():
    """Pierwszy wiersz tabeli to dzień, w którym SMA50 ma komplet danych; ostatni dzień odpada, bo nie ma celu."""
    stock = make_prices(200)
    table = build_feature_table(stock, make_prices(200, seed=2))
    # SMA50 potrzebuje 50 dni, ostatni dzień nie ma celu.
    assert table.index[0] == stock.index[config.SMA_LONG - 1]
    assert table.index[-1] == stock.index[-2]


def test_log_returns_simple_example():
    """Zwrot logarytmiczny na trzech cenach policzony ręcznie."""
    close = pd.Series([100.0, 110.0, 99.0])
    returns = log_returns(close)
    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(np.log(1.1))
    assert returns.iloc[2] == pytest.approx(np.log(0.9))


def test_rsi_bounds_and_all_gains():
    """RSI = 100 dla szeregu, który tylko rośnie; dla szeregu mieszanego zawsze w przedziale [0, 100]."""
    rising = pd.Series(np.arange(1.0, 40.0))
    rsi = rsi_wilder(rising, 14).dropna()
    assert rsi.to_numpy() == pytest.approx(100.0)
    mixed = pd.Series(100 + np.sin(np.arange(60)) * 5)
    rsi_mixed = rsi_wilder(mixed, 14).dropna()
    assert ((rsi_mixed >= 0) & (rsi_mixed <= 100)).all()
