"""Okna walk-forward: chronologia, brak nachodzenia, identyczne dni testowe dla modeli.

Dlaczego: schemat oceny jest tak dobry, jak podział na okna. Testy pilnują, że dane uczące kończą się
przed dniami testowymi, że okno rośnie (rozszerzające), że każdy dzień testowy jest dokładnie w jednym oknie
i że dwa modele dostają identyczną listę dat (warunek uczciwego porównania i testu McNemara).
"""

import numpy as np
import pandas as pd

from src import config
from src.models import NaiveModel
from src.walk_forward import make_windows, run_model_on_windows


def make_table(start="2018-01-01", end="2021-06-30") -> pd.DataFrame:
    """Sztuczna tabela cech z losowymi zwrotami i celem przesuniętym o dzień (jak w features.py)."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"log_return": rng.normal(0, 0.02, len(dates))}, index=dates)
    table[config.TARGET] = table["log_return"].shift(-1)
    table.index.name = "date"
    return table.dropna()


def test_windows_are_chronological_and_do_not_overlap():
    """Każde okno: uczenie kończy się przed testem, zaczyna od pierwszego dnia danych; kolejne testy nie nachodzą."""
    windows = make_windows(make_table(), test_start="2020-01-01", test_end="2021-06-30")
    assert len(windows) == 6
    assert [w.name for w in windows] == ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"]
    previous_test_end = None
    for window in windows:
        assert window.train.index.max() < window.test.index.min()
        assert window.train.index.min() == pd.Timestamp("2018-01-01")  # okno rozszerzające
        if previous_test_end is not None:
            assert window.test.index.min() > previous_test_end
        previous_test_end = window.test.index.max()
    assert windows[0].test.index.min() >= pd.Timestamp("2020-01-01")
    assert windows[-1].test.index.max() <= pd.Timestamp("2021-06-30")


def test_training_set_grows_with_each_window():
    """Okno rozszerzające: liczba dni uczących rośnie ściśle z każdym kwartałem."""
    windows = make_windows(make_table(), test_start="2020-01-01", test_end="2020-12-31")
    sizes = [len(w.train) for w in windows]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_max_windows_limits_count():
    """Tryb --quick korzysta z ograniczenia liczby okien."""
    windows = make_windows(make_table(), test_start="2020-01-01", test_end="2021-06-30", max_windows=2)
    assert [w.name for w in windows] == ["2020Q1", "2020Q2"]


def test_all_test_days_covered_exactly_once():
    """Suma dni testowych ze wszystkich okien to dokładnie okres testowy, bez luk i powtórzeń."""
    table = make_table()
    windows = make_windows(table, test_start="2020-01-01", test_end="2021-06-30")
    covered = pd.concat([w.test for w in windows]).index
    expected = table[(table.index >= "2020-01-01") & (table.index <= "2021-06-30")].index
    assert covered.equals(expected)


def test_same_dates_for_every_model():
    """Dwa przebiegi na tych samych oknach dają identyczne daty i identyczne y_true (wymóg specyfikacji)."""
    windows = make_windows(make_table(), test_start="2020-01-01", test_end="2020-06-30")
    predictions_a, _ = run_model_on_windows(NaiveModel, windows, "naive")
    predictions_b, _ = run_model_on_windows(NaiveModel, windows, "naive_copy")
    assert predictions_a["date"].tolist() == predictions_b["date"].tolist()
    np.testing.assert_allclose(predictions_a["y_true"], predictions_b["y_true"])
