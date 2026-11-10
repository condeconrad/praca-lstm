"""Testy modułu danych (bez dostępu do internetu).

Sprawdzają walidację (duplikaty, porządek dat, ceny dodatnie, ostatnia sesja) i porównanie ze Stooq
na małych, ręcznie zbudowanych tabelach. Dlaczego: błąd w danych psuje wszystko dalej, a walidacja musi
przerywać program czytelnym komunikatem, a nie cicho przepuszczać złe dane.
"""

import pandas as pd
import pytest

from src import config
from src.data import compare_with_control, validate_prices


def make_prices(dates, close):
    """Buduje prostą tabelę cen o zadanych datach i cenach zamknięcia."""
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1000] * len(close),
        },
        index=pd.to_datetime(dates),
    )
    frame.index.name = "Date"
    return frame


def test_validate_accepts_correct_data():
    dates = ["2025-12-30", "2025-12-31", config.REQUIRED_LAST_SESSION]
    report = validate_prices(make_prices(dates, [10.0, 11.0, 12.0]), "TEST")
    assert report["rows"] == 3
    assert report["last_date"] == config.REQUIRED_LAST_SESSION


def test_validate_rejects_duplicates():
    dates = ["2025-12-31", "2025-12-31", config.REQUIRED_LAST_SESSION]
    with pytest.raises(ValueError, match="duplikaty"):
        validate_prices(make_prices(dates, [1.0, 1.0, 1.0]), "TEST")


def test_validate_rejects_unsorted_dates():
    dates = ["2025-12-31", "2025-12-30", config.REQUIRED_LAST_SESSION]
    with pytest.raises(ValueError, match="rosnąco"):
        validate_prices(make_prices(dates, [1.0, 1.0, 1.0]), "TEST")


def test_validate_rejects_non_positive_prices():
    dates = ["2025-12-30", "2025-12-31", config.REQUIRED_LAST_SESSION]
    with pytest.raises(ValueError, match="niedodatnie"):
        validate_prices(make_prices(dates, [1.0, 0.0, 1.0]), "TEST")


def test_validate_rejects_too_early_last_session():
    dates = ["2025-12-29", "2025-12-30", "2025-12-31"]
    with pytest.raises(ValueError, match="ostatnia sesja"):
        validate_prices(make_prices(dates, [1.0, 1.0, 1.0]), "TEST")


def test_compare_with_control_polish_headers(tmp_path):
    control_path = tmp_path / "nvd_us_d.csv"
    control_path.write_text(
        "Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen\n"
        "2025-12-30,1,1,1,10.0,100\n"
        "2025-12-31,1,1,1,12.0,100\n",
        encoding="utf-8",
    )
    ours = make_prices(["2025-12-30", "2025-12-31", "2026-01-02"], [10.0, 11.0, 12.0])
    report = compare_with_control(ours, control_path)
    assert len(report) == 2  # tylko wspólne daty
    assert report.loc["2025-12-30", "abs_diff"] == pytest.approx(0.0)
    assert report.loc["2025-12-31", "abs_diff"] == pytest.approx(1.0)
    assert report.loc["2025-12-31", "pct_diff"] == pytest.approx(100 / 12)
