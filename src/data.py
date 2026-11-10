"""Pobieranie, zapis i walidacja danych giełdowych.

Co robi: pobiera dzienne notowania NVDA i QQQ z yfinance (ceny skorygowane o podziały i dywidendy), zapisuje je
do `data/raw/<TICKER>.csv` wraz z `metadata.json`, sprawdza ich poprawność i opcjonalnie porównuje z plikami
kontrolnymi ze Stooq. Dlaczego istnieje: eksperyment ma być odtwarzalny, więc dane leżą w repozytorium i nie są
pobierane ponownie bez wyraźnego żądania (`--refresh`); walidacja wyłapuje typowe problemy (duplikaty, złe daty,
brak ostatnich sesji) zanim trafią do modeli.

Uruchomienie: `.venv/bin/python -m src.data [--refresh]`
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pobiera dzienne notowania z yfinance i zwraca tabelę z kolumnami PRICE_COLUMNS.

    Indeks: daty sesji (bez strefy czasowej). Parametr `end` w yfinance jest
    wyłączny (dzień `end` nie jest pobierany), sprawdzone w wersji z requirements.txt.
    """
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance nie zwrócił danych dla {ticker}")

    # Nowsze wersje yfinance zwracają kolumny dwupoziomowe (cena, ticker); spłaszczamy je.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw[PRICE_COLUMNS].copy()
    # Daty bez strefy czasowej i bez godziny: potem łączymy NVDA z QQQ po dacie, więc muszą być identyczne.
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices.index.name = "Date"
    return prices


def save_prices(prices: pd.DataFrame, path: Path) -> None:
    """Zapisuje tabelę cen do CSV z datą jako pierwszą kolumną (6 miejsc po przecinku wystarcza dla cen)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.6f")


def load_prices(path: Path) -> pd.DataFrame:
    """Wczytuje tabelę cen zapisaną przez `save_prices` (tylko kolumny cenowe, daty w indeksie)."""
    prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return prices[PRICE_COLUMNS]


def validate_prices(prices: pd.DataFrame, ticker: str) -> dict:
    """Sprawdza jakość danych i zwraca raport.

    Przerywa (ValueError), gdy: są duplikaty dat, daty nie rosną, ceny nie są dodatnie
    albo ostatnia sesja jest wcześniejsza niż `config.REQUIRED_LAST_SESSION`.
    Braki wartości są tylko raportowane (weekendy i święta to normalne dni bez sesji).
    """
    if prices.index.has_duplicates:
        raise ValueError(f"{ticker}: duplikaty dat w danych")
    if not prices.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: daty nie są uporządkowane rosnąco")

    # Cena niedodatnia uniemożliwiłaby logarytm w zwrocie; to zawsze błąd danych.
    close = prices["Close"].dropna()
    if (close <= 0).any():
        raise ValueError(f"{ticker}: wykryto niedodatnie ceny zamknięcia")

    # Ostatnia sesja 2025 r. potrzebuje zwrotu następnego dnia, więc dane muszą sięgać co najmniej 2026-01-02.
    last_session = prices.index.max()
    required = pd.Timestamp(config.REQUIRED_LAST_SESSION)
    if last_session < required:
        raise ValueError(
            f"{ticker}: ostatnia sesja w danych to {last_session.date()}, "
            f"a wymagana jest co najmniej {required.date()}. "
            "Pobierz dane ponownie z flagą --refresh."
        )

    missing = prices.isna().sum()
    return {
        "ticker": ticker,
        "rows": int(len(prices)),
        "first_date": str(prices.index.min().date()),
        "last_date": str(last_session.date()),
        "missing_values": {column: int(count) for column, count in missing.items()},
    }


def compare_with_control(prices: pd.DataFrame, control_path: Path) -> pd.DataFrame:
    """Porównuje ceny zamknięcia z plikiem kontrolnym ze Stooq dla wspólnych dat.

    Stooq zapisuje nagłówki po polsku (`Data`, `Zamkniecie`) albo po angielsku
    (`Date`, `Close`); obsługujemy oba warianty. Zwraca tabelę z różnicą bezwzględną
    i procentową. Różnice tylko raportujemy (mogą wynikać z innej korekty o dywidendy).
    """
    control = pd.read_csv(control_path)
    column_map = {"Data": "Date", "Zamkniecie": "Close", "Zamknięcie": "Close"}
    control = control.rename(columns=column_map)
    if "Date" not in control.columns or "Close" not in control.columns:
        raise ValueError(f"{control_path.name}: brak kolumn Date/Close (Data/Zamkniecie)")

    control["Date"] = pd.to_datetime(control["Date"])
    control = control.set_index("Date")[["Close"]].rename(columns={"Close": "close_control"})

    ours = prices[["Close"]].rename(columns={"Close": "close_yfinance"})
    merged = ours.join(control, how="inner")
    merged["abs_diff"] = (merged["close_yfinance"] - merged["close_control"]).abs()
    merged["pct_diff"] = merged["abs_diff"] / merged["close_control"] * 100
    merged.index.name = "date"
    return merged


def run_control_check(prices_by_ticker: dict) -> pd.DataFrame | None:
    """Jeśli w data/control/ są pliki CSV, porównuje je z danymi i zapisuje results/data_check.csv."""
    control_files = sorted(config.DATA_CONTROL_DIR.glob("*.csv"))
    if not control_files:
        print("Brak plików kontrolnych w data/control/ – kontrola ze Stooq pominięta.")
        return None

    reports = []
    for control_file in control_files:
        # Nazwa pliku Stooq zaczyna się od tickera, np. nvd_us_d.csv -> NVD.
        ticker = _match_ticker(control_file.name, prices_by_ticker.keys())
        if ticker is None:
            print(f"Nie rozpoznano tickera dla pliku {control_file.name} – pominięto.")
            continue
        report = compare_with_control(prices_by_ticker[ticker], control_file)
        report.insert(0, "ticker", ticker)
        report.insert(1, "control_file", control_file.name)
        reports.append(report)

    if not reports:
        return None
    combined = pd.concat(reports)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.RESULTS_DIR / "data_check.csv", float_format="%.6f")
    print(f"Zapisano raport kontroli: {config.RESULTS_DIR / 'data_check.csv'}")
    return combined


def _match_ticker(file_name: str, tickers) -> str | None:
    """Dopasowuje nazwę pliku kontrolnego do tickera (np. 'nvd_us_d.csv' -> 'NVDA').

    Stooq używa krótszych symboli (nvd zamiast NVDA), więc porównujemy prefiksy w obie strony.
    """
    prefix = file_name.lower().split("_")[0].split(".")[0]
    for ticker in tickers:
        if ticker.lower().startswith(prefix) or prefix.startswith(ticker.lower()):
            return ticker
    return None


def ensure_data(refresh: bool = False) -> dict:
    """Pobiera (gdy trzeba), zapisuje, waliduje i zwraca słownik {ticker: tabela cen}.

    Jedyny punkt wejścia do danych dla reszty programu: eksperyment i testy wołają tę funkcję,
    a ona decyduje, czy czytać z dysku, czy pobierać. Dzięki temu logika "nie pobieraj bez --refresh"
    jest w jednym miejscu.
    """
    import yfinance as yf

    tickers = config.TICKERS + [config.MARKET_TICKER]
    prices_by_ticker = {}
    metadata = {
        "yfinance_version": yf.__version__,
        "study_start": config.STUDY_START,
        "download_end_exclusive": config.DOWNLOAD_END,
        "auto_adjust": True,
        "tickers": {},
    }
    metadata_path = config.DATA_RAW_DIR / "metadata.json"
    if metadata_path.exists() and not refresh:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)

    for ticker in tickers:
        path = config.DATA_RAW_DIR / f"{ticker}.csv"
        if path.exists() and not refresh:
            print(f"{ticker}: używam istniejącego pliku {path.name}")
            prices = load_prices(path)
        else:
            print(f"{ticker}: pobieram z yfinance...")
            prices = download_prices(ticker, config.STUDY_START, config.DOWNLOAD_END)
            save_prices(prices, path)
            metadata["yfinance_version"] = yf.__version__
        report = validate_prices(prices, ticker)
        metadata["tickers"][ticker] = report
        prices_by_ticker[ticker] = prices
        print(
            f"{ticker}: {report['rows']} wierszy, {report['first_date']} – {report['last_date']}, "
            f"braki: {report['missing_values']}"
        )

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    run_control_check(prices_by_ticker)
    return prices_by_ticker


def main() -> None:
    """Uruchomienie modułu z linii poleceń: pobranie (lub odczyt) i walidacja danych."""
    parser = argparse.ArgumentParser(description="Pobieranie i walidacja danych NVDA i QQQ.")
    parser.add_argument("--refresh", action="store_true", help="pobierz dane ponownie")
    args = parser.parse_args()
    ensure_data(refresh=args.refresh)


if __name__ == "__main__":
    main()
