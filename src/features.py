"""Zwroty, cel prognozy i wskaźniki techniczne.

Co robi: z surowych cen i wolumenu buduje jedną tabelę "jeden wiersz = jedna sesja" z cechami wejściowymi
obu wariantów LSTM, cechami dla Isolation Forest i celem prognozy. Dlaczego istnieje: wszystkie modele muszą
widzieć dokładnie te same dni i te same, policzone w jeden sposób cechy; ten moduł jest jedynym miejscem,
które je liczy.

Zasada bez wycieku: każda cecha w dniu t jest liczona wyłącznie z danych do dnia t włącznie (operacje `shift(1)`,
`rolling`, `ewm` patrzą tylko wstecz). Cel `target` w dniu t to zwrot logarytmiczny z dnia t na t+1,
czyli jedyna kolumna, która "patrzy w przyszłość" (i jest tym, co prognozujemy). Test `test_no_leakage_from_future`
sprawdza, że zmiana danych po dniu t nie zmienia cech dnia t.

Wszystkie wskaźniki liczymy samodzielnie w pandas (bez bibliotek do analizy technicznej), żeby każdy wzór był widoczny.
"""

import numpy as np
import pandas as pd

from src import config


def log_returns(close: pd.Series) -> pd.Series:
    """Dzienny zwrot logarytmiczny: ln(Close_t / Close_{t-1}).

    Logarytm zamiast zwrotu procentowego, bo zwroty log. z kolejnych dni się dodają i są symetryczne
    (wzrost i spadek o ten sam czynnik mają przeciwne wartości). Pierwszy dzień nie ma poprzednika -> NaN.
    """
    return np.log(close / close.shift(1))


def log_volume_change(volume: pd.Series) -> pd.Series:
    """Zmiana logarytmu wolumenu: ln(V_t) - ln(V_{t-1}). Wolumen zerowy traktujemy jako brak.

    Dlaczego zmiana, a nie sam wolumen: wolumen NVDA rósł wielokrotnie przez 11 lat, więc jego poziom
    nie jest porównywalny między latami; względna zmiana z dnia na dzień jest.
    """
    safe_volume = volume.where(volume > 0)
    log_volume = np.log(safe_volume)
    return log_volume - log_volume.shift(1)


def close_to_sma(close: pd.Series, window: int) -> pd.Series:
    """Odległość ceny od średniej kroczącej: Close / SMA - 1.

    Iloraz zamiast różnicy, żeby cecha nie zależała od poziomu ceny (0,05 = cena 5% nad średnią, w 2015 i w 2025).
    `rolling(window)` używa tylko bieżącego dnia i `window - 1` dni wstecz.
    """
    sma = close.rolling(window).mean()
    return close / sma - 1


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI metodą Wildera.

    Średnie zyski i straty wygładzamy rekurencyjnie: avg_t = avg_{t-1} + (x_t - avg_{t-1}) / period,
    co odpowiada `ewm(alpha=1/period, adjust=False)`. Pierwsze `period` wartości odrzucamy (rozbieg).
    """
    change = close.diff()
    gain = change.clip(lower=0)   # dodatnie zmiany ceny (straty zerujemy)
    loss = (-change).clip(lower=0)  # ujemne zmiany jako liczby dodatnie
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss  # RS: ile razy średni zysk przewyższa średnią stratę
    rsi = 100 - 100 / (1 + relative_strength)  # przeskalowanie RS do przedziału 0–100
    # Gdy nie było żadnych strat, RSI = 100.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Zmienność: odchylenie standardowe zwrotów z ostatnich `window` dni ("jak nerwowy był ostatnio rynek")."""
    return returns.rolling(window).std()


def build_feature_table(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Buduje wspólną tabelę cech i celu dla badanej spółki.

    `stock` i `market` to tabele cen (kolumny Open, High, Low, Close, Volume) z datami w indeksie.
    Daty są łączone po części wspólnej. Wiersze z brakami (rozbieg wskaźników oraz ostatni dzień,
    który nie ma znanego zwrotu następnego dnia) są odrzucane raz, tutaj. Zwrócona tabela to
    "wspólny zbiór dni" dla wszystkich modeli.
    """
    close = stock["Close"]
    features = pd.DataFrame(index=stock.index)
    features["close"] = close  # sama cena nie jest cechą modeli; zostaje do wykresów i kontroli
    features["log_return"] = log_returns(close)
    features["abs_log_return"] = features["log_return"].abs()  # tylko dla Isolation Forest (wielkość ruchu bez znaku)
    features["log_volume_change"] = log_volume_change(stock["Volume"])
    features["close_to_sma20"] = close_to_sma(close, config.SMA_SHORT)
    features["close_to_sma50"] = close_to_sma(close, config.SMA_LONG)
    features["rsi14"] = rsi_wilder(close, config.RSI_PERIOD)
    features["volatility20"] = rolling_volatility(features["log_return"], config.VOLATILITY_WINDOW)

    # Cel: zwrot z dnia t na t+1 (znany dopiero po zamknięciu dnia t+1). Liczony PRZED złączeniem
    # z szeregiem rynkowym, żeby ewentualny brak dnia w QQQ nie przesunął celu o dodatkową sesję.
    features[config.TARGET] = features["log_return"].shift(-1)

    market_returns = log_returns(market["Close"]).rename("market_log_return")
    features = features.join(market_returns, how="inner")
    # Cel jako ostatnia kolumna (ten sam układ kolumn co w pierwotnej wersji tabeli).
    features = features[[column for column in features.columns if column != config.TARGET] + [config.TARGET]]

    # Jedno odrzucenie braków dla wszystkich modeli: rozbieg SMA50/RSI/zmienności na początku i ostatni dzień bez celu.
    features = features.dropna()
    features.index.name = "date"
    return features
