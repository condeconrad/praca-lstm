"""Wszystkie parametry eksperymentu w jednym miejscu.

Co robi: przechowuje stałe (daty, listy cech, siatkę strojenia, prowizję, tryby uruchomienia), które importują
pozostałe moduły. Dlaczego istnieje: żeby żaden parametr nie był "zaszyty" w kodzie i żeby zmiana jednej liczby
w jednym pliku zmieniała zachowanie całego eksperymentu (a `run_info.json` zapisywał, jakie wartości użyto).
Zwykły moduł Pythona zamiast pliku YAML: prościej, bez dodatkowej biblioteki, z podpowiedziami edytora.
"""

from pathlib import Path

# --- Ścieżki ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
DATA_CONTROL_DIR = PROJECT_DIR / "data" / "control"
RESULTS_DIR = PROJECT_DIR / "results"
# Wytrenowane modele z walk-forward (LSTM jako .keras + skalery, ARIMA jako JSON).
MODELS_DIR = PROJECT_DIR / "models"

# --- Dane ------------------------------------------------------------------
# Badany instrument. Lista, żeby kod dało się rozszerzyć na więcej spółek (docs/pomysly.md), ale w tej pracy tylko NVDA.
TICKERS = ["NVDA"]
# Instrument rynkowy używany wyłącznie jako cecha wejściowa wariantu B.
MARKET_TICKER = "QQQ"
# Okres badania.
STUDY_START = "2015-01-01"
STUDY_END = "2025-12-31"
# Dane pobieramy kilka sesji dalej, żeby ostatni dzień 2025 miał znany zwrot następnego dnia.
# W yfinance koniec zakresu jest wyłączny (dzień `end` nie jest pobierany), stąd 16 stycznia, nie 15.
DOWNLOAD_END = "2026-01-16"
# Ostatnia sesja w danych musi być co najmniej tego dnia, inaczej przerywamy.
REQUIRED_LAST_SESSION = "2026-01-02"

# --- Cechy -----------------------------------------------------------------
# Długości okien wskaźników technicznych (w sesjach). Wartości typowe w analizie technicznej; nie były strojone.
SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14
VOLATILITY_WINDOW = 20

# Nazwy cech dla wariantów A i B (kolumny tabeli cech z features.py).
FEATURES_A = ["log_return", "log_volume_change"]
FEATURES_B = FEATURES_A + [
    "close_to_sma20",
    "close_to_sma50",
    "rsi14",
    "volatility20",
    "market_log_return",
]
TARGET = "target"  # nazwa kolumny celu: zwrot logarytmiczny z dnia t na t+1 (jedyna kolumna "z przyszłości")

# --- Walk-forward ----------------------------------------------------------
TEST_START = "2020-01-01"
TEST_END = "2025-12-31"
# Liczba miesięcy w jednym oknie testowym (3 = kwartał).
WINDOW_MONTHS = 3

# --- ARIMA -----------------------------------------------------------------
# Kandydaci na rząd p (część AR) i q (część MA); d = 0, bo zwroty są już "zróżnicowanymi" cenami.
# Rząd wybierany raz, kryterium AIC, na latach 2015–2019 (poza okresem testowym), potem stały.
ARIMA_ORDER_SEARCH = [0, 1, 2]  # kandydaci na p i q
ARIMA_SELECTION_START = "2015-01-01"
ARIMA_SELECTION_END = "2019-12-31"

# --- LSTM ------------------------------------------------------------------
LSTM_BATCH_SIZE = 32          # liczba sekwencji na jedną aktualizację wag
LSTM_MAX_EPOCHS = 50          # górny limit; w praktyce wczesne zatrzymanie kończy uczenie po kilkunastu epokach
LSTM_PATIENCE = 5             # ile epok bez poprawy straty walidacyjnej tolerujemy przed zatrzymaniem
LSTM_DROPOUT = 0.2            # odsetek losowo wyłączanych neuronów w uczeniu (regularyzacja)
LSTM_VALIDATION_FRACTION = 0.2  # ostatnie 20% dni uczących (chronologicznie) tylko do wczesnego zatrzymania
LSTM_SEEDS = [0, 1, 2, 3, 4]  # 5 powtórzeń z różnymi ziarnami: pokazują, ile wyniku to losowość inicjalizacji

# Siatka strojenia.
TUNING_WINDOWS = [20, 60]
TUNING_UNITS = [32, 64]
TUNING_LAYERS = [1, 2]
TUNING_TRAIN_START = "2015-01-01"
TUNING_TRAIN_END = "2018-12-31"
TUNING_EVAL_START = "2019-01-01"
TUNING_EVAL_END = "2019-12-31"
TUNING_SEEDS = [0, 1, 2]
# Próg odrzucenia modeli "zapadniętych": std prognoz < 20% najwyższego std w siatce.
# Powód: na zaszumionych zwrotach najniższe MAE osiąga sieć prognozująca prawie stałe zero (kopia modelu naiwnego).
TUNING_MIN_STD_FRACTION = 0.2

# --- Isolation Forest ------------------------------------------------------
ANOMALY_FEATURES = ["abs_log_return", "log_volume_change", "volatility20"]  # wielkość ruchu, wolumen, zmienność
ANOMALY_CONTAMINATION = 0.02  # próg ustawiony tak, by 2% dni UCZĄCYCH było nietypowych (w teście odsetek może być inny)
ANOMALY_RANDOM_STATE = 42     # ziarno lasu losowego, dla powtarzalności

# --- Strategie -------------------------------------------------------------
COMMISSION = 0.001  # 0,1% wartości portfela przy każdym wejściu i wyjściu (także dla "kup i trzymaj")
THRESHOLD = 0.001  # strategia "z progiem" wchodzi na rynek tylko, gdy prognoza > 0,1%
TRADING_DAYS_PER_YEAR = 252  # przyjęta liczba sesji w roku (do CAGR i wskaźnika Sharpe'a)

# --- Statystyka ------------------------------------------------------------
ALPHA = 0.05  # poziom istotności testów (dwumianowego i McNemara)

# --- Tryby uruchomienia ----------------------------------------------------
# Każdy tryb nadpisuje część parametrów powyżej. Pełny tryb = brak nadpisań.
# --quick: sprawdzenie w minutę, że cały przepływ działa; --demo: pełny okres z jednym ziarnem (wyniki na prezentację);
# full: 5 ziaren (wyniki do pracy). Pliki wyników mają identyczny format w każdym trybie.
RUN_MODES = {
    "quick": {
        "max_windows": 2,
        "seeds": [0],
        "max_epochs": 5,
        "tuning_seeds": [0],
        "tuning_max_epochs": 5,
    },
    "demo": {
        "max_windows": None,
        "seeds": [0],
        "max_epochs": LSTM_MAX_EPOCHS,
        "tuning_seeds": TUNING_SEEDS,
        "tuning_max_epochs": LSTM_MAX_EPOCHS,
    },
    "full": {
        "max_windows": None,
        "seeds": LSTM_SEEDS,
        "max_epochs": LSTM_MAX_EPOCHS,
        "tuning_seeds": TUNING_SEEDS,
        "tuning_max_epochs": LSTM_MAX_EPOCHS,
    },
}
