"""Wykrywanie dni nietypowych algorytmem Isolation Forest.

Co robi: oznacza dni testowe, które pod względem wielkości ruchu ceny, zmiany wolumenu i zmienności odstają
od tego, co widziano w danych uczących. Dlaczego istnieje: pytanie badawcze P4 ("czy modele mylą się bardziej
w dniach nietypowych"). Isolation Forest służy tu wyłącznie do analizy błędów, nigdy do usuwania dni
z szeregu (usunięcie zerwałoby ciągłość czasu i ukryło przed modelami krachy).

Cechy: wartość bezwzględna dziennego zwrotu, zmiana logarytmu wolumenu, zmienność z 20 dni.
Model dopasowujemy wyłącznie na danych uczących okna i oznaczamy nim dni testowe (bez wycieku).
Próg (contamination = 0,02) jest wyznaczany na danych uczących, więc odsetek dni
nietypowych w teście może być inny niż 2% (np. większy w 2020 r.).
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config
from src.walk_forward import Window


def fit_and_flag(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Dopasowuje Isolation Forest na `train` i oznacza dni z `test`.

    Zwraca tabelę: date, is_anomaly (True/False), anomaly_score (im niższy, tym bardziej nietypowy;
    wartości ujemne to dni oznaczone jako nietypowe).
    """
    forest = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.ANOMALY_RANDOM_STATE,
    )
    forest.fit(train[config.ANOMALY_FEATURES].to_numpy())  # "co jest normalne" uczymy się tylko z przeszłości
    test_features = test[config.ANOMALY_FEATURES].to_numpy()
    labels = forest.predict(test_features)  # -1 = nietypowy, 1 = zwykły (próg z contamination na danych uczących)
    scores = forest.decision_function(test_features)  # ciągły wynik: im niższy, tym łatwiej odizolować obserwację
    return pd.DataFrame(
        {"date": test.index, "is_anomaly": labels == -1, "anomaly_score": scores}
    )


def flag_windows(windows: list[Window]) -> pd.DataFrame:
    """Oznacza dni testowe w każdym oknie walk-forward (dopasowanie tylko na danych uczących okna)."""
    frames = []
    for window in windows:
        flagged = fit_and_flag(window.train, window.test)
        flagged["window"] = window.name
        frames.append(flagged)
    return pd.concat(frames, ignore_index=True)


def flag_descriptive(feature_table: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Analiza opisowa: dopasowanie i oznaczenie na tym samym okresie (np. 2015–2019).

    Używane tylko do wykresu opisowego, nigdy w ocenie modeli.
    """
    part = feature_table[(feature_table.index >= start) & (feature_table.index <= end)]
    return fit_and_flag(part, part)
