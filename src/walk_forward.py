"""Podział na okna walk-forward i pętla oceny modeli.

Co robi: dzieli wspólną tabelę cech na 24 okna kwartalne (dni uczące + dni testowe) i dla dowolnego modelu
uruchamia w każdym oknie cykl "świeży model -> ucz -> prognozuj -> zmierz czas". Dlaczego istnieje: uczciwa ocena
szeregu czasowego wymaga, by model nigdy nie widział dni późniejszych niż te, które prognozuje; ten moduł
jest jedynym miejscem, które o tym decyduje, i robi to tak samo dla każdego modelu.

Okno rozszerzające: na początku każdego kwartału testowego model uczy się na wszystkich
dniach od początku danych do ostatniego dnia przed kwartałem, a potem prognozuje każdy
dzień kwartału. Te same okna dla wszystkich modeli.
"""

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass
class Window:
    """Jedno okno walk-forward: nazwa (np. 2020Q1), dni uczące i dni testowe."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame


def make_windows(
    feature_table: pd.DataFrame,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
    window_months: int = config.WINDOW_MONTHS,
    max_windows: int | None = None,
) -> list[Window]:
    """Dzieli tabelę cech na okna walk-forward o długości `window_months` miesięcy."""
    windows = []
    period_start = pd.Timestamp(test_start)
    final_end = pd.Timestamp(test_end)
    while period_start <= final_end:
        # Koniec okna: ostatni dzień kalendarzowy kwartału (albo koniec okresu testowego, jeśli wcześniejszy).
        period_end = min(period_start + pd.DateOffset(months=window_months) - pd.Timedelta(days=1), final_end)
        # Uczenie: WSZYSTKO przed kwartałem (okno rozszerzające); test: tylko dni kwartału. Nierówność ostra
        # gwarantuje, że żaden dzień nie jest jednocześnie uczący i testowy.
        train = feature_table[feature_table.index < period_start]
        test = feature_table[(feature_table.index >= period_start) & (feature_table.index <= period_end)]
        if len(test) > 0:
            windows.append(Window(name=_window_name(period_start, window_months), train=train, test=test))
        period_start = period_start + pd.DateOffset(months=window_months)
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def _window_name(start: pd.Timestamp, window_months: int) -> str:
    """Czytelna nazwa okna do plików wyników: 2020Q1 dla kwartałów, 2020H1 dla półroczy, inaczej RRRR-MM."""
    if window_months == 3:
        return f"{start.year}Q{(start.month - 1) // 3 + 1}"
    if window_months == 6:
        return f"{start.year}H{(start.month - 1) // 6 + 1}"
    return start.strftime("%Y-%m")


def run_model_on_windows(
    model_factory, windows: list[Window], label: str, on_fitted=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uczy i ocenia jeden model w każdym oknie.

    `model_factory` to funkcja bez argumentów tworząca świeży model. `on_fitted` (opcjonalnie)
    to funkcja `(model, window)` wywoływana po uczeniu, np. do zapisania modelu na dysk;
    nie ma wpływu na prognozy. Zwraca:
    - prognozy: date, model, y_true, y_pred, window,
    - koszty: window, model, fit_seconds, n_parameters, n_train, n_test.
    """
    prediction_frames = []
    cost_rows = []
    for window in windows:
        model = model_factory()  # świeży model w każdym oknie: nic nie "przecieka" między kwartałami poza danymi
        model.timed_fit(window.train)
        if on_fitted is not None:
            on_fitted(model, window)  # np. zapis modelu na dysk; nie zmienia prognoz
        y_pred = model.predict(window.test)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "date": window.test.index,
                    "model": label,
                    "y_true": window.test[config.TARGET].to_numpy(),
                    "y_pred": y_pred,
                    "window": window.name,
                }
            )
        )
        cost_rows.append(
            {
                "window": window.name,
                "model": label,
                "fit_seconds": model.fit_seconds,
                "n_parameters": model.n_parameters(),
                "epochs_trained": getattr(model, "epochs_trained", None),
                "n_train": len(window.train),
                "n_test": len(window.test),
            }
        )
        print(f"  {label} {window.name}: {len(window.train)} dni uczących, {len(window.test)} testowych, "
              f"uczenie {model.fit_seconds:.1f} s")
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(cost_rows)
