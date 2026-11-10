"""Symulacja prostych strategii inwestycyjnych z prowizją.

Co robi: zamienia prognozy na decyzje "akcje albo gotówka" i liczy dzień po dniu wartość portfela oraz miary
(zwrot, CAGR, obsunięcie, Sharpe). Dlaczego istnieje: pytanie P3 (czy prognozy mają wartość po kosztach) i kontrola
poprawności: model naiwny ze strategią podstawową MUSI dać dokładnie "kup i trzymaj". Moduł używa tylko numpy
i pandas, dzięki czemu aplikacja może przeliczać strategie suwakami bez TensorFlow.

Zasady (specyfikacja):
- decyzja po zamknięciu dnia t dotyczy zwrotu z dnia t+1 (czyli `y_true` dnia t),
- zwroty strategii to zwroty proste: exp(y) - 1,
- prowizja za każde wejście i wyjście z pozycji, także w pierwszym i ostatnim dniu okresu,
- gotówka bez odsetek, bez krótkiej sprzedaży i dźwigni.
"""

import numpy as np
import pandas as pd

from src import config


def positions_basic(y_pred) -> np.ndarray:
    """Strategia podstawowa: w akcjach, gdy prognoza >= 0 (ta sama reguła co kierunek "wzrost")."""
    return np.asarray(y_pred, dtype=float) >= 0


def positions_threshold(y_pred, threshold: float = config.THRESHOLD) -> np.ndarray:
    """Strategia z progiem: w akcjach, gdy prognoza > próg (ostra nierówność: prognoza 0 przy progu 0 to gotówka)."""
    return np.asarray(y_pred, dtype=float) > threshold


def positions_buy_and_hold(n_days: int) -> np.ndarray:
    """Kup i trzymaj: zawsze w akcjach."""
    return np.ones(n_days, dtype=bool)


def simulate(y_true, positions, commission: float = config.COMMISSION) -> tuple[np.ndarray, int]:
    """Symuluje wartość portfela (start = 1,0) dzień po dniu.

    `positions[i]` mówi, czy w dniu i (po zamknięciu) trzymamy akcje przez zwrot `y_true[i]`.
    Kolejność w każdym dniu: najpierw zmiana pozycji (prowizja), potem zwrot z akcji.
    Na końcu okresu pozycja jest zamykana (prowizja za wyjście), żeby wycena była porównywalna.
    Zwraca krzywą kapitału (wartość po każdym dniu) i liczbę transakcji.
    """
    y_true = np.asarray(y_true, dtype=float)
    positions = np.asarray(positions, dtype=bool)
    simple_returns = np.exp(y_true) - 1

    equity = np.empty(len(y_true))
    value = 1.0
    in_stock = False
    n_trades = 0
    for i in range(len(y_true)):
        if positions[i] != in_stock:
            value *= 1 - commission  # zmiana pozycji (wejście lub wyjście) kosztuje prowizję od całej wartości
            n_trades += 1
            in_stock = positions[i]
        if in_stock:
            value *= 1 + simple_returns[i]  # w gotówce wartość się nie zmienia (brak odsetek)
        equity[i] = value

    # Zamknięcie pozycji na koniec okresu: dzięki temu "kup i trzymaj" też płaci dwie prowizje
    # i równość "naiwny + podstawowa = kup i trzymaj" jest ścisła.
    if in_stock:
        value *= 1 - commission
        n_trades += 1
        equity[-1] = value
    return equity, n_trades


def max_drawdown(equity) -> float:
    """Maksymalne obsunięcie kapitału (liczba ujemna lub 0)."""
    equity = np.asarray(equity, dtype=float)
    # Szczyt liczony od wartości początkowej 1,0, więc spadek po pierwszej prowizji też jest obsunięciem.
    running_max = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdowns = equity / running_max - 1
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def strategy_metrics(equity, positions, n_trades: int) -> dict:
    """Miary strategii: zwrot całkowity, CAGR, obsunięcie, Sharpe, transakcje, dni w akcjach."""
    equity = np.asarray(equity, dtype=float)
    n_days = len(equity)
    daily_returns = np.diff(np.concatenate([[1.0], equity])) / np.concatenate([[1.0], equity])[:-1]
    std = daily_returns.std(ddof=1) if n_days > 1 else 0.0  # próbkowe odchylenie (ddof = 1) dla wskaźnika Sharpe'a
    sharpe = float(daily_returns.mean() / std * np.sqrt(config.TRADING_DAYS_PER_YEAR)) if std > 0 else float("nan")
    final_value = equity[-1]
    return {
        "total_return": float(final_value - 1),
        "cagr": float(final_value ** (config.TRADING_DAYS_PER_YEAR / n_days) - 1),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "n_trades": int(n_trades),
        "share_days_in_stock": float(np.mean(positions)),
        "final_value": float(final_value),
    }


def evaluate_strategies(
    predictions: pd.DataFrame,
    commission: float = config.COMMISSION,
    threshold: float = config.THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Liczy miary i krzywe kapitału dla "kup i trzymaj" oraz dwóch strategii każdego modelu.

    `predictions`: kolumny date, model, y_true, y_pred (wspólne daty dla wszystkich modeli).
    Zwraca (tabela miar, tabela krzywych kapitału z kolumną na strategię).
    """
    metrics_rows = []
    curves = {}

    models = list(predictions["model"].unique())
    # Rzeczywiste zwroty są wspólne dla wszystkich modeli (te same dni), więc bierzemy je z pierwszego.
    first_model = predictions[predictions["model"] == models[0]].sort_values("date")
    dates = first_model["date"].to_numpy()
    y_true = first_model["y_true"].to_numpy()

    positions = positions_buy_and_hold(len(y_true))
    equity, n_trades = simulate(y_true, positions, commission)
    metrics_rows.append({"model": "-", "strategy": "buy_and_hold", **strategy_metrics(equity, positions, n_trades)})
    curves["buy_and_hold"] = equity

    for model in models:
        part = predictions[predictions["model"] == model].sort_values("date")
        y_pred = part["y_pred"].to_numpy()
        for strategy_name, positions in (
            ("basic", positions_basic(y_pred)),
            ("threshold", positions_threshold(y_pred, threshold)),
        ):
            equity, n_trades = simulate(y_true, positions, commission)
            metrics_rows.append(
                {"model": model, "strategy": strategy_name, **strategy_metrics(equity, positions, n_trades)}
            )
            curves[f"{model}__{strategy_name}"] = equity

    equity_curves = pd.DataFrame(curves, index=pd.Index(dates, name="date"))
    return pd.DataFrame(metrics_rows), equity_curves
