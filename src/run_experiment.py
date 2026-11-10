"""Uruchamia cały eksperyment i zapisuje wyniki do katalogu results/.

Co robi: spina moduły w ustalonej kolejności (dane -> cechy -> okna -> rząd ARIMA -> strojenie -> modele w oknach
z zapisem do models/ -> Isolation Forest -> miary -> strategie -> run_info.json). Dlaczego istnieje: jedno polecenie
ma odtwarzać wszystkie wyniki, a każdy plik w results/ ma powstawać w ten sam sposób niezależnie od trybu.

Tryby:
  --quick  2 kwartały, 1 powtórzenie LSTM, 5 epok (sprawdzenie, że wszystko działa),
  --demo   24 kwartały, 1 powtórzenie (ziarno 0), pełne epoki (wyniki na prezentację),
  bez flagi: 24 kwartały, 5 powtórzeń (wyniki do pracy).
  --skip-tuning  użyj istniejącego results/chosen_config.json zamiast stroić od nowa.

Uruchomienie: `.venv/bin/python -m src.run_experiment --quick`
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.anomalies import flag_descriptive, flag_windows
from src.data import ensure_data
from src.features import build_feature_table
from src.metrics import binomial_test_accuracy, compute_metrics, mcnemar_exact, metrics_table
from src.models import ArimaModel, NaiveModel, select_arima_order
from src.strategy import evaluate_strategies
from src.walk_forward import make_windows, run_model_on_windows


def parse_args() -> argparse.Namespace:
    """Flagi linii poleceń; --quick i --demo wykluczają się, pełny tryb to brak flagi."""
    parser = argparse.ArgumentParser(description="Eksperyment: prognozowanie zwrotów NVDA.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="szybki test działania")
    group.add_argument("--demo", action="store_true", help="pełny okres, 1 powtórzenie")
    parser.add_argument("--skip-tuning", action="store_true", help="użyj istniejącego chosen_config.json")
    parser.add_argument("--refresh", action="store_true", help="pobierz dane ponownie")
    return parser.parse_args()


def mode_name(args: argparse.Namespace) -> str:
    """Nazwa trybu zapisywana w run_info.json i pokazywana w aplikacji."""
    if args.quick:
        return "quick"
    if args.demo:
        return "demo"
    return "full"


def save_json(data: dict, name: str) -> None:
    """Zapis JSON do results/ (default=str obsługuje daty i ścieżki)."""
    with open(config.RESULTS_DIR / name, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def lstm_saver(variant: str, seed: int):
    """Zwraca funkcję zapisującą wytrenowany LSTM do models/lstm/<wariant>/<kwartał>_seed<n>."""

    def save(model, window):
        model.save(config.MODELS_DIR / "lstm" / variant / f"{window.name}_seed{seed}")

    return save


def arima_saver(model, window) -> None:
    """Zapisuje parametry ARIMA okna do models/arima/<kwartał>.json."""
    model.save(config.MODELS_DIR / "arima" / f"{window.name}.json")


def save_csv(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    """Zapis CSV do results/ z 8 miejscami po przecinku (wystarcza do odtworzenia prognoz z dokładnością 1e-8)."""
    frame.to_csv(config.RESULTS_DIR / name, index=index, float_format="%.8f")


def choose_arima_order(features: pd.DataFrame) -> tuple[int, int, int]:
    """Wybiera rząd ARIMA kryterium AIC na latach 2015–2019 i zapisuje wynik."""
    selection = features[
        (features.index >= config.ARIMA_SELECTION_START) & (features.index <= config.ARIMA_SELECTION_END)
    ]
    chosen, table = select_arima_order(selection["log_return"])
    save_json(chosen, "arima_order.json")
    save_csv(table, "arima_order_search.csv")
    print(f"ARIMA: wybrano rząd (p, d, q) = ({chosen['p']}, 0, {chosen['q']}), AIC = {chosen['aic']:.1f}")
    return (chosen["p"], 0, chosen["q"])


def run_lstm_variants(windows, chosen_config: dict, seeds: list[int], max_epochs: int):
    """Uczy LSTM A i B dla każdego ziarna; zwraca prognozy uśrednione, prognozy ziaren i koszty."""
    from src.models import LstmModel

    averaged_frames, seed_frames, cost_frames = [], [], []
    for variant, feature_names in (("A", config.FEATURES_A), ("B", config.FEATURES_B)):
        label = f"lstm_{variant}"
        settings = chosen_config[variant]
        per_seed = []
        for seed in seeds:
            print(f"{label}, ziarno {seed}:")

            def factory(seed=seed, settings=settings, feature_names=feature_names):
                return LstmModel(
                    feature_names=feature_names,
                    window_size=settings["window"],
                    units=settings["units"],
                    n_layers=settings["layers"],
                    seed=seed,
                    max_epochs=max_epochs,
                )

            predictions, costs = run_model_on_windows(factory, windows, label, on_fitted=lstm_saver(variant, seed))
            predictions["seed"] = seed
            costs["seed"] = seed
            per_seed.append(predictions)
            cost_frames.append(costs)
        seeds_frame = pd.concat(per_seed, ignore_index=True)
        seed_frames.append(seeds_frame)
        # Prognoza końcowa LSTM = średnia prognoz z ziaren (y_true jest to samo dla każdego ziarna).
        averaged = seeds_frame.groupby(["date", "model", "window"], as_index=False, sort=True).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "mean")
        )
        averaged_frames.append(averaged[["date", "model", "y_true", "y_pred", "window"]])
    return (
        pd.concat(averaged_frames, ignore_index=True),
        pd.concat(seed_frames, ignore_index=True),
        pd.concat(cost_frames, ignore_index=True),
    )


def build_metrics(predictions: pd.DataFrame, costs: pd.DataFrame, seed_predictions: pd.DataFrame | None) -> pd.DataFrame:
    """Tabela miar dla całego okresu: miary, testy statystyczne, koszt, rozrzut między ziarnami."""
    naive = predictions[predictions["model"] == "naive"].sort_values("date")  # odniesienie dla testu McNemara
    rows = []
    for model, part in predictions.groupby("model", sort=False):
        part = part.sort_values("date")  # ta sama kolejność dat co w `naive`, bo dni są identyczne (test to sprawdza)
        row = {"model": model}
        row.update(compute_metrics(part["y_true"], part["y_pred"]))
        row.update(binomial_test_accuracy(part["y_true"], part["y_pred"]))
        row.update(mcnemar_exact(part["y_true"], part["y_pred"], naive["y_pred"]))
        model_costs = costs[costs["model"] == model]
        row["n_parameters"] = int(model_costs["n_parameters"].max()) if len(model_costs) else 0
        row["fit_seconds_mean"] = float(model_costs["fit_seconds"].mean()) if len(model_costs) else 0.0
        row["fit_seconds_total"] = float(model_costs["fit_seconds"].sum()) if len(model_costs) else 0.0
        row.update(seed_accuracy_summary(seed_predictions, model))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_accuracy_summary(seed_predictions: pd.DataFrame | None, model: str) -> dict:
    """Trafność kierunku każdego ziarna: min, max, średnia (tylko LSTM)."""
    empty = {"seed_accuracy_min": np.nan, "seed_accuracy_max": np.nan, "seed_accuracy_mean": np.nan}
    if seed_predictions is None:
        return empty
    part = seed_predictions[seed_predictions["model"] == model]
    if part.empty:
        return empty
    accuracies = [
        compute_metrics(seed_part["y_true"], seed_part["y_pred"])["direction_accuracy"]
        for _, seed_part in part.groupby("seed")
    ]
    return {
        "seed_accuracy_min": float(min(accuracies)),
        "seed_accuracy_max": float(max(accuracies)),
        "seed_accuracy_mean": float(np.mean(accuracies)),
    }


def build_seed_metrics(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    """Miary dla każdego ziarna osobno."""
    rows = []
    for (model, seed), part in seed_predictions.groupby(["model", "seed"], sort=True):
        row = {"model": model, "seed": seed}
        row.update(compute_metrics(part["y_true"], part["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_anomaly_metrics(predictions: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Miary osobno dla dni nietypowych i zwykłych (opisowo, bez wartości p).

    "Dzień nietypowy" to dzień t, po którego zamknięciu powstaje prognoza na t+1 (łączenie po tej samej dacie).
    """
    merged = predictions.merge(anomalies[["date", "is_anomaly"]], on="date", how="left")
    merged["day_type"] = np.where(merged["is_anomaly"], "anomalous", "normal")
    return metrics_table(merged, group_column="day_type")


def library_versions() -> dict:
    """Wersje bibliotek do run_info.json (odtwarzalność: te same wersje dają te same liczby)."""
    import scipy
    import sklearn
    import statsmodels
    import yfinance

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "yfinance": yfinance.__version__,
    }
    try:
        import keras
        import tensorflow

        versions["tensorflow"] = tensorflow.__version__
        versions["keras"] = keras.__version__
    except ImportError:
        pass
    return versions


def config_snapshot() -> dict:
    """Parametry z config.py, które mają trafić do run_info.json."""
    names = [
        "TICKERS", "MARKET_TICKER", "STUDY_START", "STUDY_END", "TEST_START", "TEST_END", "WINDOW_MONTHS",
        "FEATURES_A", "FEATURES_B", "LSTM_BATCH_SIZE", "LSTM_MAX_EPOCHS", "LSTM_PATIENCE", "LSTM_DROPOUT",
        "LSTM_VALIDATION_FRACTION", "TUNING_WINDOWS", "TUNING_UNITS", "TUNING_LAYERS", "TUNING_MIN_STD_FRACTION",
        "ANOMALY_FEATURES", "ANOMALY_CONTAMINATION", "ANOMALY_RANDOM_STATE", "COMMISSION", "THRESHOLD", "ALPHA",
    ]
    return {name: getattr(config, name) for name in names}


def main() -> None:
    """Cały eksperyment od danych do run_info.json; kolejność kroków opisana w docstringu modułu."""
    args = parse_args()
    mode = mode_name(args)
    settings = dict(config.RUN_MODES[mode])
    start_time = time.perf_counter()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Tryb: {mode}")

    prices = ensure_data(refresh=args.refresh)
    features = build_feature_table(prices[config.TICKERS[0]], prices[config.MARKET_TICKER])
    print(f"Tabela cech: {len(features)} dni, {features.index.min().date()} – {features.index.max().date()}")

    windows = make_windows(features, max_windows=settings["max_windows"])
    print(f"Okna walk-forward: {len(windows)} ({windows[0].name} – {windows[-1].name})")

    arima_order = choose_arima_order(features)

    from src.tuning import load_or_run_tuning  # import tutaj, żeby TensorFlow ładował się dopiero, gdy jest potrzebny

    chosen_config = load_or_run_tuning(
        features,
        skip=args.skip_tuning,
        seeds=settings["tuning_seeds"],
        max_epochs=settings["tuning_max_epochs"],
    )

    # Kopia wybranych ustawień obok modeli, żeby katalog models/ był samowystarczalny.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / "chosen_config.json", "w", encoding="utf-8") as handle:
        json.dump(chosen_config, handle, indent=2, ensure_ascii=False)

    print("Model naiwny:")
    naive_predictions, naive_costs = run_model_on_windows(NaiveModel, windows, "naive")
    print("ARIMA:")
    arima_predictions, arima_costs = run_model_on_windows(
        lambda: ArimaModel(arima_order), windows, "arima", on_fitted=arima_saver
    )
    lstm_predictions, seed_predictions, lstm_costs = run_lstm_variants(
        windows, chosen_config, settings["seeds"], settings["max_epochs"]
    )

    predictions = pd.concat([naive_predictions, arima_predictions, lstm_predictions], ignore_index=True)
    costs = pd.concat([naive_costs, arima_costs, lstm_costs], ignore_index=True)
    save_csv(predictions, "predictions.csv")
    save_csv(seed_predictions, "predictions_seeds.csv")
    save_csv(costs, "costs_by_window.csv")

    print("Isolation Forest:")
    anomalies = flag_windows(windows)
    save_csv(anomalies, "anomalies.csv")
    descriptive = flag_descriptive(features, config.STUDY_START, config.ARIMA_SELECTION_END)
    save_csv(descriptive, "anomalies_descriptive_2015_2019.csv")
    print(f"  dni nietypowych w teście: {int(anomalies['is_anomaly'].sum())} z {len(anomalies)}")

    print("Miary:")
    metrics = build_metrics(predictions, costs, seed_predictions)
    save_csv(metrics, "metrics.csv")
    predictions_with_year = predictions.assign(year=pd.to_datetime(predictions["date"]).dt.year)
    save_csv(metrics_table(predictions_with_year, group_column="year"), "metrics_by_year.csv")
    save_csv(build_anomaly_metrics(predictions, anomalies), "metrics_anomalies.csv")
    save_csv(build_seed_metrics(seed_predictions), "metrics_seeds.csv")
    print(metrics[["model", "mae", "direction_accuracy", "pred_std", "mcnemar_p_value", "binomial_p_value"]].to_string(index=False))

    print("Strategie:")
    strategies, equity_curves = evaluate_strategies(predictions)
    save_csv(strategies, "strategies.csv")
    save_csv(equity_curves, "equity_curves.csv", index=True)
    print(strategies[["model", "strategy", "total_return", "max_drawdown", "n_trades", "share_days_in_stock"]].to_string(index=False))

    run_info = {
        "mode": mode,
        "mode_description": {
            "quick": "Wyniki z trybu quick: 2 kwartały, 1 powtórzenie, 5 epok (tylko test działania)",
            "demo": "Wyniki z trybu demo: 1 powtórzenie (ziarno 0)",
            "full": "Wyniki pełne: 5 powtórzeń (ziarna 0–4)",
        }[mode],
        "duration_seconds": round(time.perf_counter() - start_time, 1),
        "n_windows": len(windows),
        "windows": [w.name for w in windows],
        "n_test_days": int((predictions["model"] == "naive").sum()),
        "seeds": settings["seeds"],
        "max_epochs": settings["max_epochs"],
        "arima_order": list(arima_order),
        "chosen_config": chosen_config,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": library_versions(),
        "parameters": config_snapshot(),
        "command": " ".join(sys.argv),
    }
    save_json(run_info, "run_info.json")
    print(f"Gotowe. Czas: {run_info['duration_seconds']} s. Wyniki w {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
