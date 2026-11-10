"""Strojenie ustawień LSTM (tylko raz, przed testem walk-forward).

Co robi: dla każdej kombinacji (okno, neurony, warstwy) i każdego wariantu cech uczy sieć na latach 2015–2018
i ocenia na 2019, a potem wybiera ustawienia do testu. Dlaczego istnieje: gdybyśmy dobierali ustawienia
na latach testowych, wybralibyśmy te "pasujące do odpowiedzi"; strojenie na danych sprzed testu i zamrożenie
wyboru to warunek uczciwej oceny.

Siatka: okno {20, 60} × neurony {32, 64} × warstwy {1, 2} = 8 kombinacji, osobno dla wariantu A i B.
Nauka na 2015–2018, ocena na 2019, 3 powtórzenia na kombinację.
Kryterium: najniższe MAE, ale kombinacje "zapadnięte" (odchylenie standardowe prognoz mniejsze
niż 20% najwyższego w siatce) odpadają. Jeśli odpadają wszystkie, wybieramy najwyższe odchylenie.
"""

import json
import time

import numpy as np
import pandas as pd

from src import config
from src.metrics import compute_metrics
from src.models import LstmModel


def tuning_grid() -> list[dict]:
    """Lista wszystkich kombinacji ustawień (2 × 2 × 2 = 8); mała siatka celowo, żeby nie przeuczyć wyboru do 2019 r."""
    grid = []
    for window in config.TUNING_WINDOWS:
        for units in config.TUNING_UNITS:
            for layers in config.TUNING_LAYERS:
                grid.append({"window": window, "units": units, "layers": layers})
    return grid


def run_tuning(features: pd.DataFrame, seeds: list[int], max_epochs: int) -> pd.DataFrame:
    """Uczy każdą kombinację dla każdego ziarna i zwraca tabelę wyników (jeden wiersz na ziarno).

    Kilka ziaren na kombinację, bo pojedynczy trening jest losowy; średnia z ziaren jest stabilniejszą podstawą wyboru.
    """
    train = features[(features.index >= config.TUNING_TRAIN_START) & (features.index <= config.TUNING_TRAIN_END)]
    evaluation = features[(features.index >= config.TUNING_EVAL_START) & (features.index <= config.TUNING_EVAL_END)]
    rows = []
    for variant, feature_names in (("A", config.FEATURES_A), ("B", config.FEATURES_B)):
        for combo in tuning_grid():
            for seed in seeds:
                model = LstmModel(
                    feature_names=feature_names,
                    window_size=combo["window"],
                    units=combo["units"],
                    n_layers=combo["layers"],
                    seed=seed,
                    max_epochs=max_epochs,
                )
                start = time.perf_counter()
                model.fit(train)
                fit_seconds = time.perf_counter() - start
                y_pred = model.predict(evaluation)
                result = compute_metrics(evaluation[config.TARGET], y_pred)
                rows.append(
                    {
                        "variant": variant,
                        **combo,
                        "seed": seed,
                        "mae": result["mae"],
                        "direction_accuracy": result["direction_accuracy"],
                        "pred_std": result["pred_std"],
                        "epochs_trained": model.epochs_trained,
                        "n_parameters": model.n_parameters(),
                        "fit_seconds": fit_seconds,
                    }
                )
                print(
                    f"  strojenie {variant} okno={combo['window']} neurony={combo['units']} warstwy={combo['layers']} "
                    f"ziarno={seed}: MAE={result['mae']:.5f}, trafność={result['direction_accuracy']:.3f}, "
                    f"std={result['pred_std']:.5f}, epoki={model.epochs_trained}, {fit_seconds:.0f} s"
                )
    return pd.DataFrame(rows)


def summarize_tuning(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Uśrednia wyniki po ziarnach i stosuje regułę odrzucenia modeli zapadniętych.

    Reguła: kombinacja, której średnie odchylenie standardowe prognoz jest mniejsze niż 20% najwyższego w siatce,
    odpada; spośród pozostałych wygrywa najniższe MAE. Jeśli odpadają wszystkie, bierzemy najwyższe odchylenie
    (i odnotowujemy to w `chosen_config.json` polem `all_collapsed`).
    """
    summary = (
        seed_results.groupby(["variant", "window", "units", "layers"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
            pred_std=("pred_std", "mean"),
            epochs_trained=("epochs_trained", "mean"),
            n_parameters=("n_parameters", "first"),
            fit_seconds=("fit_seconds", "mean"),
            n_seeds=("seed", "count"),
        )
    )
    summary["collapsed"] = False
    summary["chosen"] = False
    for variant, part in summary.groupby("variant"):
        # Próg liczony osobno dla A i B, bo warianty mają inną liczbę cech i inną naturalną skalę prognoz.
        threshold = config.TUNING_MIN_STD_FRACTION * part["pred_std"].max()
        collapsed = part["pred_std"] < threshold
        summary.loc[part.index, "collapsed"] = collapsed.to_numpy()
        candidates = part[~collapsed]
        if candidates.empty:
            chosen_index = part["pred_std"].idxmax()
        else:
            chosen_index = candidates["mae"].idxmin()
        summary.loc[chosen_index, "chosen"] = True
    return summary


def chosen_config_from_summary(summary: pd.DataFrame) -> dict:
    """Wybrane ustawienia dla wariantów A i B w formie słownika zapisywanego do chosen_config.json."""
    chosen = {}
    for _, row in summary[summary["chosen"]].iterrows():
        chosen[row["variant"]] = {
            "window": int(row["window"]),
            "units": int(row["units"]),
            "layers": int(row["layers"]),
            "mae_2019": float(row["mae"]),
            "direction_accuracy_2019": float(row["direction_accuracy"]),
            "pred_std_2019": float(row["pred_std"]),
            "all_collapsed": bool(summary[(summary["variant"] == row["variant"]) & ~summary["collapsed"]].empty),
        }
    return chosen


def load_or_run_tuning(features: pd.DataFrame, skip: bool, seeds: list[int], max_epochs: int) -> dict:
    """Wczytuje chosen_config.json (gdy `skip` i plik istnieje) albo uruchamia strojenie i zapisuje wyniki.

    Flaga `--skip-tuning` oszczędza kilka minut przy powtarzaniu testu z tymi samymi ustawieniami.
    """
    chosen_path = config.RESULTS_DIR / "chosen_config.json"
    if skip and chosen_path.exists():
        with open(chosen_path, encoding="utf-8") as handle:
            chosen = json.load(handle)
        print(f"Strojenie pominięte, używam {chosen_path.name}: {chosen}")
        return chosen

    print(f"Strojenie LSTM: {len(tuning_grid())} kombinacji × 2 warianty × {len(seeds)} ziaren, maks. {max_epochs} epok")
    seed_results = run_tuning(features, seeds, max_epochs)
    summary = summarize_tuning(seed_results)
    chosen = chosen_config_from_summary(summary)
    chosen["_info"] = {
        "seeds": seeds,
        "max_epochs": max_epochs,
        "train_period": [config.TUNING_TRAIN_START, config.TUNING_TRAIN_END],
        "eval_period": [config.TUNING_EVAL_START, config.TUNING_EVAL_END],
        "min_std_fraction": config.TUNING_MIN_STD_FRACTION,
    }
    seed_results.to_csv(config.RESULTS_DIR / "tuning_seeds.csv", index=False, float_format="%.8f")
    summary.to_csv(config.RESULTS_DIR / "tuning.csv", index=False, float_format="%.8f")
    with open(chosen_path, "w", encoding="utf-8") as handle:
        json.dump(chosen, handle, indent=2, ensure_ascii=False)
    for variant in ("A", "B"):
        print(f"Wybrane ustawienia {variant}: {chosen[variant]}")
    return chosen
