"""Miary oceny prognoz: MAE, RMSE, trafność kierunku, test McNemara i dwumianowy.

Co robi: zamienia pary (y_true, y_pred) na liczby, którymi porównujemy modele, oraz liczy dwa testy
statystyczne. Dlaczego istnieje: te same wzory muszą być użyte dla każdego modelu, każdego roku, każdego ziarna
i w strojeniu; jeden moduł gwarantuje spójność (aplikacja też używa tych konwencji, np. ddof = 0 dla std prognoz).

Kierunek: wzrost, gdy zwrot >= 0; w przeciwnym razie spadek. Ta sama reguła dla prognoz i wartości
rzeczywistych (dzień z zerowym zwrotem to wzrost po obu stronach, więc model naiwny z prognozą 0 jest "wzrostowy").
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def direction(values) -> np.ndarray:
    """Zwraca tablicę True (wzrost) / False (spadek) według reguły zwrot >= 0."""
    return np.asarray(values, dtype=float) >= 0


def compute_metrics(y_true, y_pred) -> dict:
    """Podstawowe miary jakości prognozy dla jednego modelu.

    Oprócz błędów (MAE, RMSE) i trafności kierunku zwraca odsetek prognoz wzrostu, odchylenie standardowe
    i średnią prognoz: to kontrola, czy model nie "zapadł się" do prognozowania stałej (wtedy std ~ 0,
    odsetek wzrostu 100% i model jest kopią naiwnego). Odchylenie standardowe liczone z ddof = 0 (numpy).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    hits = direction(y_true) == direction(y_pred)  # True w dniach, gdy znak prognozy zgadza się z rzeczywistością
    return {
        "n_days": int(len(y_true)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "direction_accuracy": float(np.mean(hits)),
        "share_up_predictions": float(np.mean(direction(y_pred))),
        "pred_std": float(np.std(y_pred)),
        "pred_mean": float(np.mean(y_pred)),
    }


def binomial_test_accuracy(y_true, y_pred) -> dict:
    """Test dwumianowy: czy trafność kierunku jest większa niż 0,5 (jednostronnie)."""
    hits = direction(y_true) == direction(y_pred)
    n_hits = int(hits.sum())
    n_days = int(len(hits))
    # H0: trafność = 0,5 (rzut monetą); H1: trafność > 0,5. Uwaga: w okresie wzrostowym zdaje go też model naiwny.
    result = binomtest(n_hits, n_days, p=0.5, alternative="greater")
    return {"n_hits": n_hits, "n_days": n_days, "binomial_p_value": float(result.pvalue)}


def mcnemar_exact(y_true, y_pred_model, y_pred_reference) -> dict:
    """Dokładny test McNemara: czy model trafia kierunek częściej niż model odniesienia.

    Liczymy dni, w których dokładnie jeden z modeli trafił (dni niezgodne), i sprawdzamy
    testem dwumianowym (p = 0,5, jednostronnie), czy model wygrywa częściej niż przegrywa.
    """
    hits_model = direction(y_true) == direction(y_pred_model)
    hits_reference = direction(y_true) == direction(y_pred_reference)
    model_only = int(np.sum(hits_model & ~hits_reference))  # dni, w których trafił tylko badany model
    reference_only = int(np.sum(~hits_model & hits_reference))  # dni, w których trafił tylko model odniesienia
    n_discordant = model_only + reference_only  # dni zgodne (oba trafiły / oba chybiły) nic nie różnicują
    if n_discordant == 0:
        p_value = 1.0  # identyczne kierunki (np. ARIMA vs naiwny): brak podstaw do mówienia o przewadze
    else:
        p_value = float(binomtest(model_only, n_discordant, p=0.5, alternative="greater").pvalue)
    return {
        "model_only_correct": model_only,
        "reference_only_correct": reference_only,
        "n_discordant": n_discordant,
        "mcnemar_p_value": p_value,
    }


def metrics_table(predictions: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    """Liczy miary dla każdego modelu (i opcjonalnie dla każdej grupy, np. roku albo typu dnia).

    `predictions` ma kolumny: date, model, y_true, y_pred (i ewentualnie `group_column`).
    Rozbicia (rok, dni nietypowe) są opisowe: celowo bez wartości p, żeby uniknąć wielokrotnego testowania.
    """
    rows = []
    group_keys = ["model"] if group_column is None else ["model", group_column]
    for keys, part in predictions.groupby(group_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row.update(compute_metrics(part["y_true"], part["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)
