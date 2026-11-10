"""Miary na prostych danych z ręcznie policzonym wynikiem.

Dlaczego: wzory MAE, RMSE, trafności i obu testów statystycznych muszą zgadzać się z rachunkiem "na kartce";
inaczej każda liczba w pracy byłaby niepewna.
"""

import numpy as np
import pytest

from src.metrics import binomial_test_accuracy, compute_metrics, direction, mcnemar_exact


def test_direction_rule_zero_is_up():
    """Reguła kierunku: zero liczy się jako wzrost (spójnie dla prognoz i rzeczywistości)."""
    assert direction([0.0, 0.01, -0.01]).tolist() == [True, True, False]


def test_compute_metrics_hand_calculated():
    """MAE, RMSE, trafność i odsetek prognoz wzrostu na czterech dniach policzonych ręcznie."""
    y_true = [0.01, -0.02, 0.03, -0.01]
    y_pred = [0.00, -0.01, -0.01, 0.01]
    # Błędy: -0.01, 0.01, -0.04, 0.02 -> MAE = 0.02, RMSE = sqrt((1+1+16+4)e-4/4) = sqrt(5.5e-4)
    result = compute_metrics(y_true, y_pred)
    assert result["mae"] == pytest.approx(0.02)
    assert result["rmse"] == pytest.approx(np.sqrt(5.5e-4))
    # Kierunki: true = [+,-,+,-], pred = [+ (0), -, -, +] -> trafione 2 z 4
    assert result["direction_accuracy"] == pytest.approx(0.5)
    assert result["share_up_predictions"] == pytest.approx(0.5)
    assert result["n_days"] == 4


def test_binomial_test_perfect_model():
    """Pięć trafień z pięciu: p = (1/2)^5 = 1/32."""
    y_true = [0.01, -0.02, 0.03, -0.01, 0.02]
    result = binomial_test_accuracy(y_true, y_true)
    assert result["n_hits"] == 5
    # P(5 z 5 przy p=0,5) = 1/32
    assert result["binomial_p_value"] == pytest.approx(1 / 32)


def test_mcnemar_counts_discordant_days():
    """Test McNemara liczy tylko dni, w których trafił dokładnie jeden model; p z rozkładu dwumianowego."""
    y_true = [0.01, -0.02, 0.03, -0.01]
    naive = [0.0, 0.0, 0.0, 0.0]  # trafia dni 1 i 3
    model = [0.01, -0.01, -0.01, -0.01]  # trafia dni 1, 2, 4
    result = mcnemar_exact(y_true, model, naive)
    assert result["model_only_correct"] == 2  # dni 2 i 4
    assert result["reference_only_correct"] == 1  # dzień 3
    assert result["n_discordant"] == 3
    # P(X >= 2 | n=3, p=0,5) = 4/8
    assert result["mcnemar_p_value"] == pytest.approx(0.5)


def test_mcnemar_identical_models():
    """Identyczne modele: zero dni niezgodnych i p = 1 (tak wygląda ARIMA(0,0,0) wobec naiwnego)."""
    y_true = [0.01, -0.02]
    result = mcnemar_exact(y_true, [0.0, 0.0], [0.0, 0.0])
    assert result["n_discordant"] == 0
    assert result["mcnemar_p_value"] == 1.0
