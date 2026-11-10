"""Strategie: prowizja na prostym przykładzie i równość naiwny = kup i trzymaj.

Dlaczego: symulacja dzień po dniu jest łatwa do pomylenia o jeden dzień albo o jedną prowizję;
testy liczą dwa–trzy dni ręcznie i sprawdzają kluczową równość wymaganą przez specyfikację.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    evaluate_strategies,
    max_drawdown,
    positions_basic,
    positions_buy_and_hold,
    simulate,
    strategy_metrics,
)


def test_commission_simple_example():
    """Wejście, dwa dni w akcjach, wyjście: wartość liczona ręcznie krok po kroku."""
    # Dwa dni, zwrot logarytmiczny ln(1.1) i ln(0.9), zawsze w akcjach, prowizja 1%.
    y_true = np.log([1.1, 0.9])
    equity, n_trades = simulate(y_true, [True, True], commission=0.01)
    # Wejście: 0.99; dzień 1: 0.99*1.1 = 1.089; dzień 2: 1.089*0.9 = 0.9801; wyjście: 0.9801*0.99
    assert equity[0] == pytest.approx(0.99 * 1.1)
    assert equity[1] == pytest.approx(0.99 * 1.1 * 0.9 * 0.99)
    assert n_trades == 2


def test_cash_days_have_no_return_and_switch_costs_commission():
    """Wyjście do gotówki kosztuje prowizję, a w gotówce wartość portfela stoi w miejscu."""
    y_true = np.log([1.1, 1.2, 0.5])
    positions = [True, False, False]
    equity, n_trades = simulate(y_true, positions, commission=0.01)
    assert equity[0] == pytest.approx(0.99 * 1.1)
    assert equity[1] == pytest.approx(0.99 * 1.1 * 0.99)  # wyjście do gotówki
    assert equity[2] == pytest.approx(equity[1])  # gotówka bez zmian
    assert n_trades == 2


def test_naive_basic_equals_buy_and_hold():
    """Model naiwny (prognoza 0) ze strategią podstawową daje dokładnie krzywą i miary "kup i trzymaj"."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    y_true = rng.normal(0, 0.02, 300)
    predictions = pd.DataFrame({"date": dates, "model": "naive", "y_true": y_true, "y_pred": 0.0})
    metrics, curves = evaluate_strategies(predictions)
    buy_and_hold = metrics[metrics["strategy"] == "buy_and_hold"].iloc[0]
    naive_basic = metrics[(metrics["model"] == "naive") & (metrics["strategy"] == "basic")].iloc[0]
    assert naive_basic["total_return"] == pytest.approx(buy_and_hold["total_return"])
    assert naive_basic["n_trades"] == buy_and_hold["n_trades"] == 2
    np.testing.assert_allclose(curves["naive__basic"], curves["buy_and_hold"])


def test_max_drawdown_hand_calculated():
    """Obsunięcie od szczytu 1,2 do dołka 0,9."""
    equity = [1.0, 1.2, 0.9, 1.0]
    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1)


def test_strategy_metrics_share_days_and_cagr():
    """Odsetek dni w akcjach, zwrot całkowity i CAGR na trzech dniach."""
    equity = np.array([1.0, 1.0, 1.21])
    positions = np.array([False, True, True])
    result = strategy_metrics(equity, positions, n_trades=2)
    assert result["share_days_in_stock"] == pytest.approx(2 / 3)
    assert result["total_return"] == pytest.approx(0.21)
    assert result["cagr"] == pytest.approx(1.21 ** (252 / 3) - 1)


def test_positions_rules():
    """Reguły pozycji: podstawowa wchodzi przy prognozie >= 0, kup i trzymaj zawsze w akcjach."""
    assert positions_basic([0.0, -0.001, 0.002]).tolist() == [True, False, True]
    assert positions_buy_and_hold(3).all()
