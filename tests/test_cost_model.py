"""Tests de qlab.costs.cost_model : chaque formule contre une valeur calculée à la main.

Repères : bid 99,99 / ask 100,01 ⇒ m = 100, s̃ = 0,02 / 100 = 0,0002 (2 pb) ;
f_taker = 0,1 %, slip = 0,01 % ⇒ c_taker = 0,002 + 0,0002 + 0,0002 = 0,0024 (24 pb) ;
un ordre : 0,001 + 0,0001 + 0,0001 = 0,0012.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.costs import cost_model as cm


def test_relative_spread_by_hand() -> None:
    assert cm.relative_spread(99.99, 100.01) == pytest.approx(0.0002)
    out = cm.relative_spread([99.99, 50.0], [100.01, 50.0])
    assert out == pytest.approx([0.0002, 0.0])  # spread nul accepté


def test_round_trip_taker_by_hand() -> None:
    assert cm.round_trip_taker(0.001, 0.0002, 0.0001) == pytest.approx(0.0024)
    assert cm.round_trip_taker(0.00075, 0.0, 0.0) == pytest.approx(0.0015)  # remise BNB


def test_vectorized_matches_scalar() -> None:
    spreads = np.array([0.0, 0.0001, 0.0002, 0.001])
    out = cm.round_trip_taker(0.001, spreads, 0.0001)
    assert out.shape == (4,)
    assert out == pytest.approx([cm.round_trip_taker(0.001, s, 0.0001) for s in spreads])


def test_round_trip_maker_and_one_way() -> None:
    assert cm.round_trip_maker(0.001, 0.0003) == pytest.approx(0.0023)
    assert cm.one_way_taker(0.001, 0.0002, 0.0001) == pytest.approx(0.0012)
    # un aller-retour taker = deux ordres simples
    assert 2 * cm.one_way_taker(0.001, 0.0002, 0.0001) == pytest.approx(
        cm.round_trip_taker(0.001, 0.0002, 0.0001)
    )


def test_breakeven_win_rate_by_hand() -> None:
    assert cm.breakeven_win_rate(0.01, 0.005) == pytest.approx(1 / 3)
    assert cm.breakeven_win_rate(0.01, 0.01) == pytest.approx(0.5)


def test_annual_drag_by_hand() -> None:
    """250 allers-retours à 24 pb : 60 % du capital par an partent en coûts."""
    assert cm.annual_drag(250, 0.0024) == pytest.approx(0.6)
    assert cm.annual_drag(0, 0.0024) == 0.0


def test_rebalance_cost_by_hand() -> None:
    """V = 50, on vend 10 % d'un actif pour en acheter 10 % d'un autre, 12 pb par ordre :
    50 × (0,1 + 0,1) × 0,0012 = 0,012."""
    assert cm.rebalance_cost(50.0, [0.1, -0.1, 0.0], [0.0012, 0.0012, 0.0012]) == pytest.approx(
        0.012
    )
    assert cm.rebalance_cost(50.0, [0.0, 0.0], [0.0012, 0.0012]) == 0.0


def test_valid_quotes_mask() -> None:
    mask = cm.valid_quotes(
        [100.0, 101.0, 0.0, float("nan"), 100.0], [100.1, 100.0, 1.0, 100.0, 100.0]
    )
    assert mask.tolist() == [True, False, False, False, True]


# --- cas d'erreur ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bid", "ask", "msg"),
    [
        (100.01, 99.99, "croisé"),
        (0.0, 1.0, "bid doit être > 0"),
        (float("nan"), 1.0, "non finies"),
        ([1.0, 2.0], [1.0], "formes différentes"),
    ],
)
def test_relative_spread_errors(bid: object, ask: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        cm.relative_spread(bid, ask)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: cm.round_trip_taker(-0.001, 0.0, 0.0), "fee_taker doit être ≥ 0"),
        (lambda: cm.round_trip_taker(0.001, float("inf"), 0.0), "non finies"),
        (lambda: cm.breakeven_win_rate(0.0, 0.01), "avg_win doit être > 0"),
        (lambda: cm.annual_drag(-1, 0.001), "trades_per_year"),
        (lambda: cm.rebalance_cost(50.0, [0.1], [0.001, 0.001]), "formes différentes"),
        (lambda: cm.rebalance_cost(50.0, [float("nan")], [0.001]), "non finies"),
        (lambda: cm.rebalance_cost(-1.0, [0.1], [0.001]), "portfolio_value"),
    ],
)
def test_domain_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
