"""Tests de qlab.sizing.sizing : volatilités et facteurs calculés à la main, Kelly, invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.sizing import sizing as sz

ALT = np.array([0.01, -0.01, 0.01, -0.01])  # écart-type (ddof 1) : √(4e-4 / 3) = 0,011547
SD = math.sqrt(4e-4 / 3)


def test_portfolio_vol_by_hand() -> None:
    """Un actif à 50 % : 0,5 × 0,011547 × √4 = 0,011547. Deux actifs de sens opposés à 50/50 :
    risque nul. Rien de détenu : 0."""
    one = sz.portfolio_vol(np.array([0.5]), ALT[:, None], 4)
    assert one == pytest.approx(0.5 * SD * 2)
    hedge = sz.portfolio_vol(np.array([0.5, 0.5]), np.column_stack([ALT, -ALT]), 4)
    assert hedge == pytest.approx(0.0, abs=1e-12)
    assert sz.portfolio_vol(np.array([0.0, 0.0]), np.column_stack([ALT, ALT]), 4) == 0.0


def _series(n: int = 8) -> np.ndarray:
    return np.tile(ALT, n // 4)[:, None]


def test_vol_target_scales_down_by_hand() -> None:
    """Investi à 100 %, σ annuelle = 0,011547 × 2 = 0,023094 ; cible 0,01 ⇒ k = 0,433 ;
    fenêtre de 4 périodes : rien avant la 4ᵉ (k = 0, pas d'estimation)."""
    out = sz.vol_target(
        np.ones((8, 1)),
        _series(),
        target_vol_annual=0.01,
        w_max=1.0,
        lookback=4,
        periods_per_year=4,
    )
    assert out[:3, 0].tolist() == [0, 0, 0]
    assert out[3:, 0] == pytest.approx([0.01 / (SD * 2)] * 5)


def test_vol_target_never_scales_up_and_caps_exposure() -> None:
    """Cible très au-dessus du risque : k = 1 (jamais de levier), puis plafond w_max = 0,8 ;
    une poche de 30 % reste 30 % même si la cible permettrait plus."""
    kw = {"target_vol_annual": 10.0, "lookback": 4, "periods_per_year": 4}
    full = sz.vol_target(np.ones((8, 1)), _series(), w_max=1.0, **kw)  # type: ignore[arg-type]
    assert full[3:, 0].tolist() == [1.0] * 5
    capped = sz.vol_target(np.ones((8, 1)), _series(), w_max=0.8, **kw)  # type: ignore[arg-type]
    assert capped[3:, 0] == pytest.approx([0.8] * 5)
    sleeve = sz.vol_target(np.full((8, 1), 0.3), _series(), w_max=1.0, **kw)  # type: ignore[arg-type]
    assert sleeve[3:, 0] == pytest.approx([0.3] * 5)


def test_missing_returns_block_only_held_assets() -> None:
    """Trou dans A (détenu) : pas d'estimation tant qu'il est dans la fenêtre ; trou dans B
    (non détenu) : sans effet."""
    r = np.column_stack([_series()[:, 0], _series()[:, 0]])
    r[4, 0] = np.nan
    w = np.column_stack([np.ones(8), np.zeros(8)])
    out = sz.vol_target(w, r, target_vol_annual=10.0, w_max=1.0, lookback=4, periods_per_year=4)
    assert out[:, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    r2 = np.column_stack([_series()[:, 0], _series()[:, 0]])
    r2[4, 1] = np.nan
    out2 = sz.vol_target(w, r2, target_vol_annual=10.0, w_max=1.0, lookback=4, periods_per_year=4)
    assert out2[3:, 0].tolist() == [1.0] * 5


def test_future_returns_never_change_past_sizes() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.03, (60, 3))
    w = rng.dirichlet(np.ones(4), 60)[:, :3]
    kw = {"target_vol_annual": 0.3, "w_max": 1.0, "lookback": 20, "periods_per_year": 365}
    base = sz.vol_target(w, r, **kw)  # type: ignore[arg-type]
    shocked = r.copy()
    shocked[40:] *= 10
    assert np.array_equal(base[:40], sz.vol_target(w, shocked, **kw)[:40])  # type: ignore[arg-type]
    assert (base >= 0).all() and (base <= w + 1e-15).all()


def test_kelly_by_hand() -> None:
    """μ = 0,1 %, σ² = 0,04 % ⇒ f* = 2,5 ; p = 0,6, b = 1 ⇒ f* = 0,2 ; quart de Kelly de 2,5 =
    0,625, plafonné à 0,5 par la limite de risque ; Kelly négatif ⇒ 0 (pas de short)."""
    assert sz.kelly_continuous(0.001, 0.0004) == pytest.approx(2.5)
    assert sz.kelly_discrete(0.6, 1.0) == pytest.approx(0.2)
    assert sz.capped_kelly(2.5, fraction=0.25, limit=1.0) == pytest.approx(0.625)
    assert sz.capped_kelly(2.5, fraction=0.25, limit=0.5) == 0.5
    assert sz.capped_kelly(-1.0, fraction=0.25, limit=1.0) == 0.0


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: sz.portfolio_vol(np.ones(2), np.ones((1, 2)), 4), "fenêtre"),
        (
            lambda: sz.vol_target(
                np.ones((3, 1)),
                np.ones((3, 2)),
                target_vol_annual=0.1,
                w_max=1,
                lookback=2,
                periods_per_year=4,
            ),
            "même forme",
        ),
        (
            lambda: sz.vol_target(
                np.ones((3, 1)),
                np.ones((3, 1)),
                target_vol_annual=0.1,
                w_max=1.5,
                lookback=2,
                periods_per_year=4,
            ),
            "w_max",
        ),
        (
            lambda: sz.vol_target(
                np.full((3, 2), 0.6),
                np.ones((3, 2)),
                target_vol_annual=0.1,
                w_max=1,
                lookback=2,
                periods_per_year=4,
            ),
            "somme ≤ 1",
        ),
        (lambda: sz.kelly_continuous(0.1, 0.0), "variance"),
        (lambda: sz.kelly_discrete(1.0, 1.0), "p dans"),
        (lambda: sz.capped_kelly(1.0, fraction=0.0, limit=1.0), "fraction"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
