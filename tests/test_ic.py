"""Tests de qlab.research.ic : valeurs calculées à la main, loi connue, puissance, erreurs."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from qlab.core.errors import DataError
from qlab.research.ic import cross_sectional_ic, forward_returns, ic_decay, spearman
from qlab.research.stats import newey_west_tstat

NAN = float("nan")


def test_spearman_by_hand() -> None:
    """Rangs 1 2 3 4 contre 1 2 4 3 : Σd² = 2 ⇒ ρ = 1 − 6 × 2 / (4 × 15) = 0,8."""
    assert spearman([1, 2, 3, 4], [10, 20, 40, 30]) == pytest.approx(0.8)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_ties_use_average_ranks() -> None:
    """x = [1, 1, 2] → rangs [1,5 ; 1,5 ; 3] ; y → [1, 2, 3] : cov 1,5 / √(1,5 × 2) = 0,8660."""
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(1.5 / math.sqrt(3))


def test_spearman_ignores_missing_pairs_and_matches_scipy() -> None:
    assert spearman([1, NAN, 2, 3, 4], [10, 99, 20, 40, NAN]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=500), rng.normal(size=500)
    assert spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)


def test_spearman_undefined_cases_are_nan() -> None:
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))  # signal constant : aucun classement
    assert math.isnan(spearman([1, 2, NAN], [1, 2, 3]))  # 2 paires < 3
    assert math.isnan(spearman([1, 2, 3, 4], [1, 2, 3, 4], min_obs=5))


def test_forward_returns_by_hand() -> None:
    p = np.array([0.0, 0.1, 0.3, 0.2])
    assert np.allclose(forward_returns(p, 2), [0.3, 0.1, NAN, NAN], equal_nan=True)
    assert np.all(np.isnan(forward_returns(p, 4)))
    panel = forward_returns(np.column_stack([p, 2 * p]), 1)
    assert np.allclose(panel[:, 1], 2 * panel[:, 0], equal_nan=True)


def test_ic_decay_follows_known_law() -> None:
    """Signal = rendement de la période suivante, rendements iid : corrélation de Pearson avec
    r_{t→t+h} = 1/√h, donc Spearman = (6/π) asin(1 / (2√h)) en loi normale (h = 1 : 1)."""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, 200_000)
    log_p = np.concatenate([[0.0], np.cumsum(r)])
    signal = np.append(r, NAN)  # connu en t : r_{t→t+1} (fuite volontaire, loi exacte)
    points = ic_decay(signal, log_p, [4, 1, 16])
    assert [pt.horizon for pt in points] == [1, 4, 16]
    assert [pt.n for pt in points] == [200_000, 199_997, 199_985]
    for pt in points:
        expected = 6 / math.pi * math.asin(1 / (2 * math.sqrt(pt.horizon)))
        assert pt.ic == pytest.approx(expected, abs=0.005)


def test_cross_sectional_by_hand() -> None:
    """IC par date 1 ; 0,8 ; 0,6 (4ᵉ date : 2 actifs seulement ⇒ ignorée). Moyenne 0,8 ;
    écart-type 0,2 ⇒ ICIR 4 ; retards 0 : t = 0,8 / √(0,08 / 3 / 3) = 8,4853."""
    signal: list[list[float]] = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, NAN, NAN]]
    fwd = [[1, 2, 3, 4], [10, 20, 40, 30], [2, 1, 4, 3], [1, 2, 3, 4]]
    res = cross_sectional_ic(signal, fwd, horizon=1, min_assets=3, lags=0)
    assert np.allclose(res.per_date, [1.0, 0.8, 0.6, NAN], equal_nan=True)
    assert res.n_dates == 3 and res.mean == pytest.approx(0.8)
    assert res.icir == pytest.approx(4.0) and res.hit_rate == 1.0
    assert res.tstat == pytest.approx(0.8 / math.sqrt(0.08 / 9))


def _panel(
    ic_strength: float, seed: int, dates: int = 500, assets: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(dates, assets))
    fwd = ic_strength * signal + rng.normal(size=(dates, assets))
    signal[rng.random((dates, assets)) < 0.3] = NAN  # univers point-in-time incomplet
    return signal, fwd


def test_cross_sectional_detects_weak_signal_and_not_noise() -> None:
    """IC ≈ 0,05 sur 300 actifs × 500 dates : très significatif ; bruit pur : non."""
    weak = cross_sectional_ic(*_panel(0.05, 2), horizon=1, min_assets=20)
    noise = cross_sectional_ic(*_panel(0.0, 3), horizon=1, min_assets=20)
    assert weak.mean == pytest.approx(0.048, abs=0.01) and weak.tstat > 5
    assert abs(noise.mean) < 0.01 and abs(noise.tstat) < 3


def test_overlapping_horizon_forces_lags() -> None:
    """Rendements sur 10 périodes échantillonnés chaque période : IC successifs corrélés.
    Le t-stat sans retard gonfle ; la valeur par défaut prend au moins horizon − 1 retards."""
    rng = np.random.default_rng(4)
    log_p = np.cumsum(rng.normal(size=(600, 50)), axis=0)
    signal = rng.normal(size=(600, 50)).cumsum(axis=0)  # signal persistant, sans pouvoir
    fwd = forward_returns(log_p, 10)
    res = cross_sectional_ic(signal, fwd, horizon=10, min_assets=10)
    assert res.lags >= 9
    valid = res.per_date[np.isfinite(res.per_date)]
    assert abs(newey_west_tstat(valid, 0)) > 2 * abs(res.tstat)
    with pytest.raises(DataError, match="lags"):
        cross_sectional_ic(signal, fwd, horizon=10, min_assets=10, lags=0)


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: spearman([1, 2], [1, 2, 3]), "même longueur"),
        (lambda: spearman([1, math.inf, 2], [1, 2, 3]), "infinies"),
        (lambda: spearman([[1, 2]], [[1, 2]]), "dimension"),
        (lambda: forward_returns([0.0, 1.0], 0), "horizon"),
        (lambda: forward_returns([], 1), "non vide"),
        (lambda: ic_decay([1, 2, 3], [0, 1, 2], []), "horizons"),
        (lambda: ic_decay([1, 2, 3], [0, 1, 2], [1, 1]), "horizons"),
        (lambda: ic_decay([1, 2], [0, 1, 2], [1]), "même longueur"),
        (lambda: cross_sectional_ic([[1, 2, 3]], [[1, 2]], horizon=1, min_assets=3), "mêmes"),
        (lambda: cross_sectional_ic([[1, 2, 3]], [[1, 2, 3]], horizon=1, min_assets=2), "min_"),
        (lambda: cross_sectional_ic([[1, 2, 3]], [[1, 2, 3]], horizon=1, min_assets=3), "dates"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
