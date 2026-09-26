"""Tests de qlab.research.bootstrap : cas exacts, déterminisme, couverture, autocorrélation."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.research.bootstrap import (
    bootstrap,
    sharpe,
    sharpe_difference,
    stationary_indices,
)


def _mean(r: np.ndarray) -> float:
    return float(np.mean(r))


def test_block_of_one_day_is_iid_bootstrap() -> None:
    """mean_block = 1 : chaque jour ouvre un bloc ⇒ tirages uniformes indépendants, exactement
    ceux du générateur après ses n tirages de Bernoulli."""
    idx = stationary_indices(50, 1, np.random.default_rng(7))
    ref = np.random.default_rng(7)
    ref.random(50)
    assert idx.tolist() == ref.integers(0, 50, size=50).tolist()


def test_single_giant_block_is_a_rotation() -> None:
    """Blocs de longueur « infinie » : un seul bloc ⇒ rotation de la série ⇒ moyenne identique
    sur chaque tirage."""
    r = np.random.default_rng(0).normal(0.001, 0.02, 100)
    idx = stationary_indices(100, 1e12, np.random.default_rng(3))
    assert sorted(idx.tolist()) == list(range(100))
    assert np.all(np.diff(idx) % 100 == 1)  # consécutifs, bouclés
    res = bootstrap(_mean, r, n_boot=50, mean_block=1e12, seed=1)
    assert np.allclose(res.samples, r.mean()) and res.estimate == pytest.approx(r.mean())


def test_mean_block_length() -> None:
    idx = stationary_indices(200_000, 10, np.random.default_rng(0))
    breaks = np.count_nonzero(np.diff(idx) % 200_000 != 1) + 1
    assert 200_000 / breaks == pytest.approx(10, rel=0.03)
    assert idx.min() >= 0 and idx.max() < 200_000


def test_deterministic_with_seed() -> None:
    r = np.random.default_rng(0).normal(0, 1, 300)
    a = bootstrap(_mean, r, n_boot=100, mean_block=5, seed=42)
    b = bootstrap(_mean, r, n_boot=100, mean_block=5, seed=42)
    c = bootstrap(_mean, r, n_boot=100, mean_block=5, seed=43)
    assert np.array_equal(a.samples, b.samples) and not np.array_equal(a.samples, c.samples)


def test_series_are_resampled_together() -> None:
    """Stratégie et référence tirées aux mêmes dates : bootstrap de la différence = bootstrap
    sur la série des différences (mêmes indices, même graine)."""
    rng = np.random.default_rng(1)
    strat, bench = rng.normal(0.002, 0.02, 400), rng.normal(0.001, 0.02, 400)
    paired = bootstrap(
        lambda a, b: float(np.mean(a - b)), strat, bench, n_boot=200, mean_block=10, seed=5
    )
    single = bootstrap(_mean, strat - bench, n_boot=200, mean_block=10, seed=5)
    assert np.allclose(paired.samples, single.samples)


def test_coverage_of_95_percent_interval() -> None:
    """Moyenne connue : l'intervalle à 95 % la contient environ 95 % du temps."""
    rng = np.random.default_rng(2)
    hits = 0
    for rep in range(200):
        r = rng.normal(0.01, 0.05, 250)
        low, high = bootstrap(_mean, r, n_boot=400, mean_block=5, seed=rep).interval(0.95)
        hits += low <= 0.01 <= high
    assert 0.88 <= hits / 200 <= 0.99


def test_autocorrelation_needs_blocks() -> None:
    """AR(1) φ = 0,8 : le bootstrap jour par jour sous-estime l'incertitude ; les blocs non."""
    rng = np.random.default_rng(3)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.8 * x[t - 1] + rng.normal(0, 0.01)
    iid = bootstrap(_mean, x, n_boot=500, mean_block=1, seed=0).interval(0.95)
    blocks = bootstrap(_mean, x, n_boot=500, mean_block=50, seed=0).interval(0.95)
    assert (blocks[1] - blocks[0]) > 1.8 * (iid[1] - iid[0])


def test_acceptance_criterion() -> None:
    rng = np.random.default_rng(4)
    good = bootstrap(_mean, rng.normal(0.01, 0.01, 500), n_boot=300, mean_block=5, seed=0)
    noise = bootstrap(_mean, rng.normal(0.0, 0.01, 500), n_boot=300, mean_block=5, seed=0)
    assert good.strictly_positive(0.95) and not noise.strictly_positive(0.95)


def test_sharpe_statistics() -> None:
    r = np.array([0.01, 0.02, -0.01, 0.0])
    assert sharpe(r) == pytest.approx(0.005 / np.std(r, ddof=1))
    assert sharpe_difference(r, r) == 0.0


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: stationary_indices(0, 5, np.random.default_rng(0)), "série vide"),
        (lambda: stationary_indices(10, 0.5, np.random.default_rng(0)), "mean_block"),
        (lambda: bootstrap(_mean, [1.0, 2.0], n_boot=0, mean_block=1, seed=0), "n_boot"),
        (lambda: bootstrap(_mean, n_boot=10, mean_block=1, seed=0), "au moins une série"),
        (
            lambda: bootstrap(lambda a, b: 0.0, [1.0, 2.0], [1.0], n_boot=10, mean_block=1, seed=0),
            "même longueur",
        ),
        (
            lambda: bootstrap(_mean, [1.0, float("nan")], n_boot=10, mean_block=1, seed=0),
            "non finies",
        ),
        (
            lambda: bootstrap(_mean, [1.0, 2.0], n_boot=10, mean_block=1, seed=0).interval(1.5),
            "niveau",
        ),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
