"""Tests de qlab.research.stats : exemples publiés, valeurs calculées à la main, cohérences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.research import stats as st


def test_moments_by_hand() -> None:
    """r = [0,01 ; 0,02 ; −0,01 ; 0] : moyenne 0,005 ; écart-type (ddof 1) √(5e-4 / 3) = 0,012910 ;
    Sharpe 0,38730 ; asymétrie 0 (symétrique) ; m₂ = 1,25e-4, m₄ = 2,5625e-8 ⇒ γ₄ = 1,64."""
    m = st.moments([0.01, 0.02, -0.01, 0.0])
    assert m.n == 4 and m.mean == pytest.approx(0.005)
    assert m.std == pytest.approx(math.sqrt(5e-4 / 3))
    assert m.sharpe == pytest.approx(0.005 / math.sqrt(5e-4 / 3))
    assert m.skew == pytest.approx(0.0, abs=1e-12)
    assert m.kurtosis == pytest.approx(1.64)


def test_normal_sample_moments() -> None:
    r = np.random.default_rng(0).normal(0.001, 0.02, 200_000)
    m = st.moments(r)
    assert m.skew == pytest.approx(0.0, abs=0.02) and m.kurtosis == pytest.approx(3.0, abs=0.05)


def test_annualize() -> None:
    assert st.annualize(0.1, 365) == pytest.approx(0.1 * math.sqrt(365))


# --- PSR, DSR : exemple publié -----------------------------------------------------------


def test_dsr_published_example() -> None:
    """Bailey & López de Prado (2014), « The Deflated Sharpe Ratio » : SR annualisé 2,5 sur
    1 250 jours (250/an), γ₃ = −3, γ₄ = 10, N = 100 essais, V[SR annualisé] = 0,5 ⇒
    SR* = 0,1132 (par jour) et DSR = 0,9004."""
    sr, var = 2.5 / math.sqrt(250), 0.5 / 250
    assert st.expected_max_sharpe(100, var) == pytest.approx(0.1132, abs=5e-5)
    assert st.dsr(sr, 1250, -3, 10, n_trials=100, sharpe_variance=var) == pytest.approx(
        0.9004, abs=5e-5
    )


def test_psr_by_hand() -> None:
    """SR 0,1 ; SR* 0 ; n 101 ; loi normale (γ₃ 0, γ₄ 3) : facteur 1 + 2/4 × 0,01 = 1,005 ;
    z = 0,1 × 10 / √1,005 = 0,99751 ⇒ Φ(z) = 0,8407."""
    assert st.psr(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(0.8407, abs=1e-4)
    assert st.psr(0.2, 0.2, 500, -1.0, 8.0) == pytest.approx(0.5)  # SR = SR* ⇒ 1/2


def test_negative_skew_and_fat_tails_lower_psr() -> None:
    """À Sharpe égal, une distribution à queue gauche épaisse est moins convaincante."""
    assert st.psr(0.1, 0.0, 101, -2.0, 12.0) < st.psr(0.1, 0.0, 101, 0.0, 3.0)


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert st.expected_max_sharpe(1, 0.01) == 0.0  # un seul essai : pas de sélection
    values = [st.expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert values == sorted(values) and values[0] > 0
    assert st.expected_max_sharpe(100, 0.0) == 0.0


# --- MinTRL et puissance -----------------------------------------------------------------


def test_min_track_record_by_hand() -> None:
    """SR 0,1 ; SR* 0 ; loi normale ; α 5 % : 1 + 1,005 × (1,64485 / 0,1)² = 272,91 obs."""
    assert st.min_track_record(0.1, 0.0, 0.0, 3.0, 0.05) == pytest.approx(272.91, abs=0.05)
    assert st.min_track_record(0.1, 0.1, 0.0, 3.0, 0.05) == math.inf


def test_min_detectable_is_consistent_with_min_trl() -> None:
    """Avec exactement MinTRL(SR) observations et une puissance de 50 %, le Sharpe minimal
    détectable retombe sur SR : les deux formules se répondent."""
    n = st.min_track_record(0.08, 0.02, -0.5, 5.0, 0.05)
    detectable = st.min_detectable_sharpe(round(n), 0.02, -0.5, 5.0, alpha=0.05, power=0.5)
    assert detectable == pytest.approx(0.08, abs=5e-4)


def test_more_data_detects_smaller_sharpe() -> None:
    small = st.min_detectable_sharpe(3650, 0.0, 0.0, 3.0, alpha=0.05, power=0.8)
    big = st.min_detectable_sharpe(365, 0.0, 0.0, 3.0, alpha=0.05, power=0.8)
    assert small < big
    # ≈ (z₀,₉₅ + z₀,₈₀) / √(n − 1) = 2,486 / √3649 = 0,04116 en loi normale, SR faible
    assert small == pytest.approx(2.486 / math.sqrt(3649), rel=0.01)


# --- Lo, Newey–West ----------------------------------------------------------------------


def test_lo_equals_naive_without_autocorrelation() -> None:
    r = np.random.default_rng(1).normal(0.001, 0.02, 20_000)
    naive = st.annualize(st.moments(r).sharpe, 365)
    assert st.lo_annualized_sharpe(r, 365, max_lag=5) == pytest.approx(naive, rel=0.05)


def test_lo_positive_autocorrelation_lowers_sharpe() -> None:
    """Rendements lissés (autocorrélation positive) : le Sharpe annualisé naïf surestime."""
    noise = np.random.default_rng(2).normal(0.001, 0.02, 20_000)
    smooth = np.convolve(noise, np.ones(5) / 5, mode="valid")
    naive = st.annualize(st.moments(smooth).sharpe, 365)
    assert st.lo_annualized_sharpe(smooth, 365, max_lag=10) < 0.6 * naive


def test_newey_west_by_hand() -> None:
    """r = [1, 2, 3, 4] ; e = [−1,5 ; −0,5 ; 0,5 ; 1,5] ; γ₀ = 5 / 4 = 1,25 ;
    γ₁ = (0,75 − 0,25 + 0,75) / 4 = 0,3125 ; S = 1,25 + 2 × ½ × 0,3125 = 1,5625 ;
    t = 2,5 / √(1,5625 / 4) = 4,0."""
    assert st.newey_west_tstat([1.0, 2.0, 3.0, 4.0], lags=1) == pytest.approx(4.0)
    assert st.newey_west_tstat([1.0, 2.0, 3.0, 4.0], lags=0) == pytest.approx(
        2.5 / math.sqrt(1.25 / 4)
    )
    assert st.newey_west_lags(1000) == 6  # ⌊4 × 10^(2/9)⌋ = ⌊6,67⌋


# --- tests d'hypothèses ------------------------------------------------------------------


def test_binomial_by_hand() -> None:
    """60 réussites sur 100 contre p* = 0,5 : P(X ≥ 60) = 0,02844."""
    assert st.binomial_pvalue(60, 100, 0.5) == pytest.approx(0.02844, abs=1e-5)
    assert st.binomial_pvalue(0, 10, 0.5) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("pvalues", "expected"),
    [
        ([0.01, 0.04, 0.03, 0.005], [True, True, True, True]),  # p₍₄₎ = 0,04 ≤ 4 × 0,05 / 4
        ([0.01, 0.02, 0.2, 0.5], [True, True, False, False]),  # seuils 0,0125 / 0,025 / …
        ([0.3, 0.2, 0.9], [False, False, False]),
        ([], []),
    ],
)
def test_benjamini_hochberg(pvalues: list[float], expected: list[bool]) -> None:
    assert st.benjamini_hochberg(pvalues, 0.05).tolist() == expected


# --- erreurs -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: st.moments([0.01, 0.01, 0.01]), "constants"),
        (lambda: st.moments([0.01]), "au moins 2"),
        (lambda: st.moments([0.01, float("nan")]), "non finies"),
        (lambda: st.psr(0.1, 0.0, 1, 0.0, 3.0), "2 observations"),
        (lambda: st.psr(1.0, 0.0, 100, 5.0, 3.0), "facteur de variance"),
        (lambda: st.expected_max_sharpe(0, 0.01), "n_trials"),
        (lambda: st.min_track_record(0.1, 0.0, 0.0, 3.0, 1.5), "alpha"),
        (lambda: st.binomial_pvalue(11, 10, 0.5), "test binomial"),
        (lambda: st.benjamini_hochberg([1.2], 0.05), "p-valeurs"),
        (lambda: st.annualize(0.1, 0), "periods_per_year"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
