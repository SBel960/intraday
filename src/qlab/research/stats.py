"""Statistiques de validation (SPEC_INTRADAY §7, SPEC_LONG_TERME LT.6) : fonctions pures.

Conventions :
- rendements par période en ``float64`` (fractions), ``SR`` = Sharpe **par période** (moyenne /
  écart-type, ddof = 1) sauf mention « annualisé » ;
- kurtosis **de Pearson** (3 pour une loi normale), asymétrie standardisée : moments d'échantillon
  « population » (biais négligeable devant les centaines d'observations utilisées) ;
- marché 24/7 : ``periods_per_year`` = 365 en journalier, 8 760 en horaire.

Formules (références entre parenthèses) :

- PSR(SR*) = Φ( (SR − SR*) √(n−1) / √(1 − γ₃ SR + (γ₄ − 1)/4 · SR²) )   (Bailey, LdP 2012)
- SR* attendu par hasard après N essais :
  √V[SR_k] · [ (1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)) ],  γ ≈ 0,5772     (Bailey, LdP 2014)
- DSR = PSR(SR*)
- MinTRL = 1 + [1 − γ₃ SR + (γ₄ − 1)/4 SR²] (z₁₋α / (SR − SR*))²          (Bailey, LdP 2012)
- Sharpe annualisé de Lo (2002) : SR · q / √(q + 2 Σ_{k=1}^{q−1} (q − k) ρ_k)
- t-stat de la moyenne à erreurs Newey–West (noyau de Bartlett)
- test binomial unilatéral H₀ : p = p* ; Benjamini–Hochberg (fausses découvertes)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from qlab.core.errors import DataError

EULER_GAMMA = 0.5772156649015329
Floats = npt.NDArray[np.float64]


def _returns(r: npt.ArrayLike, min_obs: int = 2) -> Floats:
    a = np.asarray(r, dtype=np.float64)
    if a.ndim != 1 or a.size < min_obs:
        raise DataError(f"rendements : série 1D d'au moins {min_obs} observations attendue")
    if not np.all(np.isfinite(a)):
        raise DataError("rendements : valeurs non finies (trous à exclure avant)")
    return a


@dataclass(frozen=True, slots=True)
class Moments:
    """Moments d'une série : ``sharpe`` par période, ``skew`` γ₃, ``kurtosis`` γ₄ (Pearson)."""

    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float


def moments(returns: npt.ArrayLike) -> Moments:
    r = _returns(returns)
    std = float(r.std(ddof=1))
    if std == 0:
        raise DataError("rendements constants : Sharpe indéfini")
    centred = r - r.mean()
    m2 = float(np.mean(centred**2))
    return Moments(
        r.size,
        float(r.mean()),
        std,
        float(r.mean()) / std,
        float(np.mean(centred**3)) / m2**1.5,
        float(np.mean(centred**4)) / m2**2,
    )


def annualize(sharpe: float, periods_per_year: int) -> float:
    if periods_per_year <= 0:
        raise DataError("periods_per_year doit être > 0")
    return sharpe * math.sqrt(periods_per_year)


def _sr_variance_factor(sharpe: float, skew: float, kurtosis: float) -> float:
    factor = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if factor <= 0:
        raise DataError("moments incohérents : facteur de variance du Sharpe ≤ 0")
    return factor


def psr(sharpe: float, sr_star: float, n: int, skew: float, kurtosis: float) -> float:
    """Probabilistic Sharpe Ratio : probabilité que le vrai Sharpe dépasse ``sr_star``."""
    if n < 2:
        raise DataError("PSR : au moins 2 observations")
    z = (
        (sharpe - sr_star)
        * math.sqrt(n - 1)
        / math.sqrt(_sr_variance_factor(sharpe, skew, kurtosis))
    )
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """SR* : meilleur Sharpe attendu **par hasard** parmi ``n_trials`` essais sans talent."""
    if n_trials < 1:
        raise DataError("n_trials doit être ≥ 1")
    if sharpe_variance < 0:
        raise DataError("variance des Sharpe négative")
    if n_trials == 1:
        return 0.0  # un seul essai : pas de sélection, le seuil est 0
    q = stats.norm.ppf
    return math.sqrt(sharpe_variance) * float(
        (1 - EULER_GAMMA) * q(1 - 1 / n_trials) + EULER_GAMMA * q(1 - 1 / (n_trials * math.e))
    )


def dsr(
    sharpe: float,
    n: int,
    skew: float,
    kurtosis: float,
    *,
    n_trials: int,
    sharpe_variance: float,
) -> float:
    """Deflated Sharpe Ratio = PSR(SR*) ; > 0,95 exigé par les critères d'acceptation."""
    return psr(sharpe, expected_max_sharpe(n_trials, sharpe_variance), n, skew, kurtosis)


def min_track_record(
    sharpe: float, sr_star: float, skew: float, kurtosis: float, alpha: float
) -> float:
    """MinTRL (observations) pour conclure SR > SR* au niveau ``1 − alpha`` ; ∞ si SR ≤ SR*."""
    if not 0 < alpha < 1:
        raise DataError("alpha doit être dans ]0, 1[")
    if sharpe <= sr_star:
        return math.inf
    z = float(stats.norm.ppf(1 - alpha))
    return 1 + _sr_variance_factor(sharpe, skew, kurtosis) * (z / (sharpe - sr_star)) ** 2


def min_detectable_sharpe(
    n: int,
    sr_star: float,
    skew: float,
    kurtosis: float,
    *,
    alpha: float,
    power: float,
) -> float:
    """Plus petit Sharpe (par période) détectable avec ``n`` observations : le test PSR au
    niveau ``1 − alpha`` le reconnaît avec la probabilité ``power``. Sinon, « données
    insuffisantes » plutôt que « pas de signal »."""
    if not (0 < alpha < 1 and 0 < power < 1) or n < 2:
        raise DataError("alpha, power dans ]0, 1[ et n ≥ 2 attendus")
    need = float(stats.norm.ppf(1 - alpha) + stats.norm.ppf(power))

    def gap(sr: float) -> float:
        return (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(
            _sr_variance_factor(sr, skew, kurtosis)
        ) - need

    upper = sr_star + 1.0
    while gap(upper) < 0:
        upper += 1.0
        if upper > sr_star + 100:
            raise DataError("Sharpe minimal détectable introuvable (moments extrêmes)")
    return float(optimize.brentq(gap, sr_star, upper))


def lo_annualized_sharpe(returns: npt.ArrayLike, periods_per_year: int, max_lag: int) -> float:
    """Sharpe annualisé corrigé de l'autocorrélation (Lo 2002), autocorrélations jusqu'à
    ``max_lag`` (au-delà supposées nulles) ; sans autocorrélation : SR · √q."""
    r = _returns(returns, min_obs=max_lag + 2)
    q = periods_per_year
    m = moments(r)
    centred = r - r.mean()
    denom = float(np.sum(centred**2))
    rho = [
        float(np.sum(centred[k:] * centred[:-k]) / denom) for k in range(1, min(max_lag, q - 1) + 1)
    ]
    inside = q + 2 * sum((q - k) * rho_k for k, rho_k in enumerate(rho, start=1))
    if inside <= 0:
        raise DataError("Lo 2002 : autocorrélations incohérentes (terme sous la racine ≤ 0)")
    return m.sharpe * q / math.sqrt(inside)


def newey_west_lags(n: int) -> int:
    """Règle de Newey et West (1994) : ⌊4 (n / 100)^(2/9)⌋."""
    return int(4 * (n / 100) ** (2 / 9))


def newey_west_tstat(returns: npt.ArrayLike, lags: int) -> float:
    """t-stat de la moyenne des rendements, variance HAC de Newey–West (noyau de Bartlett)."""
    r = _returns(returns, min_obs=lags + 2)
    e = r - r.mean()
    n = r.size
    s = float(np.dot(e, e)) / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * float(np.dot(e[k:], e[:-k])) / n
    if s <= 0:
        raise DataError("Newey–West : variance estimée ≤ 0")
    return float(r.mean()) / math.sqrt(s / n)


def binomial_pvalue(successes: int, trials: int, p0: float) -> float:
    """p-valeur unilatérale de H₀ : p = p0 contre p > p0 (taux de réussite, §7)."""
    if not 0 <= successes <= trials or trials < 1 or not 0 < p0 < 1:
        raise DataError("test binomial : 0 ≤ succès ≤ essais, essais ≥ 1, p0 ∈ ]0, 1[")
    return float(stats.binomtest(successes, trials, p0, alternative="greater").pvalue)


def benjamini_hochberg(pvalues: npt.ArrayLike, q: float) -> npt.NDArray[np.bool_]:
    """Rejets de H₀ contrôlant le taux de fausses découvertes à ``q`` (Benjamini–Hochberg)."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)) or not 0 < q < 1:
        raise DataError("p-valeurs dans [0, 1] et q dans ]0, 1[ attendus")
    m = p.size
    order = np.argsort(p)
    below = p[order] <= q * np.arange(1, m + 1) / m
    rejected = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.nonzero(below)[0]))
        rejected[order[: k + 1]] = True
    return rejected
