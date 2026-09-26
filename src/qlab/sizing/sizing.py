"""Taille des positions : volatilité cible (plafond de risque, sans levier) et Kelly (information).

**Volatilité cible** (SPEC_INTRADAY §6, SPEC_LONG_TERME LT.3). À chaque date ``t``, la
volatilité du portefeuille voulu est estimée par σ̂_p = √(w' Σ w · q), où Σ est la covariance
des rendements des ``lookback`` dernières périodes **jusqu'à t compris** (connues à la clôture)
et q le nombre de périodes par an (calendrier du marché). Les poids sont multipliés par

    k_t = min(1, σ_cible / σ̂_p, w_max / Σ w)

- k ≤ 1 : la cible **réduit** le risque, elle n'agrandit jamais une position au-delà de ce que
  la stratégie voulait (pas de levier, spot uniquement) ;
- Σ w · k ≤ w_max ≤ 1 : exposition totale plafonnée ;
- fenêtre incomplète (début de série, trou) ⇒ k = 0 : pas d'estimation, pas de risque.

**Kelly** : f* = μ / σ² (continu) ou p − (1 − p) / b (discret), à titre d'information : jamais
appliqué plein, plafonné à ``fraction`` × f* (0,25) et à la limite de risque de la config.

Tableaux numpy (dates × actifs) : ce module ne dépend d'aucun format de données du volet.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from qlab.core.errors import DataError

Floats = npt.NDArray[np.float64]


def portfolio_vol(weights: Floats, window: Floats, periods_per_year: int) -> float:
    """σ annualisée de ``weights`` d'après les rendements ``window`` (périodes × actifs)."""
    if window.shape[0] < 2 or window.shape[1] != weights.size:
        raise DataError("fenêtre : au moins 2 périodes et un rendement par actif attendus")
    held = weights > 0
    if not held.any():
        return 0.0
    cov = np.atleast_2d(np.cov(window[:, held], rowvar=False, ddof=1))
    w = weights[held]
    return math.sqrt(max(float(w @ cov @ w), 0.0) * periods_per_year)


def vol_target(
    weights: npt.ArrayLike,
    returns: npt.ArrayLike,
    *,
    target_vol_annual: float,
    w_max: float,
    lookback: int,
    periods_per_year: int,
) -> Floats:
    """Poids réduits pour que la volatilité estimée ne dépasse pas la cible (voir l'en-tête).

    ``returns[t]`` = rendement de la période qui se termine en ``t`` (NaN = absent).
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if w.ndim != 2 or w.shape != r.shape:
        raise DataError("poids et rendements : tableaux dates × actifs de même forme attendus")
    if target_vol_annual <= 0 or not 0 < w_max <= 1 or lookback < 2 or periods_per_year < 1:
        raise DataError("cible > 0, w_max dans ]0, 1], lookback ≥ 2, periods_per_year ≥ 1")
    if (w < 0).any() or (w.sum(axis=1) > 1 + 1e-12).any():
        raise DataError("poids ≥ 0 de somme ≤ 1 attendus (spot, sans levier)")
    out = np.zeros_like(w)
    for t in range(lookback - 1, w.shape[0]):
        total = w[t].sum()
        if total <= 0:
            continue
        window = r[t - lookback + 1 : t + 1]
        held = w[t] > 0
        if not np.isfinite(window[:, held]).all():
            continue  # fenêtre incomplète pour un actif détenu : pas d'estimation
        sigma = portfolio_vol(w[t], np.nan_to_num(window), periods_per_year)
        cap = target_vol_annual / sigma if sigma > 0 else 1.0
        out[t] = w[t] * min(1.0, cap, w_max / total)
    return out


def kelly_continuous(mean: float, variance: float) -> float:
    """f* = μ / σ² (rendements par période)."""
    if variance <= 0:
        raise DataError("variance > 0 attendue")
    return mean / variance


def kelly_discrete(win_prob: float, payoff_ratio: float) -> float:
    """f* = p − (1 − p) / b (``b`` : gain moyen / perte moyenne)."""
    if not 0 < win_prob < 1 or payoff_ratio <= 0:
        raise DataError("p dans ]0, 1[ et b > 0 attendus")
    return win_prob - (1 - win_prob) / payoff_ratio


def capped_kelly(f_star: float, *, fraction: float, limit: float) -> float:
    """Fraction de Kelly bornée : 0 ≤ fraction · f* ≤ ``limit`` (jamais négatif : pas de short)."""
    if not 0 < fraction <= 1 or not 0 <= limit <= 1:
        raise DataError("fraction dans ]0, 1] et limit dans [0, 1] attendus")
    return min(max(fraction * f_star, 0.0), limit)
