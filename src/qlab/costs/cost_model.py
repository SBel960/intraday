"""Modèle de coûts (SPEC_INTRADAY §5, SPEC_LONG_TERME LT.4) : formules pures, vectorisées.

Toutes les grandeurs sont des **fractions sans unité** du notionnel (0.001 = 0,1 %), en
``float64`` : ce sont des statistiques sur des millions d'observations (distributions de
coûts), pas des montants à exécuter. Les montants exécutables restent en ``Decimal``
(``exchange/lot.py``, ``core/money.py``). Aucune valeur n'est codée en dur : frais, spreads et
glissement viennent des paramètres effectifs, des données mesurées ou de la config.

Formules :

- spread relatif          s̃ = (a − b) / m,  m = (a + b) / 2
- aller-retour taker      c_taker = 2 f_taker + s̃ + 2 slip
- aller-retour maker      c_maker = 2 f_maker + sélection adverse mesurée
- un ordre (long terme)   c = f_taker + s̃ / 2 + slip
- taux de réussite min.   p* = L / (W + L)   (gain et perte moyens nets, > 0)
- frottement annuel       drag = N_trades/an × c
- coût d'un rééquilibrage C = V Σ |Δw_i| c_i   (V : valeur du portefeuille, devise)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qlab.core.errors import DataError

Floats = npt.NDArray[np.float64]
ArrayLike = npt.ArrayLike  # scalaire, liste, tableau numpy ou série Polars


def _arr(name: str, x: ArrayLike, *, positive: bool = False) -> Floats:
    """Tableau ``float64`` fini (≥ 0, ou > 0 si ``positive``) ; sinon ``DataError``."""
    a = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise DataError(f"{name} : valeurs non finies (NaN / inf)")
    if np.any(a <= 0) if positive else np.any(a < 0):
        raise DataError(f"{name} doit être {'> 0' if positive else '≥ 0'}")
    return a


def relative_spread(bid: ArrayLike, ask: ArrayLike) -> Floats:
    """Spread relatif s̃ = (a − b) / m. Carnet croisé (a < b) ou prix ≤ 0 : ``DataError``
    (les exclure avant, voir ``valid_quotes``)."""
    b, a = _arr("bid", bid, positive=True), _arr("ask", ask, positive=True)
    if b.shape != a.shape:
        raise DataError(f"bid et ask de formes différentes : {b.shape} ≠ {a.shape}")
    if np.any(a < b):
        raise DataError("carnet croisé (ask < bid) : exclure ces cotations avant le calcul")
    return (a - b) / ((a + b) / 2)


def valid_quotes(bid: ArrayLike, ask: ArrayLike) -> npt.NDArray[np.bool_]:
    """Masque des cotations utilisables : prix finis, > 0, ask ≥ bid."""
    b, a = np.asarray(bid, dtype=np.float64), np.asarray(ask, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(b) & np.isfinite(a) & (b > 0) & (a >= b)


def round_trip_taker(fee_taker: ArrayLike, rel_spread: ArrayLike, slippage: ArrayLike) -> Floats:
    """c_taker = 2 f_taker + s̃ + 2 slip (entrée et sortie au marché)."""
    return (
        2 * _arr("fee_taker", fee_taker)
        + _arr("rel_spread", rel_spread)
        + 2 * _arr("slippage", slippage)
    )


def round_trip_maker(fee_maker: ArrayLike, adverse_selection: ArrayLike) -> Floats:
    """c_maker = 2 f_maker + sélection adverse mesurée (≥ 0 : coût de se faire exécuter)."""
    return 2 * _arr("fee_maker", fee_maker) + _arr("adverse_selection", adverse_selection)


def one_way_taker(fee_taker: ArrayLike, rel_spread: ArrayLike, slippage: ArrayLike) -> Floats:
    """c = f_taker + s̃ / 2 + slip : un seul ordre au marché (rééquilibrage long terme)."""
    return (
        _arr("fee_taker", fee_taker)
        + _arr("rel_spread", rel_spread) / 2
        + _arr("slippage", slippage)
    )


def breakeven_win_rate(avg_win: ArrayLike, avg_loss: ArrayLike) -> Floats:
    """p* = L / (W + L) : taux de réussite minimal, W et L gain et perte moyens **nets**."""
    w, loss = _arr("avg_win", avg_win, positive=True), _arr("avg_loss", avg_loss, positive=True)
    return loss / (w + loss)


def annual_drag(trades_per_year: ArrayLike, cost: ArrayLike) -> Floats:
    """drag = N_trades/an × c : fraction du capital perdue en coûts chaque année."""
    return _arr("trades_per_year", trades_per_year) * _arr("cost", cost)


def rebalance_cost(portfolio_value: float, delta_weights: ArrayLike, costs: ArrayLike) -> float:
    """C = V Σ |Δw_i| c_i, en devise. ``delta_weights`` peut être négatif (ventes)."""
    v = float(_arr("portfolio_value", portfolio_value))
    dw = np.asarray(delta_weights, dtype=np.float64)
    c = _arr("costs", costs)
    if not np.all(np.isfinite(dw)):
        raise DataError("delta_weights : valeurs non finies")
    if dw.shape != c.shape:
        raise DataError(f"delta_weights et costs de formes différentes : {dw.shape} ≠ {c.shape}")
    return v * float(np.sum(np.abs(dw) * c))
