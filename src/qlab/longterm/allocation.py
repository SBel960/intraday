"""Poids cibles → échanges à faire : buy & hold, rééquilibrage calendaire ou à bandes, δ_min.

Tout est en **fractions de la valeur du portefeuille** V (cash compris) : ``w_i`` = part de V
dans l'actif i, cash = 1 − Σ w_i. Les montants, arrondis au pas et contrôles ``minNotional``
d'un ordre réel restent ceux d'``exchange/lot.py`` (au backtest) ; ici on décide seulement
quoi échanger, en fractions (exception documentée : fractions statistiques en ``float``).

- **δ_min** (LT.4) : un échange |Δw_i| < δ_min_i = minNotional_i / V serait rejeté par
  Binance : il n'est pas tenté, il est compté (``skipped``) pour ``lt_costs.py`` ;
- **bande** : on n'échange l'actif i que si |w_i − w*_i| > bande ;
- ventes d'abord, puis achats avec le cash disponible : si une vente est bloquée par δ_min,
  les achats sont réduits à proportion du cash (jamais de découvert), puis revérifiés ;
- ``drift`` fait évoluer les poids avec les rendements (entre deux décisions).

Politiques : ``buy_hold`` (échange seulement à la première date), ``calendar`` (tous les
``period_days`` jours depuis la première date ; sert aussi aux dates de DCA), ``bands`` (chaque
jour, bande fixe).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY

Kind = Literal["buy_hold", "calendar", "bands"]
_EPS = 1e-12  # tolérance d'arrondi flottant sur des sommes de poids


@dataclass(frozen=True, slots=True)
class Trade:
    """Échange de ``delta_w`` (fraction de V ; > 0 achat, < 0 vente) sur ``asset``."""

    asset: str
    delta_w: float


@dataclass(frozen=True, slots=True)
class Decision:
    trades: tuple[Trade, ...]  # à exécuter
    skipped: tuple[Trade, ...]  # voulus mais sous δ_min (seraient rejetés par Binance)

    @property
    def turnover(self) -> float:
        """Σ |Δw| exécuté (fraction de V)."""
        return sum(abs(t.delta_w) for t in self.trades)


NO_TRADE = Decision((), ())


def _check_weights(name: str, w: Mapping[str, float]) -> None:
    if any(x < -_EPS for x in w.values()) or sum(w.values()) > 1 + _EPS:
        raise DataError(f"{name} : poids ≥ 0 de somme ≤ 1 attendus")


def rebalance(
    current: Mapping[str, float],
    target: Mapping[str, float],
    *,
    band: float,
    delta_min: Mapping[str, float],
) -> Decision:
    """Échanges pour rapprocher ``current`` de ``target`` (bande, δ_min, cash disponible)."""
    _check_weights("poids actuels", current)
    _check_weights("poids cibles", target)
    if band < 0:
        raise DataError("bande ≥ 0 attendue")
    wanted = {}
    for asset in sorted(set(current) | set(target)):
        diff = target.get(asset, 0.0) - current.get(asset, 0.0)
        if abs(diff) > band + _EPS:
            wanted[asset] = diff
    missing = set(wanted) - set(delta_min)
    if missing:
        raise DataError(f"δ_min inconnu pour {sorted(missing)}")
    sells, skipped = _split({a: d for a, d in wanted.items() if d < 0}, delta_min, (), scale=1.0)
    cash = 1 - sum(current.values()) - sum(t.delta_w for t in sells)
    buys_wanted = {a: d for a, d in wanted.items() if d > 0}
    need = sum(buys_wanted.values())
    scale = min(1.0, max(cash, 0.0) / need) if need > 0 else 1.0
    buys, skipped = _split(buys_wanted, delta_min, skipped, scale=scale)
    return Decision((*sells, *buys), skipped)


def _split(
    wanted: Mapping[str, float],
    delta_min: Mapping[str, float],
    skipped: tuple[Trade, ...],
    *,
    scale: float,
) -> tuple[tuple[Trade, ...], tuple[Trade, ...]]:
    """Échanges ≥ δ_min (après mise à l'échelle) à faire ; les autres rejoignent ``skipped``."""
    done: list[Trade] = []
    left = list(skipped)
    for asset, diff in wanted.items():
        trade = Trade(asset, diff * scale)
        (done if abs(trade.delta_w) >= delta_min[asset] else left).append(trade)
    return tuple(done), tuple(left)


@dataclass(frozen=True, slots=True)
class Policy:
    """``kind`` et son réglage : ``period_days`` (calendar) ou ``band`` (bands)."""

    kind: Kind
    period_days: int = 0
    band: float = 0.0

    def __post_init__(self) -> None:
        if self.kind == "calendar" and self.period_days < 1:
            raise DataError("politique calendaire : period_days ≥ 1")
        if self.kind == "bands" and not 0 < self.band < 1:
            raise DataError("politique à bandes : bande dans ]0, 1[")
        if self.kind not in ("buy_hold", "calendar", "bands"):
            raise DataError(f"politique inconnue : {self.kind!r}")


def is_calendar_day(date_ms: int, anchor_ms: int, period_days: int) -> bool:
    """Vrai tous les ``period_days`` jours à partir de ``anchor_ms`` (compris)."""
    elapsed = date_ms - anchor_ms
    return elapsed >= 0 and elapsed % (period_days * MS_PER_DAY) == 0


def decide(
    policy: Policy,
    *,
    date_ms: int,
    anchor_ms: int,
    current: Mapping[str, float],
    target: Mapping[str, float],
    delta_min: Mapping[str, float],
) -> Decision:
    """Décision du jour ``date_ms`` selon la politique (``anchor_ms`` : première date)."""
    if policy.kind == "buy_hold":
        due, band = date_ms == anchor_ms, 0.0
    elif policy.kind == "calendar":
        due, band = is_calendar_day(date_ms, anchor_ms, policy.period_days), 0.0
    else:
        due, band = True, policy.band
    return rebalance(current, target, band=band, delta_min=delta_min) if due else NO_TRADE


def apply(current: Mapping[str, float], decision: Decision) -> dict[str, float]:
    """Poids après exécution (hors coûts, déduits par le backtest)."""
    after = dict(current)
    for t in decision.trades:
        after[t.asset] = after.get(t.asset, 0.0) + t.delta_w
    return {a: w for a, w in after.items() if abs(w) > _EPS}


def drift(weights: Mapping[str, float], returns: Mapping[str, float]) -> dict[str, float]:
    """Poids après une période de rendements simples ``returns`` (cash : rendement 0).

    w'_i = w_i (1 + r_i) / (1 + Σ_j w_j r_j).
    """
    missing = set(weights) - set(returns)
    if missing:
        raise DataError(f"rendement manquant pour {sorted(missing)}")
    growth = 1 + sum(w * returns[a] for a, w in weights.items())
    if growth <= 0:
        raise DataError("valeur du portefeuille ≤ 0 après rendements")
    return {a: w * (1 + returns[a]) / growth for a, w in weights.items()}
