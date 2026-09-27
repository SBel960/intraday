"""Garde-fous du paper trading puis du réel : décision « trader », « suspendre » ou « couper ».

Fonctions pures : l'appelant (``live/paper.py``) fournit l'état et garde le verrou.

- **suspendre** (aucun ordre ce jour, positions gardées) : barre du jour absente ou périmée au
  moment de la décision (LT.7 : on ne décide jamais sur une donnée figée), ou drapeau externe
  de niveau « suspendre » ;
- **couper** (mise à plat puis arrêt **verrouillé**, levé seulement à la main) : perte d'un
  jour au-delà de ``max_day_loss_frac``, drawdown depuis le plus haut au-delà de
  ``max_drawdown_frac``, ou drapeau externe de niveau « couper ». Ces seuils sont l'enveloppe
  de ce que la stratégie a vécu en backtest : les dépasser, c'est se comporter autrement que
  ce qui a été testé ;
- cibles et ordres : au plus ``max_open_positions`` lignes, ordre au plus ``max_order_frac`` ×
  valeur du portefeuille (``base.risk``).

**Drapeaux externes** (phase 9, événements : piratage d'une plateforme, décision de la Fed…) :
acceptés dès maintenant, pour que les brancher ne touche pas ce fichier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from qlab.core.config import LtRiskConfig
from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_HOUR

Action = Literal["trade", "suspend", "flatten"]
_SEVERITY: dict[Action, int] = {"trade": 0, "suspend": 1, "flatten": 2}


@dataclass(frozen=True, slots=True)
class RiskFlag:
    """Drapeau externe : ``level`` « suspend » ou « flatten », actif jusqu'à ``until_ms``."""

    source: str
    level: Action
    reason: str
    until_ms: int

    def __post_init__(self) -> None:
        if self.level not in ("suspend", "flatten"):
            raise DataError(f"drapeau {self.source} : niveau suspend ou flatten attendu")


@dataclass(frozen=True, slots=True)
class State:
    """État au moment de la décision : valeurs du portefeuille à chaque clôture (ancienne →
    récente), clôture de la dernière barre connue, heure courante (horloge contrôlée)."""

    equity: Sequence[Decimal]
    last_bar_close_ms: int | None
    now_ms: int


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reasons: tuple[str, ...]


def check(
    limits: LtRiskConfig, state: State, flags: Sequence[RiskFlag] = (), *, halted: bool = False
) -> Decision:
    """Décision du jour ; ``halted`` : un arrêt précédent reste verrouillé (couper)."""
    if halted:
        return Decision("flatten", ("arrêt verrouillé : à lever à la main après analyse",))
    found: list[tuple[Action, str]] = []
    age = None if state.last_bar_close_ms is None else state.now_ms - state.last_bar_close_ms
    if age is None or age > limits.max_bar_age_hours * MS_PER_HOUR:
        found.append(("suspend", "barre du jour absente ou périmée"))
    found += _equity_limits(limits, state.equity)
    found += [(f.level, f"{f.source} : {f.reason}") for f in flags if f.until_ms > state.now_ms]
    if not found:
        return Decision("trade", ())
    worst = max((a for a, _ in found), key=_SEVERITY.__getitem__)
    return Decision(worst, tuple(r for _, r in found))


def _equity_limits(limits: LtRiskConfig, equity: Sequence[Decimal]) -> list[tuple[Action, str]]:
    if any(v <= 0 for v in equity):
        raise DataError("valeur du portefeuille ≤ 0 : état incohérent")
    out: list[tuple[Action, str]] = []
    if len(equity) >= 2:
        day = float(equity[-1] / equity[-2] - 1)
        if day < -limits.max_day_loss_frac:
            out.append(
                ("flatten", f"perte du jour {day:.1%} (limite −{limits.max_day_loss_frac:.0%})")
            )
    if equity:
        drawdown = float(equity[-1] / max(equity) - 1)
        if drawdown < -limits.max_drawdown_frac:
            out.append(
                ("flatten", f"drawdown {drawdown:.1%} (limite −{limits.max_drawdown_frac:.0%})")
            )
    return out


def check_target(limits: LtRiskConfig, target: Mapping[str, float]) -> None:
    """Refuse une cible à plus de ``max_open_positions`` lignes ou hors [0, 1]."""
    held = [a for a, w in target.items() if w > 0]
    if len(held) > limits.max_open_positions:
        raise DataError(f"{len(held)} positions voulues > {limits.max_open_positions} autorisées")
    if any(w < 0 for w in target.values()) or sum(target.values()) > 1 + 1e-9:
        raise DataError("cible : poids ≥ 0 de somme ≤ 1 attendus (spot, sans levier)")


def order_allowed(notional: Decimal, equity: Decimal, max_order_frac: float) -> bool:
    """Un ordre ne dépasse pas ``max_order_frac`` × valeur du portefeuille."""
    return notional <= equity * Decimal(repr(max_order_frac))
