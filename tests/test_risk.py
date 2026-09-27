"""Tests de qlab.live.risk : décisions calculées à la main, verrou, drapeaux externes, cibles."""

from __future__ import annotations

from decimal import Decimal

import pytest

from qlab.core.config import LtRiskConfig
from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.live import risk as rk

LIMITS = LtRiskConfig(
    max_open_positions=3, max_day_loss_frac=0.25, max_drawdown_frac=0.55, max_bar_age_hours=30
)
NOW = date_to_ms("2026-09-27") + 1 * MS_PER_HOUR  # 01:00, une heure après la clôture 1d
CLOSE = date_to_ms("2026-09-27") - 1  # clôture de la barre du 26
D = Decimal


def _state(*equity: float, last: int | None = CLOSE) -> rk.State:
    return rk.State([D(str(v)) for v in equity], last, NOW)


def test_normal_day_trades() -> None:
    """−10 % sur la journée, −20 % depuis le plus haut : dans l'enveloppe ⇒ on trade."""
    assert rk.check(LIMITS, _state(100, 88.9, 80)) == rk.Decision("trade", ())


def test_stale_or_missing_bar_suspends() -> None:
    """Barre close depuis 31 h > 30 h, ou absente : pas de décision ce jour (positions gardées)."""
    late = rk.State([D(50)], CLOSE - 30 * MS_PER_HOUR, NOW)
    assert rk.check(LIMITS, late).action == "suspend"
    assert rk.check(LIMITS, _state(50, last=None)).reasons == ("barre du jour absente ou périmée",)
    assert rk.check(LIMITS, rk.State([D(50)], NOW - 30 * MS_PER_HOUR, NOW)).action == "trade"


def test_day_loss_beyond_envelope_flattens() -> None:
    """50 → 37 € : −26 % en un jour > 25 % ⇒ couper ; 50 → 38 € (−24 %) : on trade."""
    cut = rk.check(LIMITS, _state(50, 37))
    assert cut.action == "flatten" and "perte du jour -26.0%" in cut.reasons[0]
    assert rk.check(LIMITS, _state(50, 38)).action == "trade"


def test_drawdown_beyond_envelope_flattens() -> None:
    """Plus haut 100, puis 44 : −56 % > 55 % ⇒ couper (baisse lente, aucun jour extrême)."""
    path = [100, 90, 80, 70, 60, 52, 44]
    cut = rk.check(LIMITS, _state(*path))
    assert cut.action == "flatten" and "drawdown -56.0%" in cut.reasons[0]
    assert rk.check(LIMITS, _state(*path[:-1])).action == "trade"  # 52 : −48 %


def test_most_severe_wins_and_all_reasons_are_kept() -> None:
    """Barre absente (suspendre) et drawdown −56 % (couper, veille à 50 : −12 % sur le jour,
    dans l'enveloppe) : on coupe, les deux raisons restent."""
    d = rk.check(LIMITS, _state(100, 50, 44, last=None))
    assert d.action == "flatten" and len(d.reasons) == 2


def test_halt_is_latched() -> None:
    """Après un arrêt, tout redevient normal mais l'arrêt tient jusqu'à levée manuelle."""
    d = rk.check(LIMITS, _state(100, 101), halted=True)
    assert d.action == "flatten" and "à lever à la main" in d.reasons[0]


def test_external_flags() -> None:
    """Drapeau actif : appliqué ; drapeau expiré : ignoré ; niveau invalide : refusé."""
    hack = rk.RiskFlag("events", "flatten", "piratage d'une plateforme", NOW + MS_PER_HOUR)
    fed = rk.RiskFlag("events", "suspend", "annonce de la Fed", NOW + MS_PER_HOUR)
    old = rk.RiskFlag("events", "flatten", "ancien", NOW - 1)
    assert rk.check(LIMITS, _state(50), [fed]).action == "suspend"
    assert rk.check(LIMITS, _state(50), [fed, hack]).action == "flatten"
    assert rk.check(LIMITS, _state(50), [old]).action == "trade"
    with pytest.raises(DataError, match="niveau"):
        rk.RiskFlag("events", "trade", "x", NOW)


def test_targets_and_orders() -> None:
    rk.check_target(LIMITS, {"BTCEUR": 1 / 3, "ETHEUR": 1 / 3, "SOLEUR": 1 / 3})
    with pytest.raises(DataError, match="4 positions"):
        rk.check_target(LIMITS, {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    with pytest.raises(DataError, match="somme ≤ 1"):
        rk.check_target(LIMITS, {"A": 0.7, "B": 0.7})
    assert rk.order_allowed(D(50), D(50), 1.0) and not rk.order_allowed(D("25.01"), D(50), 0.5)


def test_inconsistent_equity_rejected() -> None:
    with pytest.raises(DataError, match="≤ 0"):
        rk.check(LIMITS, _state(50, 0))
