"""Tests de qlab.longterm.allocation : échanges calculés à la main, politiques, invariants."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.longterm import allocation as al

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
LOW = {"A": 0.01, "B": 0.01, "C": 0.01}  # δ_min négligeable


def _trades(decision: al.Decision) -> dict[str, float]:
    return {t.asset: round(t.delta_w, 9) for t in decision.trades}


def test_rebalance_by_hand() -> None:
    """A 50 % → 30 %, B 0 → 60 % : vente de 20 %, cash 50 + 20 = 70 % ≥ 60 % ⇒ tout passe."""
    d = al.rebalance({"A": 0.5}, {"A": 0.3, "B": 0.6}, band=0.0, delta_min=LOW)
    assert _trades(d) == {"A": -0.2, "B": 0.6}
    assert d.skipped == () and d.turnover == pytest.approx(0.8)
    assert al.apply({"A": 0.5}, d) == pytest.approx({"A": 0.3, "B": 0.6})


def test_band_blocks_small_deviations() -> None:
    assert al.rebalance({"A": 0.52}, {"A": 0.5}, band=0.05, delta_min=LOW) == al.NO_TRADE
    assert _trades(al.rebalance({"A": 0.56}, {"A": 0.5}, band=0.05, delta_min=LOW)) == {"A": -0.06}


def test_delta_min_skips_and_counts() -> None:
    """Écart de 4 % avec δ_min = 10 % (5 € de minimum sur 50 €) : pas tenté, compté."""
    d = al.rebalance({"A": 0.5}, {"A": 0.54}, band=0.0, delta_min={"A": 0.1})
    assert d.trades == ()
    assert [(t.asset, round(t.delta_w, 9)) for t in d.skipped] == [("A", 0.04)]


def test_blocked_sell_shrinks_buys_to_available_cash() -> None:
    """A 95 %, cash 5 % ; cible A 90 %, B 10 %. Vente de 5 % sous δ_min(A) = 10 % ⇒ bloquée ;
    cash 5 % pour 10 % d'achat voulus ⇒ achat de B réduit de moitié : 5 %."""
    d = al.rebalance({"A": 0.95}, {"A": 0.9, "B": 0.1}, band=0.0, delta_min={"A": 0.1, "B": 0.01})
    assert _trades(d) == {"B": 0.05}
    assert [t.asset for t in d.skipped] == ["A"]
    assert sum(al.apply({"A": 0.95}, d).values()) == pytest.approx(1.0)


def test_shrunk_buy_below_delta_min_is_skipped_too() -> None:
    d = al.rebalance({"A": 0.95}, {"A": 0.9, "B": 0.1}, band=0.0, delta_min={"A": 0.1, "B": 0.08})
    assert d.trades == () and sorted(t.asset for t in d.skipped) == ["A", "B"]


def test_policies() -> None:
    """Buy & hold : seulement à la première date ; calendaire 7 j : jours 0, 7, 14 ; bandes :
    chaque jour où l'écart dépasse la bande."""
    cur, tgt = {"A": 0.2}, {"A": 0.5}
    kw = {"anchor_ms": T0, "current": cur, "target": tgt, "delta_min": LOW}
    hold, week, band = (
        al.Policy("buy_hold"),
        al.Policy("calendar", period_days=7),
        al.Policy("bands", band=0.1),
    )
    days = range(16)
    assert [d for d in days if al.decide(hold, date_ms=T0 + d * D, **kw).trades] == [0]  # type: ignore[arg-type]
    assert [d for d in days if al.decide(week, date_ms=T0 + d * D, **kw).trades] == [0, 7, 14]  # type: ignore[arg-type]
    assert len([d for d in days if al.decide(band, date_ms=T0 + d * D, **kw).trades]) == 16  # type: ignore[arg-type]
    assert not al.is_calendar_day(T0 - 7 * D, T0, 7)  # avant l'ancre : jamais


def test_drift_by_hand() -> None:
    """A 50 %, B 30 %, cash 20 % ; A +10 %, B −10 % : croissance 1 + 0,05 − 0,03 = 1,02 ⇒
    A = 0,55 / 1,02, B = 0,27 / 1,02."""
    w = al.drift({"A": 0.5, "B": 0.3}, {"A": 0.1, "B": -0.1})
    assert w == pytest.approx({"A": 0.55 / 1.02, "B": 0.27 / 1.02})


def test_random_decisions_keep_weights_valid() -> None:
    """500 tirages : après exécution, poids ≥ 0 et somme ≤ 1 (jamais de découvert)."""
    rng = np.random.default_rng(0)
    assets = ["A", "B", "C"]
    for _ in range(500):
        cur = dict(zip(assets, rng.dirichlet(np.ones(4))[:3], strict=True))
        tgt = dict(zip(assets, rng.dirichlet(np.ones(4))[:3], strict=True))
        dmin = dict(zip(assets, rng.uniform(0, 0.2, 3), strict=True))
        after = al.apply(
            cur, al.rebalance(cur, tgt, band=float(rng.uniform(0, 0.1)), delta_min=dmin)
        )
        assert all(w >= -1e-12 for w in after.values()) and sum(after.values()) <= 1 + 1e-9


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: al.rebalance({"A": 0.7}, {"A": 0.5, "B": 0.6}, band=0, delta_min=LOW), "cibles"),
        (lambda: al.rebalance({"A": -0.1}, {"A": 0.5}, band=0, delta_min=LOW), "actuels"),
        (lambda: al.rebalance({}, {"Z": 0.5}, band=0, delta_min=LOW), "δ_min inconnu"),
        (lambda: al.rebalance({}, {"A": 0.5}, band=-0.1, delta_min=LOW), "bande"),
        (lambda: al.Policy("calendar"), "period_days"),
        (lambda: al.Policy("bands", band=1.5), "bande"),
        (lambda: al.Policy("monthly"), "inconnue"),  # type: ignore[arg-type]
        (lambda: al.drift({"A": 0.5}, {}), "manquant"),
        (lambda: al.drift({"A": 1.0}, {"A": -1.0}), "≤ 0"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
