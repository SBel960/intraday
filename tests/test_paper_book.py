"""Tests de qlab.live.paper_book : identique au backtest jour par jour, garde-fous (couper
verrouillé, suspendre), journal rejoué, erreurs."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import polars as pl
import pytest
from fakes import binance_filters

from qlab.core.config import LtRiskConfig
from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, MS_PER_MIN, date_to_ms
from qlab.exchange.lot import SymbolFilters
from qlab.live import paper_book as pb
from qlab.longterm import lt_backtest as bt
from qlab.longterm.allocation import Policy
from qlab.longterm.signals import DATE

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
FEE, IMPACT = Decimal("0.001"), Decimal("0.0005")
ASSETS = ("AAA", "BBB", "CCC")
RULES = {
    a: bt.PairRules(SymbolFilters.from_binance(a, binance_filters()), FEE, IMPACT) for a in ASSETS
}
LOOSE = LtRiskConfig(
    max_open_positions=3, max_day_loss_frac=0.99, max_drawdown_frac=0.99, max_bar_age_hours=30
)
TIGHT = LtRiskConfig(
    max_open_positions=3, max_day_loss_frac=0.25, max_drawdown_frac=0.55, max_bar_age_hours=30
)
WEEKLY = Policy("calendar", period_days=7)
DAILY = Policy("calendar", period_days=1)


def _market(
    t: int,
    closes: dict[str, float],
    opens: dict[str, float],
    target: dict[str, float],
    *,
    late_h: int = 0,
) -> pb.Market:
    """Barre du jour ``t`` close à minuit ; décision 1 min après (``late_h`` : heures de retard)."""
    bar = T0 + t * D
    return pb.Market(
        bar, bar + D - 1, closes, opens, target, bar + D + MS_PER_MIN + late_h * 3_600_000
    )


def _row(m: np.ndarray, i: int) -> dict[str, float]:
    return {a: float(m[i, j]) for j, a in enumerate(ASSETS)}


def _grid(values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            DATE: [T0 + i * D for i in range(values.shape[0])],
            **{a: values[:, i] for i, a in enumerate(ASSETS)},
        }
    )


@pytest.mark.parametrize("policy", [WEEKLY, DAILY], ids=["hebdo", "quotidien"])
def test_paper_replays_the_backtest_exactly(policy: Policy) -> None:
    """40 jours, prix et cibles aléatoires (0, 1/3, 1/2 ou 1 par actif ; ≥ 10 exécutions) :
    chaque valeur de clôture et chaque exécution du paper sont celles du backtest, au centime
    près."""
    rng = np.random.default_rng(7)
    n = 40
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, (n, 3)), axis=0))
    opens = closes * np.exp(rng.normal(0, 0.01, (n, 3)))
    raw = rng.choice([0.0, 1.0], size=(n, 3))
    weights = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1)
    res = bt.run(
        _grid(weights), _grid(opens), _grid(closes), policy, RULES, initial_quote=Decimal(200)
    )
    book = pb.Book("EUR", Decimal(200), {})
    plan = pb.Plan(policy, T0, RULES, LOOSE)
    done = []
    for t in range(n - 1):
        target = {a: w for a, w in _row(weights, t).items() if w > 0}
        record, executions = pb.day(
            book, _market(t, _row(closes, t), _row(opens, t + 1), target), plan
        )
        assert Decimal(record["value_close"]) == res.equity[t], f"jour {t}"
        assert record["risk"]["action"] == "trade"
        done += [
            (e.order.symbol, e.order.side, e.qty, e.price, e.fee) for e in executions if e.accepted
        ]
    expected = [(f.asset, f.side, f.qty, f.price, f.fee) for f in res.fills]
    assert done == expected and len(done) >= 10


def test_flatten_is_latched_until_resume() -> None:
    """Détenu 1 AAA à 100 ; clôture à 70 (−30 % > 25 %) : tout est vendu à l'ouverture 69 et
    l'arrêt reste verrouillé le lendemain (aucun achat) ; un « resume » le lève."""
    book = pb.Book("EUR", Decimal(0), {"AAA": Decimal(1)}, [Decimal(100)])
    plan = pb.Plan(DAILY, T0, RULES, TIGHT)
    rec, ex = pb.day(book, _market(1, {"AAA": 70.0}, {"AAA": 69.0}, {"AAA": 1.0}), plan)
    assert (
        rec["risk"]["action"] == "flatten" and "perte du jour -30.0%" in rec["risk"]["reasons"][0]
    )
    sold = Decimal("69") * (1 - IMPACT)
    assert [(e.order.side, e.qty, e.price) for e in ex] == [("SELL", Decimal(1), sold)]
    assert book.held["AAA"] == 0 and book.cash == sold * (1 - FEE) and rec["halted"] is True
    rec2, ex2 = pb.day(book, _market(2, {"AAA": 80.0}, {"AAA": 80.0}, {"AAA": 1.0}), plan)
    assert rec2["risk"]["action"] == "flatten" and ex2 == [] and book.halted
    records = [{"kind": "start", "quote": "EUR", "capital": "0"}, {**rec, "kind": "day"}, rec2]
    assert pb.replay([*records, {"kind": "resume"}]).halted is False
    assert pb.replay(records).halted is True


def test_stale_bar_suspends_without_orders() -> None:
    """Décision 31 h après la clôture (> 30 h) : suspendre, positions gardées, rien d'exécuté."""
    book = pb.Book("EUR", Decimal(100), {})
    rec, ex = pb.day(
        book,
        _market(0, {"AAA": 10.0}, {"AAA": 10.0}, {"AAA": 1.0}, late_h=31),
        pb.Plan(DAILY, T0, RULES, TIGHT),
    )
    assert rec["risk"] == {"action": "suspend", "reasons": ["barre du jour absente ou périmée"]}
    assert ex == [] and book.cash == 100 and rec["halted"] is False


def test_journal_replays_to_the_same_book() -> None:
    """start 100 € ; achat de AAA : l'état rejoué depuis les enregistrements = l'état en mémoire."""
    book = pb.Book("EUR", Decimal(100), {})
    rec, ex = pb.day(
        book,
        _market(0, {"AAA": 10.0}, {"AAA": 10.0}, {"AAA": 1.0}),
        pb.Plan(DAILY, T0, RULES, LOOSE),
    )
    assert ex[0].accepted and rec["executions"][0]["qty"] == str(ex[0].qty)
    again = pb.replay([{"kind": "start", "quote": "EUR", "capital": "100"}, rec])
    assert (again.cash, again.held, again.equity, again.last_bar_ms) == (
        book.cash,
        book.held,
        [Decimal(100)],
        T0,
    )


def test_errors() -> None:
    plan = pb.Plan(DAILY, T0, RULES, LOOSE)
    with pytest.raises(DataError, match="start"):
        pb.replay([{"kind": "day"}])
    with pytest.raises(DataError, match="inconnu"):
        pb.replay([{"kind": "start", "quote": "EUR", "capital": "1"}, {"kind": "oops"}])
    book = pb.Book("EUR", Decimal(100), {"AAA": Decimal(1)}, last_bar_ms=T0)
    with pytest.raises(DataError, match="déjà traitée"):
        pb.day(book, _market(0, {"AAA": 10.0}, {}, {}), plan)
    with pytest.raises(DataError, match="clôture absente"):
        pb.day(book, _market(1, {}, {}, {}), plan)
    with pytest.raises(DataError, match="ouverture de la barre suivante absente"):
        pb.day(pb.Book("EUR", Decimal(100), {}), _market(1, {"AAA": 10.0}, {}, {"AAA": 1.0}), plan)
    with pytest.raises(DataError, match="positions voulues"):
        tight = pb.Plan(DAILY, T0, RULES, LtRiskConfig(1, 0.99, 0.99, 30))
        pb.day(
            pb.Book("EUR", Decimal(100), {}), _market(1, {}, {}, {"AAA": 0.5, "BBB": 0.5}), tight
        )
