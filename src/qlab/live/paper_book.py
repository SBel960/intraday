"""Une journée de paper trading, sans réseau ni disque : valoriser, contrôler, décider, exécuter.

``live/paper.py`` fournit les données du jour et écrit l'enregistrement renvoyé dans le
journal ; ce module fait le calcul, **à l'identique de ``lt_backtest.run``** (un test rejoue
un backtest jour par jour et compare fills et valeurs, au centime près) :

1. valeur à la clôture de la barre ``t`` (cash + quantités × clôtures) ;
2. garde-fous (``live/risk.py``) sur l'historique des valeurs de clôture : trader, suspendre
   (aucun ordre, positions gardées) ou couper (tout vendre ; l'arrêt reste verrouillé) ;
3. décision (``allocation.decide``, même ancre et même δ_min que le backtest) ;
4. ordres dimensionnés comme au backtest (vente : tout si la position se solde, sinon Δw × V
   au prix d'exécution ; achat : Δw × V dans la limite du cash, frais compris), exécutés par
   ``PaperBroker`` à l'**ouverture de ``t + 1``** (prix de référence fourni).

État : ``replay`` rebâtit cash, quantités, historique des valeurs et verrou depuis le journal
(enregistrements ``start``, ``day``, ``resume``) ; ``day`` renvoie l'enregistrement du jour.
Les montants sont écrits en texte (``Decimal`` exact), jamais en flottant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from qlab.core.config import LtRiskConfig
from qlab.core.errors import DataError
from qlab.core.records import Record
from qlab.exchange.lot import Side
from qlab.live import risk
from qlab.live.broker import Execution, Order, PaperBroker
from qlab.longterm import allocation
from qlab.longterm.lt_backtest import PairRules

ZERO = Decimal(0)
_CLOSED = 1e-9  # même seuil que lt_backtest : poids restant sous lequel une vente solde


@dataclass(frozen=True, slots=True)
class Market:
    """Données du jour : barre ``t`` close (``bar_ms`` : son ouverture), ouverture de ``t+1``."""

    bar_ms: int
    bar_close_ms: int
    closes: Mapping[str, float]
    next_opens: Mapping[str, float]
    target: Mapping[str, float]  # poids voulus par la stratégie à la clôture de t
    now_ms: int


@dataclass(frozen=True, slots=True)
class Plan:
    """Règles fixes du portefeuille."""

    policy: allocation.Policy
    anchor_ms: int  # première date de la grille du backtest (calendrier identique)
    rules: Mapping[str, PairRules]
    limits: LtRiskConfig
    flags: Sequence[risk.RiskFlag] = ()


@dataclass(slots=True)
class Book:
    """État rebâti depuis le journal."""

    quote: str
    cash: Decimal
    held: dict[str, Decimal]
    equity: list[Decimal] = field(default_factory=list)  # valeurs aux clôtures passées
    halted: bool = False
    last_bar_ms: int | None = None


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def replay(records: Sequence[Record]) -> Book:
    """État après tous les enregistrements (``start`` en premier, puis ``day`` / ``resume``)."""
    if not records or records[0].get("kind") != "start":
        raise DataError("journal sans enregistrement « start » en tête : lancer « paper start »")
    first = records[0]
    book = Book(first["quote"], Decimal(first["capital"]), {})
    for r in records[1:]:
        if r.get("kind") == "resume":
            book.halted = False
        elif r.get("kind") == "day":
            book.cash = Decimal(r["after"]["cash"])
            book.held = {s: Decimal(q) for s, q in r["after"]["held"].items()}
            book.equity.append(Decimal(r["value_close"]))
            book.halted = bool(r["halted"])
            book.last_bar_ms = int(r["bar_ms"])
        else:
            raise DataError(f"enregistrement inconnu dans le journal : {r.get('kind')!r}")
    return book


def _value(book: Book, closes: Mapping[str, float]) -> Decimal:
    missing = sorted(s for s, q in book.held.items() if q and s not in closes)
    if missing:
        raise DataError(f"clôture absente pour des positions détenues : {missing}")
    return book.cash + sum((q * _dec(closes[s]) for s, q in book.held.items() if q), ZERO)


def _quantity(
    trade: allocation.Trade,
    book: Book,
    *,
    value: Decimal,
    close: float,
    ref: Decimal,
    rules: PairRules,
) -> tuple[Side, Decimal]:
    """Sens et quantité, comme ``lt_backtest._execute`` : vente de tout si la position se
    solde, sinon Δw × V au prix d'exécution ; achat de Δw × V dans la limite du cash (frais
    compris)."""
    held = book.held.get(trade.asset, ZERO)
    wanted = abs(_dec(trade.delta_w)) * value
    if trade.delta_w < 0:
        weight = float(held * _dec(close) / value)
        closing = weight + trade.delta_w < _CLOSED
        return "SELL", held if closing else min(wanted / (ref * (1 - rules.impact_frac)), held)
    price = ref * (1 + rules.impact_frac)
    return "BUY", min(wanted, book.cash / (1 + rules.fee_frac)) / price


def day(book: Book, market: Market, plan: Plan) -> tuple[Record, list[Execution]]:
    """Journée ``market.bar_ms`` : enregistrement du journal et exécutions (``book`` mis à jour)."""
    if book.last_bar_ms is not None and market.bar_ms <= book.last_bar_ms:
        raise DataError("barre déjà traitée : une seule décision par barre (LT.7)")
    value = _value(book, market.closes)
    state = risk.State([*book.equity, value], market.bar_close_ms, market.now_ms)
    verdict = risk.check(plan.limits, state, plan.flags, halted=book.halted)
    current = {s: float(q * _dec(market.closes[s]) / value) for s, q in book.held.items() if q}
    delta_min = {s: float(r.filters.min_notional / value) for s, r in plan.rules.items()}
    if verdict.action == "trade":
        risk.check_target(plan.limits, market.target)
        decision = allocation.decide(
            plan.policy,
            date_ms=market.bar_ms,
            anchor_ms=plan.anchor_ms,
            current=current,
            target=market.target,
            delta_min=delta_min,
        )
    elif verdict.action == "flatten":
        decision = allocation.rebalance(current, {}, band=0.0, delta_min=delta_min)
    else:
        decision = allocation.NO_TRADE
    executions = _execute(book, decision, value, market, plan)
    record = _record(
        book, market, value=value, verdict=verdict, decision=decision, executions=executions
    )
    book.equity.append(value)
    book.halted = verdict.action == "flatten"
    book.last_bar_ms = market.bar_ms
    record["halted"] = book.halted
    return record, executions


def _execute(
    book: Book, decision: allocation.Decision, value: Decimal, market: Market, plan: Plan
) -> list[Execution]:
    broker = PaperBroker(book.quote, book.cash, plan.rules, book.held)
    out = []
    for trade in decision.trades:
        ref = market.next_opens.get(trade.asset)
        if ref is None:
            raise DataError(f"{trade.asset} : ouverture de la barre suivante absente")
        close, price = market.closes[trade.asset], _dec(ref)
        side, qty = _quantity(
            trade, book, value=value, close=close, ref=price, rules=plan.rules[trade.asset]
        )
        ex = broker.submit(Order(trade.asset, side, qty, price), ts_ms=market.now_ms)
        out.append(ex)
        balances = broker.balances()
        book.cash = balances[book.quote]
        book.held = {s: q for s, q in balances.items() if s != book.quote}
    return out


def _record(
    book: Book,
    market: Market,
    *,
    value: Decimal,
    verdict: risk.Decision,
    decision: allocation.Decision,
    executions: Sequence[Execution],
) -> Record:
    trades = [{"asset": t.asset, "delta_w": t.delta_w} for t in decision.trades]
    return {
        "kind": "day",
        "bar_ms": market.bar_ms,
        "run_ms": market.now_ms,
        "closes": dict(market.closes),
        "next_opens": dict(market.next_opens),
        "target": dict(market.target),
        "value_close": str(value),
        "risk": {"action": verdict.action, "reasons": list(verdict.reasons)},
        "trades": trades,
        "skipped": [{"asset": t.asset, "delta_w": t.delta_w} for t in decision.skipped],
        "executions": [_execution(e) for e in executions],
        "after": {"cash": str(book.cash), "held": {s: str(q) for s, q in book.held.items()}},
    }


def _execution(e: Execution) -> dict[str, Any]:
    o = e.order
    return {
        "symbol": o.symbol,
        "side": o.side,
        "qty_wanted": str(o.qty),
        "ref_price": str(o.ref_price),
        "accepted": e.accepted,
        "qty": str(e.qty),
        "price": str(e.price),
        "notional": str(e.notional),
        "fee": str(e.fee),
        "reason": e.reason,
    }
