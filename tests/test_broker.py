"""Tests de qlab.live.broker : exécution simulée calculée à la main, identique au backtest,
refus sans effet, et réel impossible quoi qu'on fournisse."""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

import polars as pl
import pytest
from fakes import binance_filters

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.lot import SymbolFilters
from qlab.live import broker as br
from qlab.longterm import lt_backtest as bt
from qlab.longterm.allocation import Policy
from qlab.longterm.signals import DATE

FEE, IMPACT = Decimal("0.001"), Decimal("0.001")
RULES = {
    "BTCEUR": bt.PairRules(SymbolFilters.from_binance("BTCEUR", binance_filters()), FEE, IMPACT)
}
STEP = Decimal("0.00001")
TS = date_to_ms("2026-10-01")
# Droits d'une clé de trading spot correcte (réponse apiRestrictions)
GOOD_KEY = {"enableReading": True, "enableSpotAndMarginTrading": True, "ipRestrict": True}


def _floor(x: Decimal) -> Decimal:
    return x.quantize(STEP, rounding=ROUND_FLOOR)


def _paper(cash: str = "50", btc: str = "0") -> br.PaperBroker:
    return br.PaperBroker("EUR", Decimal(cash), RULES, {"BTCEUR": Decimal(btc)})


def test_buy_by_hand() -> None:
    """50 € ; achat de 0,00049 BTC à 100 000 × 1,001 = 100 100 € : 49,049 € + 0,049049 € frais."""
    p = _paper()
    ex = p.submit(br.Order("BTCEUR", "BUY", Decimal("0.00049"), Decimal(100_000)), ts_ms=TS)
    assert ex.accepted and ex.reason is None
    assert (ex.qty, ex.price, ex.notional, ex.fee) == (
        Decimal("0.00049"),
        Decimal("100100.000"),
        Decimal("49.04900000000"),
        Decimal("0.04904900000000000"),
    )
    assert p.balances() == {"EUR": Decimal("0.90195100000000000"), "BTCEUR": Decimal("0.00049")}


def test_sell_by_hand_and_rounding_to_step() -> None:
    """Vente de 0,000499 → arrondie à 0,00049 au pas ; prix 100 000 × 0,999 = 99 900 €."""
    p = _paper(cash="0", btc="0.0005")
    ex = p.submit(br.Order("BTCEUR", "SELL", Decimal("0.000499"), Decimal(100_000)), ts_ms=TS)
    assert (ex.qty, ex.price) == (Decimal("0.00049"), Decimal("99900.000"))
    assert p.balances()["BTCEUR"] == Decimal("0.00001")
    assert p.balances()["EUR"] == ex.notional - ex.fee == Decimal("48.95100") * (1 - FEE)


def test_refusals_change_nothing() -> None:
    """Sous le minimum de 5 € ; plus que le cash ; plus que détenu : refusés, soldes intacts."""
    p = _paper(cash="50", btc="0.0001")
    before = p.balances()
    small = p.submit(br.Order("BTCEUR", "BUY", Decimal("0.00004"), Decimal(100_000)), ts_ms=TS)
    rich = p.submit(br.Order("BTCEUR", "BUY", Decimal("0.0005"), Decimal(100_000)), ts_ms=TS)
    short = p.submit(br.Order("BTCEUR", "SELL", Decimal("0.0002"), Decimal(100_000)), ts_ms=TS)
    assert [e.reason for e in (small, rich, short)] == [
        "notional_below_min",
        "insufficient_balance",
        "insufficient_balance",
    ]
    assert not any(e.accepted for e in (small, rich, short))
    assert all(e.qty == e.fee == 0 for e in (small, rich, short))
    assert p.balances() == before


def test_invalid_orders_and_books() -> None:
    p = _paper()
    with pytest.raises(DataError, match="aucune règle"):
        p.submit(br.Order("ETHEUR", "BUY", Decimal(1), Decimal(1)), ts_ms=TS)
    with pytest.raises(DataError, match="ordre invalide"):
        p.submit(br.Order("BTCEUR", "BUY", Decimal(0), Decimal(1)), ts_ms=TS)
    with pytest.raises(DataError, match="solde négatif"):
        br.PaperBroker("EUR", Decimal(-1), RULES)
    with pytest.raises(DataError, match="sans règles"):
        br.PaperBroker("EUR", Decimal(1), RULES, {"ETHEUR": Decimal(1)})


def test_paper_executes_exactly_like_the_backtest() -> None:
    """Même achat puis même vente que ``lt_backtest.run`` (quantités qu'il a choisies) :
    mêmes prix, quantités, notionnels, frais, et même cash final, au centime près."""
    grid = [date_to_ms("2024-01-01") + i * MS_PER_DAY for i in range(4)]

    def col(values: list[float]) -> pl.DataFrame:
        return pl.DataFrame({DATE: grid, "BTCEUR": values})

    opens = [100_000.0, 100_000.0, 110_000.0, 120_000.0]
    res = bt.run(
        col([1.0, 1.0, 0.0, 0.0]),
        col(opens),
        col([100_000.0, 110_000.0, 115_000.0, 120_000.0]),
        Policy("calendar", period_days=1),
        RULES,
        initial_quote=Decimal(1000),
    )
    p = br.PaperBroker("EUR", Decimal(1000), RULES)
    for fill in res.fills:
        ref = Decimal(repr(opens[grid.index(fill.date_ms)]))
        ex = p.submit(br.Order("BTCEUR", fill.side, fill.qty, ref), ts_ms=fill.date_ms)  # type: ignore[arg-type]
        assert (ex.qty, ex.price, ex.notional, ex.fee) == (
            fill.qty,
            fill.price,
            fill.notional,
            fill.fee,
        )
    assert [f.side for f in res.fills] == ["BUY", "SELL"]
    assert p.balances() == {"EUR": res.equity[-1], "BTCEUR": Decimal(0)}


def test_live_blockers() -> None:
    assert br.live_blockers(True, GOOD_KEY) == []
    assert br.live_blockers(False, None) == [
        "réglage explicite du réel absent (allow_live)",
        "droits de la clé non vérifiés",
    ]
    risky = {
        **GOOD_KEY,
        "enableWithdrawals": True,
        "enableInternalTransfer": True,
        "ipRestrict": False,
    }
    assert br.live_blockers(True, risky) == [
        "clé avec des droits interdits : enableWithdrawals, enableInternalTransfer",
        "clé non restreinte à l'adresse IP de ce PC",
    ]
    assert br.live_blockers(True, {"enableReading": True, "ipRestrict": True}) == [
        "clé sans droit de trading spot"
    ]
    assert "enableSpotAndMarginTrading" not in br.LIVE_FORBIDDEN
    assert "enableWithdrawals" in br.LIVE_FORBIDDEN


def test_live_is_impossible_even_with_flag_and_good_key() -> None:
    args = {"quote": "EUR", "cash": Decimal(50), "rules": RULES}
    assert isinstance(br.open_broker("paper", **args), br.PaperBroker)  # type: ignore[arg-type]
    with pytest.raises(br.LiveDisabledError, match=r"réglage explicite.*non vérifiés.*non écrite"):
        br.open_broker("live", **args)  # type: ignore[arg-type]
    with pytest.raises(
        br.LiveDisabledError, match=r"^réel impossible : exécution réelle non écrite"
    ):
        br.open_broker("live", allow_live=True, rights=GOOD_KEY, **args)  # type: ignore[arg-type]
    with pytest.raises(DataError, match="mode inconnu"):
        br.open_broker("demo", **args)  # type: ignore[arg-type]
