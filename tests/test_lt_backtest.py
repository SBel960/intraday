"""Tests de qlab.longterm.lt_backtest : exécution calculée à la main, rejets, apports, fuites."""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_filters, binance_symbol, exchange_info

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.lot import SymbolFilters
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import lt_backtest as bt
from qlab.longterm.allocation import Policy
from qlab.longterm.signals import DATE

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
DAILY = Policy("calendar", period_days=1)
FEE, IMPACT = Decimal("0.001"), Decimal("0.001")
RULES = {"A": bt.PairRules(SymbolFilters.from_binance("A", binance_filters()), FEE, IMPACT)}
STEP = Decimal("0.00001")


def _grid(**cols: list[float]) -> pl.DataFrame:
    n = len(next(iter(cols.values())))
    return pl.DataFrame({DATE: [T0 + i * D for i in range(n)], **cols})


def _floor(x: Decimal) -> Decimal:
    return x.quantize(STEP, rounding=ROUND_FLOOR)


def test_buy_then_sell_by_hand() -> None:
    """1 000 € ; décidé à la clôture du jour 0, achat à l'ouverture du jour 1 (100 × 1,001),
    quantité arrondie au pas ; vente totale à l'ouverture du jour 3 (120 × 0,999)."""
    res = bt.run(
        _grid(A=[1.0, 1.0, 0.0, 0.0]),
        _grid(A=[100.0, 100.0, 110.0, 120.0]),
        _grid(A=[100.0, 110.0, 115.0, 120.0]),
        DAILY,
        RULES,
        initial_quote=Decimal(1000),
    )
    buy_price = Decimal("100.1")
    qty = _floor(Decimal(1000) / (1 + FEE) / buy_price)
    cost = qty * buy_price * (1 + FEE)
    cash = Decimal(1000) - cost
    assert res.equity[0] == 1000
    assert res.equity[1] == cash + qty * Decimal(110)
    assert res.equity[2] == cash + qty * Decimal(115)
    sell = qty * Decimal("119.88")
    assert res.equity[3] == cash + sell * (1 - FEE)
    assert [(f.side, f.qty) for f in res.fills] == [("BUY", qty), ("SELL", qty)]
    assert res.fees == qty * buy_price * FEE + sell * FEE


def test_decision_at_close_executes_at_next_open() -> None:
    """Saut de nuit 100 → 200 : un moteur qui exécuterait à la clôture de la veille doublerait
    la mise ; ici l'achat se fait à 200 : la valeur ne peut que baisser (coûts)."""
    res = bt.run(
        _grid(A=[1.0, 1.0]),
        _grid(A=[100.0, 200.0]),
        _grid(A=[100.0, 200.0]),
        DAILY,
        RULES,
        initial_quote=Decimal(1000),
    )
    assert res.fills[0].price == Decimal("200.2")
    assert res.equity[1] < 1000


def test_order_below_min_notional_is_rejected() -> None:
    """10 € et cible 30 % : ordre de 3 € < δ_min = 5 / 10 ⇒ pas tenté (compté chaque jour de
    décision), rien d'exécuté ; le refus de ``lot.py`` n'est donc jamais atteint."""
    res = bt.run(
        _grid(A=[0.3, 0.3, 0.3]),
        _grid(A=[100.0] * 3),
        _grid(A=[100.0] * 3),
        DAILY,
        RULES,
        initial_quote=Decimal(10),
    )
    assert res.fills == [] and res.skipped == 3 and not res.rejected
    assert res.equity == [Decimal(10)] * 3


def test_rounding_to_step_can_trigger_a_lot_rejection() -> None:
    """Pas de quantité de 1 (paire fictive) : 5,5 € au prix 10,01 donnent 0,549 → 0 après
    arrondi ⇒ refus ``qty_zero_after_rounding`` compté par ``lot.py``."""
    coarse = {
        "A": bt.PairRules(SymbolFilters.from_binance("A", binance_filters(step="1")), FEE, IMPACT)
    }
    res = bt.run(
        _grid(A=[1.0, 1.0]),
        _grid(A=[10.0, 10.0]),
        _grid(A=[10.0, 10.0]),
        DAILY,
        coarse,
        initial_quote=Decimal("5.5"),
    )
    assert res.rejected == {"qty_zero_after_rounding": 1} and res.fills == []


def test_missing_open_misses_the_order() -> None:
    res = bt.run(
        _grid(A=[1.0, 0.0]),
        _grid(A=[100.0, float("nan")]),
        _grid(A=[100.0, 100.0]),
        DAILY,
        RULES,
        initial_quote=Decimal(1000),
    )
    assert (res.missed, res.fills) == (1, [])


def test_flows_are_neutral_for_twr() -> None:
    """Apport de 100 € au jour 2, sans position : valeur 100, 100, 200 ; TWR 0, 0."""
    res = bt.run(
        _grid(A=[0.0] * 3),
        _grid(A=[1.0] * 3),
        _grid(A=[1.0] * 3),
        DAILY,
        RULES,
        initial_quote=Decimal(100),
        flows={T0 + 2 * D: Decimal(100)},
    )
    assert res.equity == [100, 100, 200] and res.flows == [0, 0, 100]
    assert res.twr_returns().tolist() == [0.0, 0.0]


def test_peek_is_the_cheater_and_it_wins() -> None:
    """Test anti-fuite (LT.5) : une stratégie qui connaît le rendement d'ouverture à ouverture
    qu'elle va détenir bat largement la même règle décalée dans le passé."""
    rng = np.random.default_rng(0)
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    held_return = np.append(opens[2:] / opens[1:-1] - 1, [0.0, 0.0])  # détenu de t+1 à t+2
    honest = np.append([0.0, 0.0], (held_return > 0).astype(float)[:-2])
    w = _grid(A=list(honest))
    assert bt.peek(w, 2)["A"].to_list()[:-2] == list((held_return > 0).astype(float))[:-2]
    kw = {"initial_quote": Decimal(1000)}
    prices = _grid(A=list(opens))
    fair = bt.run(w, prices, prices, DAILY, RULES, **kw).equity[-1]  # type: ignore[arg-type]
    cheat = bt.run(bt.peek(w, 2), prices, prices, DAILY, RULES, **kw).equity[-1]  # type: ignore[arg-type]
    assert cheat > 5 * fair


def test_pair_rules_from_snapshot(config_dir: Path) -> None:
    """Frais taker du snapshot, impact = ½ spread (mesuré ou supposé) + slippage + conversion."""
    config = load_config(config_dir)
    paths = DataPaths(config_dir.parent / "data")
    symbols = [binance_symbol(s, "USDT") for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
    SnapshotStore(paths, "binance").save(exchange_info(symbols), FALLBACK_FEES, T0, "https://x")
    snapshot = SnapshotStore(paths, "binance").latest()
    assert snapshot is not None
    rules = bt.pair_rules(config, snapshot, {"BTCUSDT": 0.0002})
    assert rules["BTCUSDT"].fee_frac == Decimal("0.001")
    assert rules["BTCUSDT"].impact_frac == Decimal(repr(0.0001 + 0.0005 + 0.002))
    assert rules["ETHUSDT"].impact_frac == Decimal(repr(0.0005 + 0.0005 + 0.002))
    assert rules["SOLUSDT"].filters.min_notional == 5


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (
            lambda: bt.run(
                _grid(A=[1.0]),
                _grid(A=[1.0, 1.0]),
                _grid(A=[1.0]),
                DAILY,
                RULES,
                initial_quote=Decimal(1),
            ),
            "grille",
        ),
        (
            lambda: bt.run(
                _grid(Z=[1.0]),
                _grid(Z=[1.0]),
                _grid(Z=[1.0]),
                DAILY,
                RULES,
                initial_quote=Decimal(1),
            ),
            "inconnues",
        ),
        (lambda: bt.peek(_grid(A=[1.0]), 0), "bars"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
