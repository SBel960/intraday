"""Tests de qlab.longterm.lt_evaluate : univers par fiche, moitiés, référence, fenêtre, verdict."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import lt_backtest, strategies
from qlab.longterm import lt_evaluate as ev
from qlab.longterm import signals as sg
from qlab.longterm.signals import DATE
from qlab.research.hypothesis import load_hypothesis

D = MS_PER_DAY
T0 = date_to_ms("2021-01-01")
DAYS = 800
ROOT = Path(__file__).resolve().parent.parent / "hypotheses"
TRADE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
QUOTED = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
XS_EUR = load_hypothesis(ROOT / "lt_xs_momentum_eur.yaml")
TS = load_hypothesis(ROOT / "lt_ts_momentum.yaml")


def _panel(symbols: tuple[str, ...], seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {s: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, DAYS))) for s in symbols}
    return pl.DataFrame({DATE: [T0 + i * D for i in range(DAYS)], **data})


def _setup(config_dir: Path) -> ev.Setup:
    config = load_config(config_dir)
    paths = DataPaths(config_dir.parent / "data")
    listed = [binance_symbol(s, "USDT") for s in (*TRADE, *QUOTED)]
    SnapshotStore(paths, "binance").save(exchange_info(listed), FALLBACK_FEES, T0, "https://x")
    snapshot = SnapshotStore(paths, "binance").latest()
    assert snapshot is not None
    closes, quoted = _panel(TRADE, 0), _panel(QUOTED, 1)
    # FFF n'est membre qu'à partir du jour 400 (paire devenue liquide)
    members = quoted.select(
        DATE,
        *(pl.lit(value=True).alias(s) for s in QUOTED[:-1]),
        (pl.int_range(pl.len()) >= 400).alias(QUOTED[-1]),
    )
    flat = pl.DataFrame({DATE: closes[DATE], "funding_1d": [0.0001] * DAYS})
    market = strategies.Market(
        closes,
        closes,
        flat,
        {},
        365,
        config.longterm.signals,
        {s: s.removesuffix("USDT") for s in TRADE},
        strategies.Panel(quoted, quoted, members),
    )
    rules = lt_backtest.pair_rules(config, snapshot, {}, [*TRADE, *QUOTED])
    return ev.Setup(config, snapshot, market, closes, rules, config.base.capital_tiers[1], {})


def test_panel_follows_the_fiche_universe(config_dir: Path) -> None:
    setup = _setup(config_dir)
    assert ev.panel(setup, TS).closes.columns == [DATE, *TRADE]
    assert ev.panel(setup, XS_EUR).closes.columns == [DATE, *QUOTED]
    bare = dataclasses.replace(setup, market=dataclasses.replace(setup.market, quoted=None))
    with pytest.raises(DataError, match="trade_eur non chargé"):
        ev.panel(bare, XS_EUR)


def test_trade_eur_stability_uses_two_disjoint_halves(config_dir: Path) -> None:
    """Univers trade_eur : critère « actifs » sur deux moitiés disjointes (AAA, CCC, EEE) et
    (BBB, DDD, FFF), chacune face à son propre panier de membres."""
    setup = _setup(config_dir)
    groups = ev._groups(setup, XS_EUR)
    halves = [g.quoted.closes.columns[1:] for _, g in groups if g.quoted is not None]
    assert halves == [["AAAUSDT", "CCCUSDT", "EEEUSDT"], ["BBBUSDT", "DDDUSDT", "FFFUSDT"]]
    params = {"lookback_days": 30.0, "top_k": 3.0}
    result = ev.per_asset(setup, XS_EUR, params, strategies.policy_for(XS_EUR))
    assert set(result) == {"moitié 1", "moitié 2"}
    assert all(s.size == b.size > 300 for s, b in result.values())


def test_benchmark_of_a_changing_universe_holds_the_day_members(config_dir: Path) -> None:
    """Avant le jour 400, FFF n'est pas membre : la référence ne le détient pas."""
    setup = _setup(config_dir)
    universe = ev.panel(setup, XS_EUR)
    weights = sg.equal_weight(universe.closes, universe.members)
    assert weights["FFFUSDT"][:400].sum() == 0 and weights["FFFUSDT"][500] == pytest.approx(1 / 6)
    assert weights["AAAUSDT"][100] == pytest.approx(1 / 5)
    assert ev.benchmark(setup, XS_EUR, universe).size == DAYS - 1


def test_judge_writes_a_complete_verdict(config_dir: Path) -> None:
    setup = _setup(config_dir)
    params = {"lookback_days": 30.0, "top_k": 3.0}
    policy = strategies.policy_for(XS_EUR)
    weights = strategies.strategy_for(XS_EUR).build(setup.market, params)
    universe = ev.panel(setup, XS_EUR)
    r = ev.returns(setup, weights, policy, universe.closes, universe.opens)
    r = r[ev.first_decision(weights) :]
    text = ev.judge(setup, (XS_EUR, params, policy, weights), r, n_trials=1, variance=0.0)
    assert text.startswith("## lt_xs_momentum_eur · lookback_days=30, top_k=3")
    assert "**Verdict :" in text and "| Buy & hold |" in text and "(chauffe exclue)" in text
    assert "spread supposé 0.10%" in text


def test_never_invested_trial_counts_with_zero_sharpe() -> None:
    """Rendements tous nuls (jamais investi) : Sharpe 0 par convention, l'essai compte dans N."""
    r = ev.trial_result(np.zeros(10), 365)
    assert (r.sharpe, r.n_obs, r.skew, r.kurtosis) == (0.0, 10, 0.0, 3.0)
    assert ev.trial_result(np.array([0.01, -0.02, 0.03]), 365).sharpe != 0


def test_first_decision_skips_the_warm_up() -> None:
    """Cash les jours 0 à 2 (signal pas encore défini), investi dès le jour 3 ; jamais
    investi ⇒ toute la période (évalué à plat, Sharpe 0 au registre)."""
    w = pl.DataFrame({DATE: [T0 + i * D for i in range(5)], "A": [0.0, 0.0, 0.0, 0.5, 0.0]})
    assert ev.first_decision(w) == 3
    assert ev.first_decision(w.with_columns(A=pl.lit(0.0))) == 0
