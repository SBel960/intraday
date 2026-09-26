"""Tests de qlab.longterm.strategies : table complète, politiques, poids valides, gate complet."""

from __future__ import annotations

import dataclasses
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.longterm import strategies as st
from qlab.longterm.signals import DATE
from qlab.research.hypothesis import Hypothesis, load_all

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
FICHES = list(load_all(Path(__file__).resolve().parent.parent / "hypotheses"))
TRADE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")  # symbols.trade de la config de test


def _fiche(fiche_id: str) -> Hypothesis:
    return next(h for h in FICHES if h.id == fiche_id)


def test_every_longterm_fiche_has_a_strategy() -> None:
    ids = {h.id for h in FICHES if h.volet == "longterm"}
    assert ids == set(st.REGISTRY)


def test_policies_follow_fiche_horizons() -> None:
    """Semaine ⇒ 7 j ; mois (30 j) ⇒ 30 j ; fin de mois : le signal fixe ses dates ⇒ 1 j."""
    assert st.policy_for(_fiche("lt_ts_momentum")).period_days == 7
    assert st.policy_for(_fiche("lt_low_volatility")).period_days == 30
    assert st.policy_for(_fiche("lt_turn_of_month")).period_days == 1
    odd = dataclasses.replace(_fiche("lt_ts_momentum"), horizon_s=90_000)
    with pytest.raises(DataError, match="nombre entier de jours"):
        st.policy_for(odd)
    with pytest.raises(DataError, match="aucune stratégie"):
        st.policy_for(dataclasses.replace(_fiche("lt_ts_momentum"), id="lt_inconnue"))


def test_multi_asset_fiches_are_flagged() -> None:
    flagged = {k for k, s in st.REGISTRY.items() if s.multi_asset}
    assert flagged == {
        "lt_xs_momentum",
        "lt_low_volatility",
        "lt_btc_alt_rotation",
        "lt_xs_momentum_eur",
    }


def test_rotation_needs_a_btc_pair() -> None:
    market = _market()
    no_btc = dataclasses.replace(market, bases=dict.fromkeys(TRADE, "ALT"))
    with pytest.raises(DataError, match="aucune paire tradée sur BTC"):
        st.strategy_for(_fiche("lt_btc_alt_rotation")).build(no_btc, {"lookback_days": 30.0})


def test_trial_name_and_integer_days() -> None:
    assert st.trial_name(_fiche("lt_xs_momentum"), {"lookback_days": 30.0, "top_k": 1.0}) == (
        "lt_xs_momentum · lookback_days=30, top_k=1"
    )
    with pytest.raises(DataError, match="entier"):
        st._days({"lookback_days": 1.5}, "lookback_days")


def _market(days: int = 420) -> st.Market:
    rng = np.random.default_rng(0)
    dates = [T0 + i * D for i in range(days)]
    closes = pl.DataFrame(
        {DATE: dates, **{s: np.exp(np.cumsum(rng.normal(0, 0.03, days))) for s in TRADE}}
    )
    fund = pl.DataFrame({DATE: dates, "funding_1d": rng.normal(0.0003, 0.0002, days)})
    breadth = pl.DataFrame({DATE: dates, "breadth": rng.uniform(0, 1, days)})
    cfg = load_config(Path(__file__).resolve().parent.parent / "config").longterm.signals
    volumes = closes.with_columns(pl.exclude(DATE).abs() * 1e6)
    bases = {s: s.removesuffix("USDT") for s in TRADE}
    members = closes.select(DATE, *(pl.col(s).is_not_null() for s in TRADE))
    quoted = st.Panel(closes, closes, members)  # univers trade_eur : les mêmes paires ici
    return st.Market(closes, volumes, fund, {50: breadth, 100: breadth}, 365, cfg, bases, quoted)


def test_every_trial_builds_valid_weights() -> None:
    """Les 27 essais (vagues 1 et 2) : poids ≥ 0, somme ≤ 1, grille des prix de leur univers."""
    market, n = _market(), 0
    for h in FICHES:
        for params in h.grid():
            w = st.strategy_for(h).build(market, params)
            assert w[DATE].to_list() == market.closes[DATE].to_list()
            x = w.drop(DATE).to_numpy()
            assert (x >= 0).all() and (x.sum(axis=1) <= 1 + 1e-12).all(), st.trial_name(h, params)
            n += 1
    assert n == 27


def _write_klines(paths: DataPaths, symbol: str, closes: np.ndarray, volume: float) -> None:
    out = paths.lt_klines("1d", symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "open_time_ms": [T0 + i * D for i in range(closes.size)],
            "close": closes,
            "volume_quote": [volume] * closes.size,
        }
    ).write_parquet(out)


def _write_funding(paths: DataPaths, symbol: str, days: int) -> None:
    key = f"futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-2024.zip"
    path = paths.raw_archive("binance_vision", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{T0 + k * 8 * 3_600_000 + 1},8,0.0001" for k in range(3 * days))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("f.csv", "calc_time,funding_interval_hours,last_funding_rate\n" + rows + "\n")
