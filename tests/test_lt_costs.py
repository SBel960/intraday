"""Tests de qlab.longterm.lt_costs : coûts d'un palier, rejeu calculé à la main, verdict."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import lt_costs as lc
from qlab.longterm.allocation import Policy
from qlab.longterm.signals import DATE
from qlab.research.hypothesis import load_hypothesis

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
FICHES = Path(__file__).resolve().parent.parent / "hypotheses"
DAILY = Policy("calendar", period_days=1)
CHEAP = {"A": lc.AssetCost(0.01, 0.01, True), "B": lc.AssetCost(0.01, 0.01, True)}


def _grid(**cols: list[float]) -> pl.DataFrame:
    n = len(next(iter(cols.values())))
    return pl.DataFrame({DATE: [T0 + i * D for i in range(n)], **cols})


def test_tier_costs(config_dir: Path) -> None:
    """Frais taker 0,10 % + ½ spread + slippage 0,05 % + conversion 0,20 % (config de test) ;
    minNotional 5 au palier de 50 ⇒ δ_min 10 %, au palier de 200 ⇒ 2,5 %."""
    config = load_config(config_dir)
    paths = DataPaths(config_dir.parent / "data")
    symbols = [binance_symbol(s, "USDT") for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
    SnapshotStore(paths, "binance").save(exchange_info(symbols), FALLBACK_FEES, T0, "https://x")
    snapshot = SnapshotStore(paths, "binance").latest()
    assert snapshot is not None
    t0, t1 = config.base.capital_tiers
    costs = lc.tier_costs(config, snapshot, t0, {"BTCUSDT": 0.0002})
    assert costs["BTCUSDT"] == lc.AssetCost(pytest.approx(0.0036), pytest.approx(0.1), True)  # type: ignore[arg-type]
    assert costs["ETHUSDT"].cost_frac == pytest.approx(0.001 + 0.0005 + 0.0005 + 0.002)
    assert not costs["ETHUSDT"].spread_measured
    assert lc.tier_costs(config, snapshot, t1, {})["SOLUSDT"].delta_min == pytest.approx(0.025)


def test_simulate_by_hand() -> None:
    """Prix constants, 5 jours = 1 « année » (periods_per_year 5) : achat de A (1) au jour 0,
    passage de A à B au jour 4 (vente 1 + achat 1) ⇒ TO 3/an, drag 3 × 1 % = 3 %/an."""
    weights = _grid(A=[1.0, 1, 1, 1, 0], B=[0.0, 0, 0, 0, 1])
    closes = _grid(A=[10.0] * 5, B=[20.0] * 5)
    res = lc.simulate(weights, closes, DAILY, CHEAP, periods_per_year=5)
    assert (res.years, res.orders, res.rejected) == (1.0, 3, 0)
    assert res.turnover_annual == pytest.approx(3.0)
    assert res.drag_annual == pytest.approx(0.03)
    assert res.mean_cost_frac == pytest.approx(0.01)


def test_drift_creates_turnover_only_when_rebalanced() -> None:
    """Cible A 50 % ; A double au jour 1 ⇒ poids 1 / 1,5 = 66,7 % ; rééquilibrage quotidien :
    vente de 16,7 % ; bande de 20 % : rien."""
    weights = _grid(A=[0.5, 0.5])
    closes = _grid(A=[10.0, 20.0])
    daily = lc.simulate(weights, closes, DAILY, CHEAP, periods_per_year=2)
    assert daily.turnover_annual == pytest.approx(0.5 + 1 / 6)
    banded = lc.simulate(weights, closes, Policy("bands", band=0.2), CHEAP, periods_per_year=2)
    assert banded.turnover_annual == pytest.approx(0.5)


def test_minnotional_rejections_are_counted() -> None:
    """Cible 50 % puis 55 % : écart de 5 % sous δ_min = 10 % ⇒ rejet compté, pas de coût."""
    costs = {"A": lc.AssetCost(0.01, 0.1, False)}
    res = lc.simulate(_grid(A=[0.5, 0.55]), _grid(A=[1.0, 1.0]), DAILY, costs, periods_per_year=2)
    assert (res.orders, res.rejected, res.rejected_share) == (1, 1, 0.5)


def test_missing_price_keeps_value() -> None:
    """Jour sans prix (trou) : rendement nul ce jour-là, pas d'erreur."""
    res = lc.simulate(
        _grid(A=[1.0, 1.0, 1.0]),
        _grid(A=[1.0, float("nan"), 1.0]),
        DAILY,
        CHEAP,
        periods_per_year=3,
    )
    assert res.turnover_annual == pytest.approx(1.0)


def test_gate_against_frozen_fiches() -> None:
    """``lt_ts_momentum`` : edge 3 %/an ; drag 1,5 %/an = 50 % ⇒ testée (seuil 50 %).
    ``lt_short_reversal`` : edge 0,4 %/trade ; aller-retour 2 × 0,4 % = 200 % ⇒ non testée."""
    per_year = load_hypothesis(FICHES / "lt_ts_momentum.yaml")
    per_trade = load_hypothesis(FICHES / "lt_short_reversal.yaml")
    res = lc.CostResult(1.0, 5.0, 0.015, 10, 0, 0.004)
    ok = lc.gate(res, per_year, 0.5)
    assert ok.passed and ok.ratio == pytest.approx(0.5)
    bad = lc.gate(res, per_trade, 0.5)
    assert not bad.passed and bad.ratio == pytest.approx(2.0)
    assert "aller-retour 0.80%" in bad.detail


def test_render(config_dir: Path) -> None:
    cfg = load_config(config_dir).longterm.costs
    res = lc.CostResult(1.0, 5.0, 0.015, 10, 2, 0.004)
    row = lc.Row("lt_ts_momentum · lookback_days=90", "t0", res, lc.Verdict(True, 0.5, ""))
    text = lc.render([row], cfg, spread_measured=False)
    assert (
        "| lt_ts_momentum · lookback_days=90 | t0 | 5.0 | 1.50% | 10 | 2 (17%) | 50% | testée |"
        in text
    )
    assert "spread supposé 0.10%" in text


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: lc.simulate(_grid(A=[1.0]), _grid(A=[1.0]), DAILY, CHEAP, 1), "2 dates"),
        (lambda: lc.simulate(_grid(A=[1.0, 1]), _grid(B=[1.0, 1]), DAILY, CHEAP, 1), "grille"),
        (lambda: lc.simulate(_grid(Z=[1.0, 1]), _grid(Z=[1.0, 1]), DAILY, CHEAP, 1), "inconnus"),
        (lambda: lc.simulate(_grid(A=[1.0, 1]), _grid(A=[1.0, 1]), DAILY, CHEAP, 0), "periods"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
