"""Tests de qlab.live.paper_report : résumé du journal, intervalle du backtest sur N jours,
verdict, fidélité des prix aux archives, commande."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from qlab.core.config import AcceptanceConfig, load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.records import append_record
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.live import paper_report as pr

D = MS_PER_DAY
T0 = date_to_ms("2026-10-02")
CFG = AcceptanceConfig(0.95, 0.95, 3, 2, True, 500, 10, 0, 60)
TRIAL = "lt_ts_momentum · lookback_days=30"


def _ex(accepted: bool, fee: str) -> dict[str, Any]:
    return {"accepted": accepted, "fee": fee}


def _day(i: int, value: str, action: str = "trade", **kw: Any) -> dict[str, Any]:
    base = {"kind": "day", "bar_ms": T0 + i * D, "value_close": value, "risk": {"action": action}}
    return {"executions": [], "closes": {}, "next_opens": {}, **base, **kw}


RECORDS = [
    {"kind": "start", "capital": "50"},
    _day(0, "50", executions=[_ex(True, "0.0166"), _ex(True, "0.0167")]),
    _day(1, "55"),
    _day(4, "44", "suspend", missed_bars=2, executions=[_ex(False, "0")]),
]


def test_paper_run_by_hand() -> None:
    """50 → 44 : −12 % en 4 jours ; 2 barres manquées ; 2 ordres exécutés, 1 refusé."""
    run = pr.paper_run(RECORDS)
    assert (run.first_bar_ms, run.last_bar_ms, run.days) == (T0, T0 + 4 * D, 4)
    assert run.total_return == pytest.approx(-0.12)
    assert run.decisions == Counter({"trade": 2, "suspend": 1})
    assert (run.missed_bars, run.filled, run.refused, run.fees) == (2, 2, 1, Decimal("0.0333"))
    with pytest.raises(DataError, match="aucune barre"):
        pr.paper_run(RECORDS[:1])


def test_band_of_a_constant_backtest_is_exact() -> None:
    """Backtest à +1 %/jour : tout chemin de 10 jours fait 1,01¹⁰ − 1 ≈ +10,46 %."""
    bt = np.full(300, 0.01)
    band = pr.backtest_band(bt, 10, 0.05, CFG)
    assert band.low == band.high == pytest.approx(1.01**10 - 1)
    assert band.percentile == 0 and pr.backtest_band(bt, 10, 0.2, CFG).percentile == 100
    with pytest.raises(DataError, match="hors de l'historique"):
        pr.backtest_band(bt, 301, 0.0, CFG)


def test_band_of_a_noisy_backtest() -> None:
    """Rendements ±2 % : intervalle autour de 0, déterministe (graine), plus large si N grandit."""
    rng = np.random.default_rng(1)
    bt = rng.normal(0.0, 0.02, 2000)
    short, long = pr.backtest_band(bt, 10, 0.0, CFG), pr.backtest_band(bt, 60, 0.0, CFG)
    assert short.low < 0 < short.high and long.high - long.low > short.high - short.low
    assert 30 < short.percentile < 70
    assert pr.backtest_band(bt, 10, 0.0, CFG) == short


def test_verdict() -> None:
    band = pr.Band(-0.10, 0.20, 50)
    run = pr.paper_run(RECORDS)
    assert pr.verdict(run, band, 60) == "en cours (4 / 60 jours)"
    assert pr.verdict(run, band, 4) == "en dessous de l'intervalle du backtest : rejet"
    assert pr.verdict(run, pr.Band(-0.2, 0.2, 50), 4) == "conforme au backtest"
    assert pr.verdict(run, pr.Band(-0.5, -0.3, 50), 4).startswith("au-dessus")


def test_data_gaps_against_archives() -> None:
    """Clôture du jour 0 identique ; ouverture du jour 1 différente ; jour 4 pas encore archivé."""
    archive = pl.DataFrame(
        {"open_time_ms": [T0, T0 + D], "open": [99.0, 101.0], "close": [100.0, 102.0]}
    )
    records = [
        _day(0, "50", closes={"BTCEUR": 100.0}, next_opens={"BTCEUR": 101.5}),
        _day(4, "50", closes={"BTCEUR": 1.0}, next_opens={"BTCEUR": 1.0}),
    ]
    checked, gaps = pr.data_gaps(records, {"BTCEUR": archive})
    assert checked == 2
    assert gaps == ["BTCEUR ouverture du 2026-10-03 : journal 101.5, archive 101.0"]


def test_render() -> None:
    run = pr.paper_run(RECORDS)
    text = pr.render(TRIAL, run, pr.Band(-0.10, 0.20, 12.0), CFG, (2, ["écart X"]))
    assert text.startswith(f"## {TRIAL}\n\n**Verdict : en cours (4 / 60 jours)**")
    assert "Rendement du paper : -12.00%" in text and "[-10.00% ; +20.00%]" in text
    assert "mieux que 12 %" in text and "suspend 1, trade 2" in text
    assert "2 prix comparés aux archives, 1 écart(s)\n  - écart X" in text


def test_cli(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = DataPaths(load_config(config_dir).base.data.root)
    args = ["--config", str(config_dir), "--trial", TRIAL]
    assert pr.main(args) == 1  # pas de journal
    journal = paths.paper_journal("lt_ts_momentum_lookback_days_30")
    for r in RECORDS:
        append_record(journal, r)
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        out = paths.lt_klines("1d", s)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"open_time_ms": [T0], "open": [1.0], "close": [1.0]}).write_parquet(out)
    monkeypatch.setattr(pr, "_backtest_returns", lambda ctx, trial: np.full(100, 0.001))
    assert pr.main(args) == 0
    report = next(paths.reports.glob("paper_*.md")).read_text()
    assert (
        report.startswith("# Paper trading face au backtest")
        and "en cours (4 / 60 jours)" in report
    )
    assert "Rapport :" in capsys.readouterr().out
