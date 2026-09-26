"""Tests de qlab.costs.gate_run : journées Tardis construites à la main, frais du snapshot."""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.jsonlog import read_log
from qlab.core.paths import DataPaths
from qlab.costs import gate_run
from qlab.data import tardis
from qlab.exchange.snapshots import SnapshotStore

T = 1_704_067_200_000  # 2024-01-01T00:00:00Z
DAY_MS = 86_400_000


def _tardis_day(symbol: str, start_ms: int, growth: float, seconds: int = 30) -> bytes:
    """Journée Tardis : une cotation par seconde, mid × ``growth``/s, spread relatif 0,0002."""
    rows = [tardis.HEADER]
    for k in range(seconds + 1):
        mid = 100.0 * growth**k
        ts_us = (start_ms + 1000 * k) * 1000
        rows.append(f"binance,{symbol},{ts_us},{ts_us},1,{mid * 1.0001},{mid * 0.9999},1")
    return gzip.compress(("\n".join(rows) + "\n").encode())


def _setup(config_dir: Path, fees: dict[str, object]) -> DataPaths:
    paths = DataPaths(config_dir.parent / "data")
    SnapshotStore(paths, "binance").save(
        exchange_info([binance_symbol(s, "USDT") for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]),
        fees,
        T,
        "https://x",
    )
    for i, day in enumerate(("2024-01-01", "2024-02-01")):
        dest = paths.raw_archive("tardis", tardis.file_key("BTCUSDT", day))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_tardis_day("BTCUSDT", T + 31 * DAY_MS * i, 1.001))
    return paths


def test_end_to_end_with_real_fees(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Frais réels 0,095 % : c = 2 × 0,00095 + 0,0002 = 0,0021.
    À 5 s : 5 ln 1,001 / 0,0021 = 2,38 < 3 (26 instants × 2 journées = 52 échantillons).
    À 30 s : un seul instant par journée (t = 0), 30 ln 1,001 / 0,0021 = 14,3 > 3 ⇒ 30 s."""
    fees = {
        **FALLBACK_FEES,
        "origin": "account",
        "by_symbol": {"BTCUSDT": {"effective_maker_frac": 0.001, "effective_taker_frac": 0.00095}},
    }
    paths = _setup(config_dir, fees)
    code = gate_run.main(["--config", str(config_dir), "--symbols", "BTCUSDT"])
    assert code == 0
    assert "BTCUSDT : frais taker 0.0950% (account)" in capsys.readouterr().out
    report = next(paths.reports.glob("cost_gate_v1_*.md")).read_text()
    assert "- Frais taker : 0.0950% (frais réels du compte)" in report
    assert "- Journées : 2 (1er du mois, 2024-01-01 → 2024-02-01)" in report
    assert f"| 5 s | 52 | 0.2100% | 0.2100% | {5 * math.log(1.001):.4%} | 2.38 |" in report
    assert f"| 30 s | 2 | 0.2100% | 0.2100% | {30 * math.log(1.001):.4%} | 14.28 |" in report
    assert "**Verdict : Plus petit horizon rentable (ratio > 3) : 30 s.**" in report
    kinds = [
        r["kind"] for f in (paths.logs / "gate_run").glob("*.jsonl") for r in read_log(f).records
    ]
    assert "gate.v1" in kinds


def test_fallback_fees_are_flagged(config_dir: Path) -> None:
    paths = _setup(config_dir, FALLBACK_FEES)
    assert gate_run.main(["--config", str(config_dir), "--symbols", "BTCUSDT"]) == 0
    report = next(paths.reports.glob("cost_gate_v1_*.md")).read_text()
    assert "barème de REPLI" in report


def test_missing_data(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate_run.main(["--config", str(config_dir), "--symbols", "BTCUSDT"]) == 1
    assert "aucun snapshot" in capsys.readouterr().err
    _setup(config_dir, FALLBACK_FEES)
    assert gate_run.main(["--config", str(config_dir), "--symbols", "ETHUSDT"]) == 1
    assert "ETHUSDT : aucune journée Tardis" in capsys.readouterr().err
