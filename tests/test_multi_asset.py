"""Tests de qlab.markets.multi_asset : étude multi-actifs, balayage, verdicts, hold-out unique."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from fakes import markets_fiche, markets_series

from qlab.core.cli import Context
from qlab.core.config import load_config
from qlab.core.paths import DataPaths
from qlab.core.records import read_records
from qlab.markets import multi_asset as ma
from qlab.research.trials import TrialRegistry


def _setup(config_dir: Path, tmp_path: Path) -> tuple[DataPaths, list[str]]:
    paths = DataPaths(config_dir.parent / "data")
    markets_series(paths)
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    markets_fiche(fiches, "mk_momentum", [10, 20], [5])
    markets_fiche(fiches, "mk_reversal", [5], [5])
    return paths, ["--config", str(config_dir), "--fiches", str(fiches)]


def test_study_rules(config_dir: Path, tmp_path: Path) -> None:
    paths, _ = _setup(config_dir, tmp_path)
    ctx = Context(load_config(config_dir), paths, None, argparse.Namespace())  # type: ignore[arg-type]
    study = ma.prepare(ctx)
    assert (study.volet, study.unit, study.weighting, study.bars_per_day) == (
        "multi",
        "d",
        "inverse_vol",
        1,
    )
    assert study.costs == {"AAA": 0.0005, "BBB": 0.0005, "CCC": 0.0005, "BTC-USD": 0.0015}
    assert study.periods_per_year(5) == 50  # 250 jours ouvrés / 5
    bench = study.benchmark(5)
    assert bench.height > 0 and bench["bench"].abs().sum() > 0


def test_scan_best_and_single_holdout(config_dir: Path, tmp_path: Path) -> None:
    paths, common = _setup(config_dir, tmp_path)
    assert ma.main([*common, "scan"]) == 0
    assert TrialRegistry(paths.trials).n_trials("multi") == 3
    rows = json.loads((paths.reports / "multi_scan_latest.json").read_text())["rows"]
    assert {r["id"] for r in rows} == {"mk_momentum", "mk_reversal"}
    assert ma.main([*common, "best", "--top", "2"]) == 0
    blob = json.loads((paths.reports / "multi_verdict_latest.json").read_text())
    assert len(blob["verdicts"]) == 2 and blob["oracle_sharpe"] > 1
    hold = [*common, "holdout", "--id", "mk_momentum"]
    assert ma.main([*hold, "--params", "lookback_d=10,horizon_d=5"]) == 0
    assert ma.main([*hold, "--params", "lookback_d=20,horizon_d=5"]) == 1  # déjà utilisé
    assert len(read_records(paths.markets_holdout)) == 1
    assert not paths.timing_holdout.exists()  # hold-out propre au volet


def test_budget_checked_before_running(config_dir: Path, tmp_path: Path) -> None:
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    markets_fiche(fiches, "mk_trend", list(range(1, 22)), [5])  # 21 essais > 20
    assert ma.main(["--config", str(config_dir), "--fiches", str(fiches), "scan"]) == 1
    with pytest.raises(FileNotFoundError):
        (config_dir.parent / "data" / "reports" / "multi_scan_latest.json").read_text()
