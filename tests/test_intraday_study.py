"""Tests de qlab.trading.intraday_study : étude, balayage, verdicts, hold-out unique."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fakes import trading_fiche, trading_series

from qlab.core.cli import Context
from qlab.core.config import load_config
from qlab.core.paths import DataPaths
from qlab.core.records import read_records
from qlab.research.trials import TrialRegistry
from qlab.trading import intraday_study as tr


def _setup(config_dir: Path, tmp_path: Path) -> tuple[DataPaths, list[str]]:
    paths = DataPaths(config_dir.parent / "data")
    trading_series(paths)
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    trading_fiche(fiches, "tr_momentum", [4, 24], [4])
    trading_fiche(fiches, "tr_reversal", [1], [1])
    return paths, ["--config", str(config_dir), "--fiches", str(fiches)]


def test_study_rules(config_dir: Path, tmp_path: Path) -> None:
    paths, _ = _setup(config_dir, tmp_path)
    ctx = Context(load_config(config_dir), paths, None, argparse.Namespace())  # type: ignore[arg-type]
    study = tr.prepare(ctx)
    assert (study.volet, study.unit, study.weighting, study.bars_per_day) == (
        "trading",
        "h",
        "inverse_vol",
        24,
    )
    assert study.costs["EURUSD"] == 0.00004 and study.costs["SPXUSD"] == 0.00005
    assert study.bars_per_year == 360 * 24
    assert "spread_frac" in study.panel.columns  # ½ spread facturé par le portefeuille


def test_scan_best_and_single_holdout(config_dir: Path, tmp_path: Path) -> None:
    paths, common = _setup(config_dir, tmp_path)
    assert tr.main([*common, "scan"]) == 0
    assert TrialRegistry(paths.trials).n_trials("trading") == 3
    rows = json.loads((paths.reports / "trading_scan_latest.json").read_text())["rows"]
    assert all(r["cost_per_year"] > 0 for r in rows if r["invested"] > 0)
    assert tr.main([*common, "best", "--top", "1"]) == 0
    assert (paths.reports / "trading_verdict_latest.json").exists()
    hold = [*common, "holdout", "--id", "tr_reversal", "--params", "lookback_h=1,horizon_h=1"]
    assert tr.main(hold) == 0
    assert tr.main(hold) == 1  # une seule fois
    assert len(read_records(paths.trading_holdout)) == 1
