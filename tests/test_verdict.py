"""Tests de qlab.timing.verdict : moitiés de l'univers, blocs, verdicts, hold-out unique."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from fakes import timing_fiche, timing_market

from qlab.core.config import load_config
from qlab.core.paths import DataPaths
from qlab.core.records import read_records
from qlab.research.hypothesis import load_hypothesis
from qlab.research.trials import TrialRegistry, TrialResult
from qlab.timing import scan, verdict


def test_halves_are_disjoint_and_cover_everything() -> None:
    panel = pl.DataFrame({"symbol": ["A", "B", "C", "D", "E"], "t_ms": [0] * 5})
    parts = verdict.halves(panel)
    first, second = (set(p["symbol"]) for p in parts.values())
    assert first == {"A", "C", "E"} and second == {"B", "D"}


def _setup(config_dir: Path, tmp_path: Path) -> tuple[DataPaths, list[str]]:
    paths = DataPaths(config_dir.parent / "data")
    timing_market(paths)
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    timing_fiche(fiches, "tm_momentum", [1, 4])
    common = ["--config", str(config_dir), "--fiches", str(fiches)]
    assert scan.main([*common, "run"]) == 0
    return paths, common


def test_best_writes_full_verdicts(config_dir: Path, tmp_path: Path) -> None:
    paths, common = _setup(config_dir, tmp_path)
    assert verdict.main([*common, "best", "--top", "1"]) == 0
    blob = json.loads((paths.reports / "timing_verdict_latest.json").read_text())
    assert len(blob["verdicts"]) == 1
    assert blob["verdicts"][0]["id"] == "tm_momentum"
    assert "dsr" in blob["verdicts"][0]
    text = next(paths.reports.glob("timing_verdict_2*.md")).read_text()
    assert "DSR" in text and "moitié 1" in text
    assert blob["oracle_sharpe"] > 5  # le tricheur gagne nettement : pas de décalage caché


def test_best_needs_a_scan(config_dir: Path, tmp_path: Path) -> None:
    timing_market(DataPaths(config_dir.parent / "data"))
    assert verdict.main(["--config", str(config_dir), "--fiches", str(tmp_path), "best"]) == 1


def test_holdout_runs_once_and_only_declared_combinations(config_dir: Path, tmp_path: Path) -> None:
    paths, common = _setup(config_dir, tmp_path)
    hold = [*common, "holdout", "--id", "tm_momentum"]
    assert verdict.main([*hold, "--params", "lookback_h=2,horizon_h=8"]) == 1  # non déclarée
    assert verdict.main([*hold, "--params", "lookback_h=2,horizon_h=4"]) == 0
    records = read_records(paths.timing_holdout)
    assert len(records) == 1 and records[0]["periods"] > 0
    assert verdict.main([*hold, "--params", "lookback_h=2,horizon_h=1"]) == 1  # déjà utilisé
    assert len(read_records(paths.timing_holdout)) == 1


def test_criteria_block_in_periods(config_dir: Path) -> None:
    """mean_block_days = 10 (config de test) ⇒ 240 périodes d'1 h, 10 de 24 h, 1 au minimum."""
    config = load_config(config_dir)
    assert verdict.timing_criteria(config, 24, 1).mean_block == pytest.approx(240)
    assert verdict.timing_criteria(config, 24, 24).mean_block == pytest.approx(10)
    assert verdict.timing_criteria(config, 24, 1000).mean_block == pytest.approx(1)


def test_sharpe_variance_on_a_common_time_scale(tmp_path: Path) -> None:
    """Sharpe 0,1 par période à 400 périodes/an = 0,2 à 100 périodes/an (même Sharpe annuel 2) :
    ramenés à 100, les essais valent 0,2 ; 0,2 ; 0 ⇒ variance (ddof 1) = 0,04/3."""
    hyp = load_hypothesis(timing_fiche(tmp_path, "tm_momentum", [1, 4, 8]))
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    for h, sharpe, ppy in ((1, 0.1, 400), (4, 0.2, 100), (8, 0.0, 100)):
        result = TrialResult(sharpe, 50, 0.0, 3.0, ppy)
        registry.record(hyp, {"lookback_h": 2, "horizon_h": h}, result, ts_ms=1)
    assert verdict.sharpe_variance(registry, "timing", 100) == pytest.approx(0.04 / 3)
