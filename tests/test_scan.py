"""Tests de qlab.timing.scan : étude des perpétuels, balayage complet sur un petit marché."""

from __future__ import annotations

import json
from pathlib import Path

from fakes import TIMING_T0, timing_fiche, timing_market

from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.research.trials import TrialRegistry
from qlab.timing import scan

H = MS_PER_HOUR
T0 = TIMING_T0


def test_run_records_trials_and_writes_reports(config_dir: Path, tmp_path: Path) -> None:
    paths = DataPaths(config_dir.parent / "data")
    timing_market(paths)
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    timing_fiche(fiches, "tm_momentum", [1, 4])
    timing_fiche(fiches, "tm_reversal", [1])
    args = ["--config", str(config_dir), "--fiches", str(fiches), "run"]
    assert scan.main(args) == 0
    registry = TrialRegistry(paths.trials)
    assert registry.n_trials("timing") == 3
    blob = json.loads((paths.reports / "timing_scan_latest.json").read_text())
    assert [r["id"] for r in blob["rows"]].count("tm_momentum") == 2
    sharpes = [r["sharpe"] for r in blob["rows"]]
    assert sharpes == sorted(sharpes, reverse=True)
    holdout = date_to_ms("2024-06-01")
    assert all(r["periods"] < (holdout - T0) // H for r in blob["rows"])  # rien du hold-out
    assert scan.main(args) == 0  # relancer ne recompte pas
    assert registry.n_trials("timing") == 3
    assert (paths.reports / f"timing_scan_{blob['date']}.md").exists()


def test_budget_checked_before_running(config_dir: Path, tmp_path: Path) -> None:
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    timing_fiche(fiches, "tm_momentum", list(range(1, 52)))  # 51 essais > 50 (config de test)
    assert scan.main(["--config", str(config_dir), "--fiches", str(fiches), "run"]) == 1
