"""Tests de qlab.live.timing_state : lecture des sorties du volet timing pour le cockpit."""

from __future__ import annotations

import json
from pathlib import Path

from qlab.core.paths import DataPaths
from qlab.core.records import append_record
from qlab.live.timing_state import TOP, timing_state


def test_nothing_before_a_scan(tmp_path: Path) -> None:
    assert timing_state(DataPaths(tmp_path)) is None


def test_scan_verdicts_and_holdout(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.reports.mkdir(parents=True)
    rows = [{"id": f"tm_{i}", "sharpe": -i} for i in range(TOP + 5)]
    scan = {"date": "2026-10-14", "n_trials": 77, "rows": rows}
    (paths.reports / "timing_scan_latest.json").write_text(json.dumps(scan))
    got = timing_state(paths)
    assert got is not None
    assert (got["n_trials"], len(got["rows"]), got["verdicts"], got["holdout"]) == (
        77,
        TOP,
        None,
        None,
    )
    verdicts = {"verdicts": [{"id": "tm_0", "verdict": "rejeté"}]}
    (paths.reports / "timing_verdict_latest.json").write_text(json.dumps(verdicts))
    append_record(paths.timing_holdout, {"ts_ms": 1, "name": "tm_0 x"})
    got = timing_state(paths)
    assert got is not None
    assert got["verdicts"] == [{"id": "tm_0", "verdict": "rejeté"}]
    assert got["holdout"]["name"] == "tm_0 x"
