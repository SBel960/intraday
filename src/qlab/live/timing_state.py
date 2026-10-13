"""Résultats du volet timing affichés par le cockpit : dernier balayage, verdicts, hold-out.

Lecture seule des sorties de ``timing/scan.py`` et ``timing/verdict.py`` (JSON des rapports,
registre du hold-out) ; rien n'est recalculé ici. Absent ⇒ ``None`` (balayage pas encore fait).
"""

from __future__ import annotations

import json
from typing import Any

from qlab.core.paths import DataPaths
from qlab.core.records import read_records

TOP = 15  # lignes du classement envoyées à la page


def _read(paths: DataPaths, name: str) -> Any:
    path = paths.reports / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def timing_state(paths: DataPaths) -> dict[str, Any] | None:
    """``scan`` (date, N, meilleurs essais), ``verdicts``, ``holdout`` ; ``None`` sans balayage."""
    scan = _read(paths, "timing_scan_latest.json")
    if scan is None:
        return None
    verdicts = _read(paths, "timing_verdict_latest.json")
    holdout = read_records(paths.timing_holdout)
    return {
        "date": scan["date"],
        "n_trials": scan["n_trials"],
        "rows": scan["rows"][:TOP],
        "verdicts": None if verdicts is None else verdicts["verdicts"],
        "holdout": holdout[0] if holdout else None,
    }
