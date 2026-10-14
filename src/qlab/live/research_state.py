"""Résultats de recherche et paper des annonces affichés par le cockpit (lecture seule).

- volets ``timing`` (perpétuels), ``multi`` (multi-actifs), ``trading`` (horaire) : dernier
  balayage (``reports/{volet}_scan_latest.json``), verdicts, hold-out (registre propre au
  volet) ; ``None`` tant que le volet n'a pas été balayé ;
- ``events`` : dernières annonces suivies (retraits, cotations) et trades du paper des annonces
  (``live/paper_events.py``), avec le résultat cumulé.

Rien n'est recalculé ici : seules les sorties des commandes sont relues.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qlab.core.paths import DataPaths
from qlab.core.records import read_records
from qlab.events.classify import classify
from qlab.events.news import stored
from qlab.live.paper_events import JOURNAL

TOP = 15  # lignes du classement envoyées à la page
RECENT = 8  # dernières annonces suivies affichées
FOLLOWED = ("delisting", "listing")


def _read(paths: DataPaths, name: str) -> Any:
    path = paths.reports / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def volet_state(paths: DataPaths, volet: str, holdout: Path) -> dict[str, Any] | None:
    """Date, N, meilleurs essais, verdicts et hold-out d'un volet ; ``None`` sans balayage."""
    scan = _read(paths, f"{volet}_scan_latest.json")
    if scan is None:
        return None
    verdicts = _read(paths, f"{volet}_verdict_latest.json")
    used = read_records(holdout)
    return {
        "date": scan["date"],
        "n_trials": scan["n_trials"],
        "rows": scan["rows"][:TOP],
        "verdicts": None if verdicts is None else verdicts["verdicts"],
        "holdout": used[0] if used else None,
    }


def events_state(paths: DataPaths) -> dict[str, Any]:
    """Dernières annonces suivies et trades du paper des annonces."""
    news = [r for r in stored(paths) if classify(r["title"], r["catalog"]).kind in FOLLOWED]
    recent = sorted(news, key=lambda r: r["published_ms"])[-RECENT:][::-1]
    journal = read_records(paths.paper_journal(JOURNAL))
    closes = {j["id"]: j for j in journal if j["type"] == "close"}
    trades = [{**s, "close": closes.get(s["id"])} for s in journal if s["type"] == "signal"][::-1]
    return {
        "recent": [
            {
                "title": r["title"],
                "published_ms": r["published_ms"],
                "live": r["received_ms"] is not None,
            }
            for r in recent
        ],
        "trades": trades,
        "total_quote": sum(c["net_quote"] for c in closes.values()),
        "closed": len(closes),
    }


def research_state(paths: DataPaths) -> dict[str, Any]:
    """Clés ``timing``, ``multi``, ``trading`` et ``events`` de l'état du cockpit."""
    return {
        "timing": volet_state(paths, "timing", paths.timing_holdout),
        "multi": volet_state(paths, "multi", paths.markets_holdout),
        "trading": volet_state(paths, "trading", paths.trading_holdout),
        "events": events_state(paths),
    }
