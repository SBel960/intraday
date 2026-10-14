"""Tests de qlab.live.research_state : volets de recherche et paper des annonces (lecture)."""

from __future__ import annotations

import json
from pathlib import Path

from qlab.core.paths import DataPaths
from qlab.core.records import append_record
from qlab.live import research_state as rs
from qlab.live.paper_events import JOURNAL


def test_nothing_before_a_scan(tmp_path: Path) -> None:
    got = rs.research_state(DataPaths(tmp_path))
    assert (got["timing"], got["multi"], got["trading"]) == (None, None, None)
    assert got["events"] == {"recent": [], "trades": [], "total_quote": 0, "closed": 0}


def test_scan_verdicts_and_holdout_per_volet(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.reports.mkdir(parents=True)
    rows = [{"id": f"mk_{i}", "sharpe": -i} for i in range(rs.TOP + 5)]
    scan = {"date": "2026-10-14", "n_trials": 23, "rows": rows}
    (paths.reports / "multi_scan_latest.json").write_text(json.dumps(scan))
    verdicts = {"verdicts": [{"id": "mk_0", "verdict": "rejeté"}]}
    (paths.reports / "multi_verdict_latest.json").write_text(json.dumps(verdicts))
    append_record(paths.markets_holdout, {"ts_ms": 1, "name": "mk_0 x"})
    got = rs.research_state(paths)
    multi = got["multi"]
    assert multi is not None and got["timing"] is None
    assert (multi["n_trials"], len(multi["rows"]), multi["holdout"]["name"]) == (
        23,
        rs.TOP,
        "mk_0 x",
    )
    assert multi["verdicts"] == [{"id": "mk_0", "verdict": "rejeté"}]


def test_events_recent_followed_announcements_and_trades(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    for i, (title, catalog, received) in enumerate(
        [
            ("Binance Will Delist AA on 2026-11-01", "delisting", 5),
            ("Notice of Removal of Spot Trading Pairs", "delisting", None),
            ("Binance Lists BB", "listing", None),
        ]
    ):
        append_record(
            paths.events_news("2026-10-14"),
            {
                "source_id": i,
                "title": title,
                "catalog": catalog,
                "published_ms": i,
                "received_ms": received,
                "backfilled": received is None,
            },
        )
    journal = paths.paper_journal(JOURNAL)
    append_record(journal, {"type": "signal", "id": 1, "symbol": "AAUSDT", "decision": "pending"})
    append_record(journal, {"type": "close", "id": 1, "net_quote": 1.5})
    got = rs.events_state(paths)
    assert [r["title"] for r in got["recent"]] == [
        "Binance Lists BB",
        "Binance Will Delist AA on 2026-11-01",
    ]
    assert got["recent"][1]["live"] and not got["recent"][0]["live"]
    assert got["trades"][0]["close"]["net_quote"] == 1.5
    assert (got["closed"], got["total_quote"]) == (1, 1.5)
