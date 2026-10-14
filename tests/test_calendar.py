"""Tests de qlab.events.calendar : heure d'été, pages de la Fed, fenêtres, contrôle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qlab.core import http
from qlab.core.config import load_config
from qlab.core.timeutils import date_to_ms
from qlab.events import calendar as cal

H = 3_600_000


def _meeting(month: str, days: str) -> str:
    return (
        f'<div class="fomc-meeting__month"><strong>{month}</strong></div>'
        f'<div class="fomc-meeting__date">{days}</div>'
    )


CURRENT = "<h4>2024 FOMC Meetings</h4>" + "".join(
    _meeting(m, d)
    for m, d in (
        ("January", "30-31"),
        ("March", "19-20*"),
        ("Apr/May", "30-1"),
        ("August", "22 (notation vote)"),
    )
)
HISTORICAL = """
<h5>January 28-29 Meeting - 2020</h5><h5>March 2 (unscheduled) Meeting - 2020</h5>
<h5>March 17-18 (cancelled) Meeting - 2020</h5><h5>July 28-29 Meeting - 2020</h5>
"""


def test_statement_time_follows_new_york_summer_time() -> None:
    """14:00 à New York : 19:00 UTC en hiver (31 janvier), 18:00 UTC en été (20 mars 2024)."""
    assert cal.statement_ms("2024-01-31", "14:00") == date_to_ms("2024-01-31") + 19 * H
    assert cal.statement_ms("2024-03-20", "14:00") == date_to_ms("2024-03-20") + 18 * H


def test_parse_pages_keep_scheduled_meetings_only() -> None:
    assert cal.parse_current(CURRENT) == ["2024-01-31", "2024-03-20", "2024-05-01"]
    assert cal.parse_historical(HISTORICAL) == ["2020-01-29", "2020-07-29"]


def test_risk_windows(config_dir: Path) -> None:
    cfg = load_config(config_dir).events
    start, end, reason = cal.risk_windows(cfg)[0]
    t = date_to_ms("2020-01-29") + 19 * H
    assert (start, end) == (t - 30 * 60_000, t + 120 * 60_000)
    assert "2020-01-29" in reason


def test_check_reports_differences_without_editing(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Config de test : 2020-01-29, 2020-07-29, 2024-03-20. La Fed publie aussi 2024-01-31 :
    manquant ; 2020 relu sur la page historique."""
    pages = {"fomccalendars": CURRENT, "fomchistorical2020": HISTORICAL}

    def fake_get(url: str, **_: Any) -> http.HttpResult:
        key = next(k for k in pages if k in url)
        return http.HttpResult(pages[key].encode(), {}, 0, 0, 1)

    monkeypatch.setattr(http, "get", fake_get)
    assert cal.main(["--config", str(config_dir), "check"]) == 1
    out = capsys.readouterr().out
    assert "absent de la config : communiqué du 2024-01-31" in out
    assert "inconnu de la Fed" not in out
