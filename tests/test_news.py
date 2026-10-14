"""Tests de qlab.events.news : lecture des pages, dédoublonnage, réception, refus d'horloge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qlab.core import http
from qlab.core.config import EventCatalog
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.events import news
from qlab.exchange.exchange_info import ClockOffset

DELIST = EventCatalog(161, "delisting")


def _page(ids: list[int]) -> bytes:
    articles = [
        {
            "id": i,
            "code": f"c{i}",
            "title": f"Binance Will Delist T{i} on 2024-01-0{i}",
            "releaseDate": 1_704_067_200_000 + i * 86_400_000,
        }
        for i in ids
    ]
    return json.dumps({"data": {"catalogs": [{"articles": articles}]}}).encode()


def test_parse_page_and_malformed_answer() -> None:
    got = news.parse_page(_page([1]), DELIST)
    assert got == [
        {
            "source_id": 1,
            "code": "c1",
            "title": "Binance Will Delist T1 on 2024-01-01",
            "published_ms": 1_704_153_600_000,
            "catalog": "delisting",
        }
    ]
    assert news.parse_page(json.dumps({"data": {"catalogs": []}}).encode(), DELIST) == []
    with pytest.raises(DataError, match="forme"):
        news.parse_page(b'{"data": {}}', DELIST)


def test_backfill_pages_until_empty_and_never_twice(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = {1: [1, 2], 2: [3], 3: []}

    def fake_get(url: str, **_: Any) -> http.HttpResult:
        page = int(url.split("pageNo=")[1].split("&", maxsplit=1)[0])
        ids = pages[page] if "catalogId=161" in url else []
        return http.HttpResult(_page(ids), {}, 0, 0, 1)

    monkeypatch.setattr(http, "get", fake_get)
    monkeypatch.setattr(news.time, "sleep", lambda s: None)
    assert news.main(["--config", str(config_dir), "backfill"]) == 0
    assert news.main(["--config", str(config_dir), "backfill"]) == 0  # rien en double
    stored = news.stored(DataPaths(config_dir.parent / "data"))
    assert [r["source_id"] for r in stored] == [1, 2, 3]
    assert all(r["backfilled"] and r["received_ms"] is None for r in stored)
    assert len(stored[0]["sha256"]) == 64


def test_poll_stamps_reception_with_server_time_or_refuses(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        http,
        "get",
        lambda url, **_: http.HttpResult(_page([5] if "161" in url else []), {}, 0, 0, 1),
    )
    monkeypatch.setattr(news, "now_ms", lambda: 1_000_000)
    monkeypatch.setattr(news, "measure_clock", lambda *a, **k: ClockOffset(250, 10))
    assert news.main(["--config", str(config_dir), "poll"]) == 0
    got = news.stored(DataPaths(config_dir.parent / "data"))
    assert got[0]["received_ms"] == 1_000_250 and not got[0]["backfilled"]
    monkeypatch.setattr(news, "measure_clock", lambda *a, **k: ClockOffset(60_000, 10))
    assert news.main(["--config", str(config_dir), "poll"]) == 1  # horloge trop décalée
