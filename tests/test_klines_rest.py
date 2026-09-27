"""Tests de qlab.longterm.klines_rest : bougies REST closes / en cours, contrôle qualité,
requête envoyée, raccord aux archives."""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import pytest

from qlab.core.errors import DataError, ExchangeError
from qlab.core.http import HttpResult
from qlab.core.timeutils import MS_PER_DAY, MS_PER_HOUR, date_to_ms
from qlab.longterm import klines
from qlab.longterm import klines_rest as kr

D = MS_PER_DAY
T0 = date_to_ms("2026-09-25")


def _row(
    t: int, o: str = "100.0", h: str = "110.0", low: str = "90.0", c: str = "105.0"
) -> list[Any]:
    """Une bougie 1d au format de l'API (prix en texte, horodatages et nombre en entiers)."""
    return [t, o, h, low, c, "2.5", t + D - 1, "250.0", 40, "1.0", "100.0", "0"]


ROWS = [_row(T0), _row(T0 + D, c="107.5"), _row(T0 + 2 * D, o="107.5")]
NOW = T0 + 2 * D + 5 * MS_PER_HOUR  # 27/09 05:00 UTC : la bougie du 27 est en cours


def test_closed_and_current_bars() -> None:
    r = kr.parse(ROWS, D, NOW)
    assert r.closed.columns == list(klines.OUTPUT)
    assert r.closed["open_time_ms"].to_list() == [T0, T0 + D]
    assert r.closed["close"].to_list() == [105.0, 107.5]
    assert r.closed["volume_quote"].to_list() == [250.0, 250.0]
    assert r.closed["n_trades"].dtype == pl.Int64 and not r.closed["partial"].any()
    assert (r.current_open_ms, r.current_open, r.anomalies) == (T0 + 2 * D, 107.5, 0)


def test_bar_closes_exactly_at_its_close_time() -> None:
    """À close_time + 1 ms, la bougie du 27 est close ; il n'y a plus de bougie en cours."""
    r = kr.parse(ROWS, D, T0 + 3 * D)
    assert r.closed.height == 3 and r.current_open_ms is None and r.current_open is None
    assert kr.parse(ROWS, D, T0 + 3 * D - 1).closed.height == 2


def test_quality_control_drops_and_counts() -> None:
    """Haut sous la clôture ; bougie non alignée sur le jour : écartées, comptées."""
    bad = [_row(T0, h="100.0", c="105.0"), _row(T0 + D + 1), _row(T0 + 2 * D)]
    r = kr.parse(bad, D, T0 + 3 * D)
    assert r.closed["open_time_ms"].to_list() == [T0 + 2 * D] and r.anomalies == 2


def test_malformed_response() -> None:
    with pytest.raises(ExchangeError, match="12 champs"):
        kr.parse({"code": -1121, "msg": "Invalid symbol."}, D, NOW)
    with pytest.raises(ExchangeError, match="12 champs"):
        kr.parse([[1, 2]], D, NOW)
    empty = kr.parse([], D, NOW)
    assert empty.closed.height == 0 and empty.current_open is None


def test_fetch_sends_the_query_and_limits_the_catch_up() -> None:
    seen: list[str] = []

    def get(url: str, **_: Any) -> HttpResult:
        seen.append(url)
        return HttpResult(json.dumps(ROWS).encode(), {}, 0, 0, 1)

    r = kr.fetch("https://api.x.com/", "BTCEUR", "1d", T0, NOW, notify=print, get=get)
    assert seen == [
        f"https://api.x.com/api/v3/klines?symbol=BTCEUR&interval=1d&startTime={T0}&limit=1000"
    ]
    assert r.closed.height == 2
    with pytest.raises(DataError, match="plus de 1000 bougies"):
        kr.fetch("https://x", "BTCEUR", "1d", NOW - 1000 * D, NOW, notify=print, get=get)
    bad = lambda url, **_: HttpResult(b"<html>", {}, 0, 0, 1)  # noqa: E731
    with pytest.raises(ExchangeError, match="non JSON"):
        kr.fetch("https://x", "BTCEUR", "1d", T0, NOW, notify=print, get=bad)


def test_extend_keeps_archive_and_appends_newer_bars() -> None:
    """Archive jusqu'au 25 (clôture 104, faisant foi) ; REST 25 et 26 : seul le 26 s'ajoute."""
    rest = kr.parse(ROWS, D, NOW).closed
    archive = rest.head(1).with_columns(close=pl.lit(104.0))
    merged = kr.extend(archive, rest)
    assert merged["open_time_ms"].to_list() == [T0, T0 + D]
    assert merged["close"].to_list() == [104.0, 107.5]
    assert kr.extend(archive.clear(), rest).height == 2
    with pytest.raises(DataError, match="colonnes"):
        kr.extend(archive.drop("partial"), rest)
