"""Tests de qlab.data.histdata : périodes, jeton, minutes → heures UTC, commande (sans réseau)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from qlab.core import http
from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.data import histdata as hd

PAGE = b'<input type="hidden" name="tk" id="tk" value="abc123" />'


def _zip(lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("DAT_ASCII_EURUSD_M1_2019.csv", "\n".join(lines))
        z.writestr("DAT_ASCII_EURUSD_M1_2019.txt", "rapport")
    return buffer.getvalue()


def test_periods_years_then_months() -> None:
    """Années passées entières, puis mois achevés de l'année en cours."""
    got = hd.periods(2019, date_to_ms("2021-03-15"))
    assert got == [("2019", "2019"), ("2020", "2020"), ("2021/1", "202101"), ("2021/2", "202102")]
    assert hd.raw_key("EURUSD", "202102") == "EURUSD/202102.zip"
    assert b"datemonth=202102" in hd.form("tk1", "EURUSD", "202102")


def test_minutes_become_utc_hours() -> None:
    """New York sans heure d'été (+5 h) : 17:58 et 17:59 → 22:00 UTC ; 18:00 → 23:00 UTC."""
    blob = _zip(
        [
            "20190101 175800;1.10;1.12;1.09;1.11;0",
            "20190101 175900;1.11;1.13;1.10;1.12;0",
            "20190101 180000;1.12;1.12;1.12;1.12;0",
        ]
    )
    got = hd.hourly(blob, 5)
    h22 = date_to_ms("2019-01-01") + 22 * MS_PER_HOUR
    assert got["t_ms"].to_list() == [h22, h22 + MS_PER_HOUR]
    assert got.row(0)[1:] == pytest.approx((1.10, 1.13, 1.09, 1.12, 2.0))
    two = io.BytesIO()
    with zipfile.ZipFile(two, "w") as z:
        z.writestr("a.csv", "x")
        z.writestr("b.csv", "y")
    with pytest.raises(DataError, match="un CSV"):
        hd.hourly(two.getvalue(), 5)


def test_download_then_build(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Page (jeton) puis formulaire avec la page en Referer ; 2020 absente chez la source ;
    un fichier présent n'est pas redemandé."""
    monkeypatch.setattr(hd, "now_ms", lambda: date_to_ms("2021-01-20"))
    monkeypatch.setattr(hd.time, "sleep", lambda s: None)
    calls: list[tuple[str, Any]] = []

    def fake_get(
        url: str, *, headers: Any = None, data: Any = None, **_: object
    ) -> http.HttpResult:
        calls.append((url, data))
        if data is None:
            return http.HttpResult(PAGE, {}, 0, 0, 1)
        assert headers["Referer"].endswith("eurusd/2019") or headers["Referer"].endswith("2020")
        body = _zip(["20190102 000000;1.0;1.0;1.0;1.0;0"]) if b"2019" in data else b""
        return http.HttpResult(body, {}, 0, 0, 1)

    monkeypatch.setattr(http, "get", fake_get)
    args = ["--config", str(config_dir)]
    assert hd.main([*args, "download", "--only", "EURUSD"]) == 0
    assert len(calls) == 4 and b"tk=abc123" in calls[1][1]
    assert hd.main([*args, "download", "--only", "EURUSD"]) == 0
    assert len(calls) == 4
    assert hd.main([*args, "build", "--only", "EURUSD"]) == 0
    bars = pl.read_parquet(DataPaths(config_dir.parent / "data").trading_hourly("EURUSD"))
    assert bars["t_ms"].to_list() == [date_to_ms("2019-01-02") + 5 * MS_PER_HOUR]
    assert bars["spread_frac"].to_list() == [0.00005]
    cfg = load_config(config_dir).trading
    with pytest.raises(DataError, match="aucun fichier"):
        hd.build(DataPaths(config_dir.parent / "data"), cfg, cfg.instruments[1], hd.now_ms())


def test_page_without_token_means_nothing_to_fetch(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http, "get", lambda url, **_: http.HttpResult(b"<html/>", {}, 0, 0, 1))
    assert hd.fetch(load_config(config_dir).trading, "EURUSD", "2019", "2019") == b""
