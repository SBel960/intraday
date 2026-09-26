"""Tests de qlab.data.tardis : jours gratuits, vérification des fichiers, lecture, commande."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from qlab.core.errors import DataError, ExchangeError
from qlab.core.http import HttpResult
from qlab.core.jsonlog import read_log
from qlab.core.paths import DataPaths
from qlab.data import tardis
from qlab.data.tardis import HEADER, availability, free_days, jobs, read_quotes, verify

ROWS = [  # horodatages en µs, comme Tardis
    "binance,BTCUSDT,1704067200000999,1704067200001500,0.5,42000.02,42000.00,0.7",
    "binance,BTCUSDT,1704067200250000,1704067200250100,0.4,42000.03,42000.01,0.2",
]


def _csv_gz(rows: list[str] = ROWS, header: str = HEADER) -> bytes:
    return gzip.compress(("\n".join([header, *rows]) + "\n").encode())


# --- jours gratuits ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("since", "today", "expected"),
    [
        ("2020-01-03", "2020-05-15", ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]),
        ("2020-02-01", "2020-03-15", ["2020-02-01", "2020-03-01"]),  # 1er du mois disponible
        ("2020-02-01", "2020-03-01", ["2020-02-01"]),  # le 1er mars n'est pas encore fini
        ("2020-12-15", "2021-02-02", ["2021-01-01", "2021-02-01"]),  # passage d'année
        ("2026-09-20", "2026-09-26", []),
    ],
)
def test_free_days(since: str, today: str, expected: list[str]) -> None:
    assert free_days(since, today) == expected


def test_urls_and_keys() -> None:
    assert tardis.file_url("https://datasets.tardis.dev/", "BTCEUR", "2026-09-01") == (
        "https://datasets.tardis.dev/v1/binance/book_ticker/2026/09/01/BTCEUR.csv.gz"
    )
    assert tardis.file_key("BTCEUR", "2026-09-01") == "binance/book_ticker/BTCEUR/2026-09-01.csv.gz"


def test_availability() -> None:
    doc = {
        "availableSymbols": [
            {"id": "btceur", "type": "spot", "availableSince": "2020-01-03T00:00:00.000Z"},
            {"id": "btcusdt", "type": "perpetual", "availableSince": "2019-09-01T00:00:00.000Z"},
        ]
    }
    assert availability(lambda url: json.dumps(doc).encode(), "https://api.x") == {
        "BTCEUR": "2020-01-03"
    }  # les contrats à terme sont ignorés
    with pytest.raises(ExchangeError, match="illisibles"):
        availability(lambda url: b"<html>", "https://api.x")


# --- vérification et lecture -------------------------------------------------------------


def test_verify_and_read_by_hand() -> None:
    data = _csv_gz()
    assert verify(data, "BTCUSDT") is data
    quotes = read_quotes(data)
    assert quotes.schema == {"ts_ms": pl.Int64, "bid": pl.Float64, "ask": pl.Float64}
    # 1704067200000999 µs → 1704067200000 ms (troncature) ; horodatage d'échange, pas local
    assert quotes["ts_ms"].to_list() == [1_704_067_200_000, 1_704_067_200_250]
    assert quotes["bid"].to_list() == [42000.00, 42000.01]
    assert quotes["ask"].to_list() == [42000.02, 42000.03]


@pytest.mark.parametrize(
    ("data", "msg"),
    [
        (b"pas du gzip", "gzip corrompu"),
        (_csv_gz()[:-10], "gzip"),  # tronqué
        (_csv_gz(header="a,b,c"), "en-tête"),
        (_csv_gz(rows=[]), "vide"),
        (_csv_gz(rows=[ROWS[0].replace("BTCUSDT", "ETHUSDT", 1)]), "autre paire"),
        (_csv_gz(rows=["bybit" + ROWS[0].removeprefix("binance")]), "autre échange"),
    ],
)
def test_verify_rejects(data: bytes, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        verify(data, "BTCUSDT")


def test_jobs(tmp_path: Path) -> None:
    todo = jobs(
        DataPaths(tmp_path),
        lambda url: _csv_gz(),
        base_url="https://d.x",
        symbols=["BTCUSDT"],
        since={"BTCUSDT": "2024-01-01"},
        today="2024-03-10",
    )
    assert [j.key for j in todo] == [
        "binance/book_ticker/BTCUSDT/2024-01-01.csv.gz",
        "binance/book_ticker/BTCUSDT/2024-02-01.csv.gz",
        "binance/book_ticker/BTCUSDT/2024-03-01.csv.gz",
    ]
    assert todo[0].dest == tmp_path / "raw" / "tardis" / todo[0].key
    assert todo[0].fetch_verified() == _csv_gz()
    with pytest.raises(DataError, match="inconnu de Tardis"):
        jobs(
            DataPaths(tmp_path),
            lambda url: b"",
            base_url="https://d.x",
            symbols=["XYZ"],
            since={},
            today="2024-03-10",
        )


# --- commande ----------------------------------------------------------------------------


def _serve(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> list[str]:
    calls: list[str] = []
    meta = {
        "availableSymbols": [
            {"id": s, "type": "spot", "availableSince": "2024-01-15T00:00:00.000Z"}
            for s in ("btcusdt", "ethusdt", "solusdt")
        ]
    }

    def get(url: str, **kw: Any) -> HttpResult:
        calls.append(url)
        if url.startswith("https://api.tardis.dev"):
            body = json.dumps(meta).encode()
        else:
            body = files[url.rsplit("/v1/binance/book_ticker/", 1)[1]]
        return HttpResult(body, {}, 0, 0, 1)

    monkeypatch.setattr("qlab.core.http.get", get)
    monkeypatch.setattr(tardis, "now_ms", lambda: 1_709_337_600_000)  # 2024-03-02
    return calls


def test_cli_sync_and_resume(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    files = {
        f"2024/{m}/01/{s}.csv.gz": _csv_gz(rows=[r.replace("BTCUSDT", s) for r in ROWS])
        for m in ("02", "03")
        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    }
    _serve(monkeypatch, files)
    assert tardis.main(["--config", str(config_dir), "sync"]) == 0
    assert "téléchargées : 6" in capsys.readouterr().out  # 3 paires × (1er fév., 1er mars)
    calls = _serve(monkeypatch, files)
    assert tardis.main(["--config", str(config_dir), "sync"]) == 0
    assert "Déjà présentes : 6" in capsys.readouterr().out
    assert len(calls) == 1  # seulement la disponibilité : rien de retéléchargé
    logs = sorted((tmp_path / "data" / "logs" / "tardis").glob("*.jsonl"))
    assert [r["kind"] for f in logs for r in read_log(f).records].count("tardis.sync") == 2


def test_cli_corrupt_file_fails_cleanly(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    files = {f"2024/{m}/01/BTCUSDT.csv.gz": _csv_gz() for m in ("02", "03")}
    files["2024/03/01/BTCUSDT.csv.gz"] = b"corrompu"
    _serve(monkeypatch, files)
    assert tardis.main(["--config", str(config_dir), "sync", "--symbols", "BTCUSDT"]) == 1
    assert "ÉCHEC : binance/book_ticker/BTCUSDT/2024-03-01.csv.gz : BTCUSDT : gzip corrompu" in (
        capsys.readouterr().out
    )
    assert not (
        tmp_path
        / "data"
        / "raw"
        / "tardis"
        / "binance"
        / "book_ticker"
        / "BTCUSDT"
        / "2024-03-01.csv.gz"
    ).exists()
