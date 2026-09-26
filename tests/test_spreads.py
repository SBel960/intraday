"""Tests de qlab.exchange.spreads : spreads calculés à la main, médianes, commande."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.errors import ExchangeError
from qlab.core.paths import DataPaths
from qlab.core.records import append_record
from qlab.core.timeutils import date_to_ms
from qlab.exchange import spreads as sp
from qlab.exchange.snapshots import SnapshotStore

T0 = date_to_ms("2026-09-26")
BOOK = [
    {"symbol": "BTCEUR", "bidPrice": "99.95", "askPrice": "100.05"},  # 0,10 / 100 = 0,1 %
    {"symbol": "PEPEEUR", "bidPrice": "0.00000990", "askPrice": "0.00001010"},  # 2 %
    {"symbol": "DEADEUR", "bidPrice": "0", "askPrice": "0"},  # carnet vide
    {"symbol": "ETHBTC", "bidPrice": "0.05", "askPrice": "0.0501"},  # hors liste
]


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"x-mbx-used-weight-1m": "4"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_spreads_by_hand() -> None:
    got = sp.spreads_from(BOOK, {"BTCEUR", "PEPEEUR", "DEADEUR"})
    assert got == pytest.approx({"BTCEUR": 0.001, "PEPEEUR": 0.02})


def test_unreadable_book_rejected() -> None:
    with pytest.raises(ExchangeError, match="illisible pour BTCEUR"):
        sp.spreads_from([{"symbol": "BTCEUR", "bidPrice": "x", "askPrice": "1"}], {"BTCEUR"})

    def open_url(url: str, timeout: int) -> _Resp:
        return _Resp(b'{"not": "a list"}')

    with pytest.raises(ExchangeError, match="liste attendue"):
        sp.fetch("https://x", notify=lambda _: None, opener=open_url, clock_ms=lambda: T0)


def test_medians_need_enough_samples(tmp_path: Path) -> None:
    """BTCEUR relevé 3 fois (0,1 %, 0,3 %, 0,2 % ⇒ médiane 0,2 %) ; PEPEEUR une seule fois :
    ignoré avec un minimum de 2 relevés."""
    paths = DataPaths(tmp_path)
    for i, value in enumerate((0.001, 0.003, 0.002)):
        spreads = {"BTCEUR": value} | ({"PEPEEUR": 0.02} if i == 0 else {})
        append_record(paths.spreads, {"ts_ms": T0 + i, "spreads": spreads})
    assert sp.medians(paths, 2) == pytest.approx({"BTCEUR": 0.002})


def test_cli_sample_and_show(
    config_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = DataPaths(config_dir.parent / "data")
    assert sp.main(["--config", str(config_dir), "sample"]) == 1  # pas de snapshot
    symbols = [binance_symbol("BTCEUR", "EUR"), binance_symbol("PEPEEUR", "EUR")]
    symbols += [binance_symbol(s, "USDT") for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
    SnapshotStore(paths, "binance").save(exchange_info(symbols), FALLBACK_FEES, T0, "https://x")
    book = [*BOOK, {"symbol": "BTCUSDT", "bidPrice": "99", "askPrice": "101"}]
    monkeypatch.setattr(sp, "fetch", lambda url, notify: book)
    assert sp.main(["--config", str(config_dir), "sample"]) == 0
    assert sp.main(["--config", str(config_dir), "show", "--min-samples", "1"]) == 0
    out = capsys.readouterr().out
    # config de test : devise USDT, paires tradées BTC/ETH/SOL-USDT ; BTCEUR, PEPEEUR hors liste
    assert "1 spreads relevés sur 3 paires" in out
    assert "BTCUSDT" in out and "2.0000%" in out
    assert json.loads(paths.spreads.read_text().splitlines()[0])["spreads"] == {"BTCUSDT": 0.02}
