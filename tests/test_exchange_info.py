"""Tests de qlab.exchange.exchange_info : horloge, échéance, commande complète sans réseau."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from qlab.core.config import load_base
from qlab.core.errors import DataError, ExchangeError
from qlab.core.jsonlog import read_log
from qlab.exchange import exchange_info as ei
from qlab.exchange.exchange_info import clock_offset, fees_record, fetch_exchange_info, is_due
from qlab.exchange.snapshots import Snapshot

T = 1_704_067_200_000  # 2024-01-01T00:00:00Z
HOUR, DAY = 3_600_000, 86_400_000
URL = "https://api.binance.com/api/v3/exchangeInfo"


def _filters() -> list[Any]:
    return [
        {
            "filterType": "PRICE_FILTER",
            "minPrice": "0.01",
            "maxPrice": "1000000",
            "tickSize": "0.01",
        },
        {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000", "stepSize": "0.00001"},
        {
            "filterType": "NOTIONAL",
            "minNotional": "5",
            "applyMinToMarket": True,
            "maxNotional": "9000000",
            "applyMaxToMarket": False,
        },
    ]


def _info(server_time: int = T, btcusdt_status: str = "TRADING") -> dict[str, Any]:
    def sym(name: str, base: str, quote: str, status: str = "TRADING") -> dict[str, Any]:
        return {
            "symbol": name,
            "status": status,
            "baseAsset": base,
            "quoteAsset": quote,
            "filters": _filters(),
        }

    return {
        "serverTime": server_time,
        "symbols": [
            sym("BTCUSDT", "BTC", "USDT", btcusdt_status),
            sym("ETHUSDT", "ETH", "USDT"),
            sym("BTCEUR", "BTC", "EUR"),
            sym("OLDUSDT", "OLD", "USDT", "BREAK"),
        ],
    }


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = {"x-mbx-used-weight-1m": "20"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes | Exception,
    *,
    now: int,
    local_clock: int | None = None,
) -> None:
    """Simule Binance : ``/api/v3/time`` rend ``now`` ; ``exchangeInfo`` rend ``body``.

    ``local_clock`` : ce que lit l'horloge locale pendant les requêtes (défaut ``now``).
    """

    def open_url(url: str, timeout: int) -> _Resp:
        if isinstance(body, Exception):
            raise body
        if url.endswith("/api/v3/time"):
            return _Resp(json.dumps({"serverTime": now}).encode())
        return _Resp(body)

    local = now if local_clock is None else local_clock
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    monkeypatch.setattr(ei, "now_ms", lambda: now)
    monkeypatch.setattr("qlab.core.http.now_ms", lambda: local)


def _kinds(data_root: Path) -> list[str]:
    files = sorted((data_root / "logs" / "exchange_info").glob("*.jsonl"))
    return [str(r["kind"]) for f in files for r in read_log(f).records]


# --- fonctions pures ---------------------------------------------------------------------


def test_clock_offset_by_hand() -> None:
    """Envoi 9 000, réception 9 200 ⇒ milieu 9 100 ; serveur 10 000 ⇒ écart +900 ± 100 ms."""
    c = clock_offset(10_000, 9_000, 9_200)
    assert (c.offset_ms, c.uncertainty_ms) == (900, 100)
    assert clock_offset(8_650, 10_000, 10_000).offset_ms == -1_350  # PC en avance de 1,35 s
    with pytest.raises(DataError, match="revenue en arrière"):
        clock_offset(0, 9_200, 9_000)


def test_is_due() -> None:
    snap = Snapshot(Path("x"), "binance", T, "h", {}, {"symbols": []})
    assert is_due(None, T, 24)
    assert not is_due(snap, T + 24 * HOUR - 1, 24)
    assert is_due(snap, T + 24 * HOUR, 24)  # borne incluse


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        (b"<html>", "illisible"),
        (b'{"x": 1}', "illisible"),
        (b'{"serverTime": 1.5}', "non entier"),
        (b'{"serverTime": true}', "non entier"),
    ],
)
def test_measure_clock_rejects_bad_body(body: bytes, msg: str) -> None:
    def open_url(url: str, timeout: int) -> _Resp:
        return _Resp(body)

    with pytest.raises(ExchangeError, match=msg):
        ei.measure_clock(URL, notify=lambda _: None, opener=open_url, clock_ms=lambda: T)


def test_measure_clock_by_hand() -> None:
    """Horloge locale 9 000 puis 9 200, serveur 10 000 ⇒ +900 ± 100 ms."""
    reads = iter([9_000, 9_200])

    def open_url(url: str, timeout: int) -> _Resp:
        return _Resp(b'{"serverTime": 10000}')

    c = ei.measure_clock(URL, notify=lambda _: None, opener=open_url, clock_ms=lambda: next(reads))
    assert (c.offset_ms, c.uncertainty_ms) == (900, 100)


def test_fees_record_from_config(config_dir: Path) -> None:
    fees = fees_record(load_base(config_dir / "base.yaml").exchange("binance"))
    assert fees["origin"] == "config" and fees["effective_taker_frac"] == 0.001


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        (b"<html>maintenance</html>", "non JSON"),
        (b'{"x": 1}', "absente"),
    ],
)
def test_fetch_rejects_bad_body(body: bytes, msg: str) -> None:
    def open_url(url: str, timeout: int) -> _Resp:
        return _Resp(body)

    with pytest.raises(ExchangeError, match=msg):
        fetch_exchange_info(URL, notify=lambda _: None, opener=open_url, clock_ms=lambda: T)


# --- commande de bout en bout ------------------------------------------------------------


def test_fetch_then_unchanged(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = str(config_dir)
    _serve(monkeypatch, json.dumps(_info()).encode(), now=T + 500)
    assert ei.main(["--config", cfg, "fetch"]) == 0
    out = capsys.readouterr().out
    assert "Nouveau snapshot : 20240101T000000Z.json.zst" in out
    assert "4 paires, 3 en cotation (USDT 2, EUR 1) | 1 version(s)" in out
    _serve(monkeypatch, json.dumps(_info(T + 60_000)).encode(), now=T + 60_000)
    assert ei.main(["--config", cfg, "fetch"]) == 0
    assert "Inchangé" in capsys.readouterr().out
    assert _kinds(tmp_path / "data") == [
        "run.start",
        "clock.offset",
        "fetch.ok",
        "run.start",
        "clock.offset",
        "fetch.ok",
    ]


def test_if_due(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = str(config_dir)
    _serve(monkeypatch, json.dumps(_info()).encode(), now=T)
    assert ei.main(["--config", cfg, "fetch", "--if-due"]) == 0  # aucun snapshot : dû
    assert "Nouveau snapshot" in capsys.readouterr().out
    _serve(monkeypatch, urllib.error.URLError("ne doit pas être appelé"), now=T + 23 * HOUR)
    assert ei.main(["--config", cfg, "fetch", "--if-due"]) == 0
    assert "Pas encore dû" in capsys.readouterr().out
    _serve(monkeypatch, json.dumps(_info(btcusdt_status="BREAK")).encode(), now=T + 24 * HOUR)
    assert ei.main(["--config", cfg, "fetch", "--if-due"]) == 0
    assert "Nouveau snapshot : 20240102T000000Z.json.zst" in capsys.readouterr().out


def test_traded_pair_change_alerts(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """BTCUSDT (tradée dans la config de test) passe en BREAK : alerte visible et journalisée."""
    cfg = str(config_dir)
    _serve(monkeypatch, json.dumps(_info()).encode(), now=T)
    ei.main(["--config", cfg, "fetch"])
    capsys.readouterr()
    _serve(monkeypatch, json.dumps(_info(T + DAY, "BREAK")).encode(), now=T + DAY)
    assert ei.main(["--config", cfg, "fetch"]) == 0
    captured = capsys.readouterr()
    assert "Changements depuis 20240101T000000Z.json.zst : +0 paire(s), -0, 1 statut(s)" in (
        captured.out
    )
    assert "ALERTE : paire(s) tradée(s) touchée(s) : BTCUSDT : TRADING → BREAK" in captured.err
    assert _kinds(tmp_path / "data")[-3:] == [
        "fetch.ok",
        "snapshot.diff",
        "snapshot.traded_pairs_changed",
    ]


def test_clock_drift_alert(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Horloge locale en avance de 1 350 ms (seuil 500) : alerte ; à 400 ms, rien."""
    cfg = str(config_dir)
    _serve(monkeypatch, json.dumps(_info()).encode(), now=T, local_clock=T + 1_350)
    assert ei.main(["--config", cfg, "fetch"]) == 0
    assert "ALERTE : horloge locale décalée de +1350 ms" in capsys.readouterr().err
    assert "clock.drift" in _kinds(tmp_path / "data")
    _serve(
        monkeypatch, json.dumps(_info()).encode(), now=T + 1000, local_clock=T + 1400
    )  # 400 ms d'avance
    ei.main(["--config", cfg, "fetch"])
    assert "ALERTE" not in capsys.readouterr().err


def test_show_symbol(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _serve(monkeypatch, json.dumps(_info()).encode(), now=T)
    ei.main(["--config", str(config_dir), "fetch"])
    capsys.readouterr()
    assert ei.main(["--config", str(config_dir), "show", "--symbol", "BTCEUR"]) == 0
    out = capsys.readouterr().out
    assert "BTCEUR : tick 0.01, pas 0.00001, minNotional 5" in out
    assert "Frais (config) : maker 0.1000%, taker 0.1000%" in out


def test_cli_errors(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = str(config_dir)
    assert ei.main(["--config", cfg, "show"]) == 1
    assert "aucun snapshot" in capsys.readouterr().err
    error_418 = urllib.error.HTTPError(URL, 418, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]
    _serve(monkeypatch, error_418, now=T)
    assert ei.main(["--config", cfg, "fetch"]) == 1
    assert "ERREUR (ExchangeError) : HTTP 418" in capsys.readouterr().err
    assert ei.main(["--config", cfg, "--exchange", "bybit", "show"]) == 1
    assert "non configuré : 'bybit'" in capsys.readouterr().err
