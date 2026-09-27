"""Tests de qlab.live.cockpit : essai suivi, signal du jour, minuteurs, routes du serveur local."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.records import append_record
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.live import cockpit as ck
from qlab.longterm import strategies
from qlab.longterm.allocation import Policy
from qlab.longterm.signals import DATE
from qlab.research.hypothesis import load_hypothesis

ROOT = Path(__file__).resolve().parent.parent / "hypotheses"
TS = load_hypothesis(ROOT / "lt_ts_momentum.yaml")
T0 = date_to_ms("2026-08-01")
DAYS = 40


def _market(config_dir: Path) -> strategies.Market:
    """BTC monte, ETH baisse, SOL monte puis retombe sous son niveau d'il y a 30 jours."""
    config = load_config(config_dir)
    i = np.arange(DAYS, dtype=float)
    closes = pl.DataFrame(
        {
            DATE: [T0 + k * MS_PER_DAY for k in range(DAYS)],
            "BTCUSDT": 100 + i,
            "ETHUSDT": 100 - i,
            "SOLUSDT": np.where(i < 35, 100 + i, 90.0),
        }
    )
    flat = pl.DataFrame({DATE: closes[DATE], "funding_1d": [0.0] * DAYS})
    return strategies.Market(closes, closes, flat, {}, 365, config.longterm.signals, {})


def test_signal_of_the_last_close(config_dir: Path) -> None:
    """Jour 39 : BTC 139 > 109 (investi), ETH 61 < 91 (cash), SOL 90 < 109 (cash)."""
    s = ck.signal(_market(config_dir), TS, {"lookback_days": 30.0})
    assert s["date"] == "2026-09-09"
    assert s["close_ms"] == date_to_ms("2026-09-10") - 1
    assert s["weights"] == {"BTCUSDT": pytest.approx(1 / 3), "ETHUSDT": 0.0, "SOLUSDT": 0.0}
    assert s["closes"] == {"BTCUSDT": 139.0, "ETHUSDT": 61.0, "SOLUSDT": 90.0}
    assert s["policy_days"] == 7


def test_qlab_timers_only_and_in_ms() -> None:
    raw = json.dumps(
        [
            {"unit": "qlab-spreads.timer", "activates": "qlab-spreads.service",
             "next": 1_790_539_395_098_712, "last": 0},
            {"unit": "apt-daily.timer", "activates": "apt-daily.service", "next": 1, "last": 1},
        ]
    )  # fmt: skip
    assert ck.qlab_timers(raw) == [
        {"unit": "qlab-spreads.timer", "service": "qlab-spreads.service",
         "next": 1_790_539_395_098, "last": None}
    ]  # fmt: skip
    assert ck.qlab_timers("") == []


def test_collect_needs_a_snapshot(config_dir: Path) -> None:
    config = load_config(config_dir)
    paths = DataPaths(config.base.data.root)
    paths.root.mkdir(parents=True)
    with pytest.raises(DataError, match="aucun snapshot"):
        ck.collect(config, paths, ROOT, "lt_ts_momentum · lookback_days=30", "binance")


class _Stub:
    def __init__(self, body: bytes | None) -> None:
        self._body = body

    def body(self) -> bytes:
        if self._body is None:
            raise DataError("klines absentes")
        return self._body


@pytest.fixture
def serve() -> Iterator[object]:
    servers = []

    def start(body: bytes | None) -> str:
        handler = ck._handler(_Stub(body), b"<title>Cockpit qlab</title>")  # type: ignore[arg-type]
        server = ThreadingHTTPServer((ck.HOST, 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://{ck.HOST}:{server.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _get(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers["Content-Type"], r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers["Content-Type"], e.read().decode()


def test_routes(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(b'{"trials": {"n": 14}}')
    assert _get(url + "/") == (200, "text/html; charset=utf-8", "<title>Cockpit qlab</title>")
    assert _get(url + "/api/state")[2] == '{"trials": {"n": 14}}'
    assert _get(url + "/../.env")[0] == 404


def test_data_error_is_reported_not_crashed(serve) -> None:  # type: ignore[no-untyped-def]
    status, kind, body = _get(serve(None) + "/api/state")
    assert (status, kind, json.loads(body)) == (
        503,
        "application/json",
        {"error": "klines absentes"},
    )


def test_page_ships_with_the_module() -> None:
    page = ck.PAGE.read_text(encoding="utf-8")
    assert "<title>Cockpit qlab</title>" in page and "/api/state" in page


def test_next_rebalance() -> None:
    """Hebdomadaire ancré au jour 0 : après le jour 2 → jour 7 ; après le jour 7 → jour 14."""
    weekly = Policy("calendar", period_days=7)
    assert ck.next_rebalance(T0 + 2 * MS_PER_DAY, T0, weekly) == T0 + 7 * MS_PER_DAY
    assert ck.next_rebalance(T0 + 7 * MS_PER_DAY, T0, weekly) == T0 + 14 * MS_PER_DAY
    assert ck.next_rebalance(T0, T0, Policy("buy_hold")) is None


def test_paper_state_from_the_journal(config_dir: Path) -> None:
    """Pas de journal ⇒ None ; start 50 € puis deux barres (achat BTC, puis refus) ⇒ état."""
    paths = DataPaths(load_config(config_dir).base.data.root)
    weekly = Policy("calendar", period_days=7)
    trial = "lt_ts_momentum · lookback_days=30"
    assert ck.paper_state(paths, trial, T0, weekly, 60) is None
    journal = paths.paper_journal("lt_ts_momentum_lookback_days_30")
    buy = {
        "side": "BUY",
        "symbol": "BTCUSDT",
        "qty": "0.0001",
        "accepted": True,
        "reason": None,
        "fee": "0.01",
    }
    no = {
        "side": "BUY",
        "symbol": "ETHUSDT",
        "qty": "0.001",
        "accepted": False,
        "reason": "x",
        "fee": "0",
    }
    for record in (
        {"kind": "start", "run_ms": T0, "capital": "50.0", "tier": "t0", "quote": "USDT"},
        _day(0, "50.0", [buy], cash="40", held={"BTCUSDT": "0.0001", "ETHUSDT": "0"}),
        _day(3, "51.5", [no], cash="40", held={"BTCUSDT": "0.0001"}, missed_bars=2),
    ):
        append_record(journal, record)
    st = ck.paper_state(paths, trial, T0, weekly, 60)
    assert st is not None
    assert (st["capital"], st["cash"], st["held"]) == ("50.0", "40", {"BTCUSDT": "0.0001"})
    assert (st["days"], st["min_days"], st["missed_bars"], st["halted"]) == (3, 60, 2, False)
    assert st["equity"] == [[T0, "50.0"], [T0 + 3 * MS_PER_DAY, "51.5"]]
    assert [r["orders"] for r in st["recent"]] == [
        ["BUY ETHUSDT 0.001 refusé (x)"],
        ["BUY BTCUSDT 0.0001"],
    ]
    assert st["next_rebalance_bar_ms"] == T0 + 7 * MS_PER_DAY


def _day(
    i: int, value: str, executions: list[dict[str, object]], **after: object
) -> dict[str, object]:
    missed = after.pop("missed_bars", 0)
    return {
        "kind": "day",
        "bar_ms": T0 + i * MS_PER_DAY,
        "value_close": value,
        "risk": {"action": "trade", "reasons": []},
        "executions": executions,
        "after": after,
        "halted": False,
        "missed_bars": missed,
    }
