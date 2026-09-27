"""Tests de qlab.live.paper : start / run / status / resume en ligne de commande, horloge,
barre déjà traitée, fiche modifiée, raccord des bougies REST (horloge et API simulées)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.config import load_config
from qlab.core.paths import DataPaths
from qlab.core.records import append_record, read_records
from qlab.core.timeutils import MS_PER_DAY, MS_PER_MIN, date_to_ms
from qlab.exchange.exchange_info import ClockOffset
from qlab.exchange.snapshots import SnapshotStore
from qlab.live import paper
from qlab.longterm import funding, klines, klines_rest, strategies

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
ARCHIVED = 63  # barres 0 à 62 archivées ; la 63 (multiple de 7 : jour de rééquilibrage) par l'API
TRADE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TRIAL = "lt_ts_momentum · lookback_days=30"
HYPOTHESES = Path(__file__).resolve().parent.parent / "hypotheses"
NOW = T0 + (ARCHIVED + 1) * D + 5 * MS_PER_MIN  # 00:05 UTC le lendemain de la barre 63
# BTC monte (investi), ETH baisse (cash), SOL monte (investi)
DRIFT = {"BTCUSDT": 0.01, "ETHUSDT": -0.01, "SOLUSDT": 0.005}


def _price(s: str, i: int) -> float:
    return round(100 * float(np.exp(DRIFT[s] * i)), 2)


def _row(s: str, i: int) -> list[Any]:
    p = _price(s, i)
    return [
        T0 + i * D,
        str(p),
        str(p * 1.02),
        str(p * 0.98),
        str(p),
        "1",
        T0 + (i + 1) * D - 1,
        str(p),
        10,
        "0.5",
        str(p / 2),
        "0",
    ]


def _setup(config_dir: Path) -> tuple[DataPaths, list[str]]:
    paths = DataPaths(config_dir.parent / "data")
    SnapshotStore(paths, "binance").save(
        exchange_info([binance_symbol(s, "USDT") for s in TRADE]), FALLBACK_FEES, T0, "https://x"
    )
    for s in TRADE:
        rows = [_row(s, i) for i in range(ARCHIVED)]
        bars = klines_rest.parse(rows, D, T0 + ARCHIVED * D).closed
        out = paths.lt_klines("1d", s)
        out.parent.mkdir(parents=True, exist_ok=True)
        bars.write_parquet(out)
        key = f"futures/um/monthly/fundingRate/{s}/{s}-fundingRate-2024.zip"
        path = paths.raw_archive("binance_vision", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv = "\n".join(f"{T0 + k * 8 * 3_600_000 + 1},8,0.0001" for k in range(3 * ARCHIVED))
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("f.csv", "calc_time,funding_interval_hours,last_funding_rate\n" + csv + "\n")
    assert funding.main(["--config", str(config_dir), "build"]) == 0
    return paths, ["--config", str(config_dir), "--trial", TRIAL, "--hypotheses", str(HYPOTHESES)]


def _fake_api(monkeypatch: pytest.MonkeyPatch, offset_ms: int = 20) -> list[int]:
    """API : barres 63 (close) et 64 (en cours) ; horloge : écart ``offset_ms`` ± 100 ms."""
    asked: list[int] = []

    def fetch(
        rest_url: str, symbol: str, interval: str, since_ms: int, server_ms: int, **_: Any
    ) -> klines_rest.Recent:
        asked.append(since_ms)
        return klines_rest.parse([_row(symbol, ARCHIVED), _row(symbol, ARCHIVED + 1)], D, server_ms)

    monkeypatch.setattr(klines_rest, "fetch", fetch)
    monkeypatch.setattr(paper, "measure_clock", lambda url, **_: ClockOffset(offset_ms, 100))
    monkeypatch.setattr(paper, "now_ms", lambda: NOW - offset_ms)
    return asked


def test_full_cycle(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths, args = _setup(config_dir)
    asked = _fake_api(monkeypatch)
    journal = paths.paper_journal("lt_ts_momentum_lookback_days_30")
    assert paper.main([*args, "run"]) == 1  # pas encore ouvert
    assert paper.main([*args, "start"]) == 0
    assert paper.main([*args, "start"]) == 1  # un seul start
    assert paper.main([*args, "run"]) == 0
    assert asked == [T0 + ARCHIVED * D] * 3  # rattrapage depuis la barre après les archives
    start, day = read_records(journal)
    assert (start["capital"], start["quote"], start["tier"]) == ("50.0", "USDT", "t0")
    assert day["bar_ms"] == T0 + ARCHIVED * D and day["run_ms"] == NOW
    assert day["risk"] == {"action": "trade", "reasons": []}
    assert day["target"] == {"BTCUSDT": pytest.approx(1 / 3), "SOLUSDT": pytest.approx(1 / 3)}
    buys = [(e["symbol"], e["side"], e["accepted"]) for e in day["executions"]]
    assert buys == [("BTCUSDT", "BUY", True), ("SOLUSDT", "BUY", True)]
    assert day["next_opens"] == {s: _price(s, ARCHIVED + 1) for s in TRADE}
    assert day["closes"]["BTCUSDT"] == _price("BTCUSDT", ARCHIVED)  # barre REST raccordée
    assert (day["missed_bars"], day["clock_offset_ms"], day["value_close"]) == (0, 20, "50.0")
    assert "BUY BTCUSDT" in capsys.readouterr().out
    # Deuxième passage sur la même barre : rien à faire, journal inchangé
    assert paper.main([*args, "run"]) == 0
    assert "déjà traitée" in capsys.readouterr().out and len(read_records(journal)) == 2
    assert paper.main([*args, "status"]) == 0
    out = capsys.readouterr().out
    assert "1 barre(s)" in out and "Actif" in out and "BTCUSDT" in out
    assert paper.main([*args, "resume", "--reason", "test"]) == 1  # aucun arrêt à lever


def test_clock_drift_blocks_the_decision(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Écart 800 ms, incertitude 100 : 700 > 500 ⇒ aucune décision, rien au journal."""
    paths, args = _setup(config_dir)
    _fake_api(monkeypatch, offset_ms=-800)
    assert paper.main([*args, "start"]) == 0
    assert paper.main([*args, "run"]) == 1
    assert len(read_records(paths.paper_journal("lt_ts_momentum_lookback_days_30"))) == 1


def test_changed_fiche_is_refused(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, args = _setup(config_dir)
    _fake_api(monkeypatch)
    start = {
        "kind": "start",
        "run_ms": T0,
        "trial": TRIAL,
        "fingerprint": "autre",
        "quote": "USDT",
        "tier": "t0",
        "capital": "50",
    }
    append_record(paths.paper_journal("lt_ts_momentum_lookback_days_30"), start)
    assert paper.main([*args, "run"]) == 1


def test_resume_lifts_a_latched_halt(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, args = _setup(config_dir)
    journal = paths.paper_journal("lt_ts_momentum_lookback_days_30")
    assert paper.main([*args, "start"]) == 0
    halted = {
        "kind": "day",
        "bar_ms": T0,
        "value_close": "50",
        "halted": True,
        "after": {"cash": "50", "held": {}},
    }
    append_record(journal, halted)
    assert paper.main([*args, "resume", "--reason", "analyse faite"]) == 0
    assert read_records(journal)[-1]["kind"] == "resume"
    assert paper.main([*args, "resume", "--reason", "encore"]) == 1


def test_book_name_and_missed_bars() -> None:
    assert paper.book_name(TRIAL) == "lt_ts_momentum_lookback_days_30"
    assert paper.book_name("lt_xs · lookback_days=90, top_k=3") == "lt_xs_lookback_days_90_top_k_3"
    days: list[dict[str, Any]] = [{"kind": "start"}, {"kind": "day", "bar_ms": T0}]
    assert paper._missed(days, T0 + D) == 0 and paper._missed(days, T0 + 4 * D) == 3
    assert paper._missed([{"kind": "start"}], T0) == 0


def test_extend_market_keeps_the_archive_grid(config_dir: Path) -> None:
    paths, _ = _setup(config_dir)
    archives = {s: klines.load(paths, "1d", s) for s in TRADE}
    recent = {s: klines_rest.parse([_row(s, ARCHIVED)], D, NOW) for s in TRADE}
    config = load_config(config_dir)
    snapshot = SnapshotStore(paths, "binance").latest()
    assert snapshot is not None
    market = paper.extend_market(
        strategies.load_market(paths, config, snapshot, (), "binance"), recent, archives
    )
    assert market.closes.height == ARCHIVED + 1 and market.volumes.height == ARCHIVED + 1
    assert market.closes["BTCUSDT"][-1] == _price("BTCUSDT", ARCHIVED)
    assert json.dumps(market.closes.columns) == json.dumps(["date_ms", *TRADE])
    assert pl.Series(market.closes["date_ms"]).is_sorted()
