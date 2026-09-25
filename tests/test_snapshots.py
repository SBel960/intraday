"""Tests de qlab.exchange.snapshots : versions, point-in-time, intégrité, différences."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import zstandard

from qlab.core.errors import DataError, ExchangeError
from qlab.core.paths import DataPaths
from qlab.exchange.snapshots import (
    SnapshotStore,
    content_hash,
    diff_snapshots,
    validate_exchange_info,
)

T = 1_704_067_200_000  # 2024-01-01T00:00:00Z
DAY = 86_400_000
URL = "https://api.binance.com/api/v3/exchangeInfo"
FEES = {
    "origin": "config",
    "maker_frac": 0.001,
    "taker_frac": 0.001,
    "bnb_discount_frac": 0.25,
    "pay_in_bnb": False,
    "effective_maker_frac": 0.001,
    "effective_taker_frac": 0.001,
}


def filters(tick: str = "0.01", step: str = "0.00001", min_notional: str = "5") -> list[Any]:
    return [
        {"filterType": "PRICE_FILTER", "minPrice": tick, "maxPrice": "1000000", "tickSize": tick},
        {"filterType": "LOT_SIZE", "minQty": step, "maxQty": "9000", "stepSize": step},
        {
            "filterType": "NOTIONAL",
            "minNotional": min_notional,
            "applyMinToMarket": True,
            "maxNotional": "9000000",
            "applyMaxToMarket": False,
        },
    ]


def info(server_time: int = T) -> dict[str, Any]:
    def sym(name: str, base: str, quote: str, status: str = "TRADING") -> dict[str, Any]:
        return {
            "symbol": name,
            "status": status,
            "baseAsset": base,
            "quoteAsset": quote,
            "filters": filters(),
        }

    return {
        "timezone": "UTC",
        "serverTime": server_time,
        "rateLimits": [],
        "exchangeFilters": [],
        "symbols": [
            sym("BTCEUR", "BTC", "EUR"),
            sym("ETHEUR", "ETH", "EUR"),
            sym("BTCUSDT", "BTC", "USDT"),
            sym("OLDEUR", "OLD", "EUR", status="BREAK"),
        ],
    }


def store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(DataPaths(tmp_path), "binance")


# --- validation et hash ------------------------------------------------------------------


def test_validate_rejects_bad_symbols() -> None:
    bad = info()
    del bad["symbols"][1]["filters"]
    with pytest.raises(ExchangeError, match="incomplète"):
        validate_exchange_info(bad)
    bad = info()
    bad["symbols"].append(copy.deepcopy(bad["symbols"][0]))
    with pytest.raises(ExchangeError, match="en double"):
        validate_exchange_info(bad)
    with pytest.raises(ExchangeError, match="aucune paire"):
        validate_exchange_info({"symbols": []})
    with pytest.raises(ExchangeError, match="absente"):
        validate_exchange_info({"x": 1})


def test_hash_ignores_server_time_only() -> None:
    h = content_hash(info(T), FEES)
    assert content_hash(info(T + 60_000), FEES) == h  # serverTime seul : même contenu
    changed = info()
    changed["symbols"][0]["filters"] = filters(min_notional="10")
    assert content_hash(changed, FEES) != h
    assert content_hash(info(), {**FEES, "taker_frac": 0.00075}) != h  # frais : nouvelle version


# --- versions ----------------------------------------------------------------------------


def test_save_versions_only_on_change(tmp_path: Path) -> None:
    s = store(tmp_path)
    r1 = s.save(info(T), FEES, T + 123, URL)
    r2 = s.save(info(T + 60_000), FEES, T + 60_000, URL)  # serverTime seul change
    changed = info()
    changed["symbols"][0]["filters"] = filters(min_notional="10")
    r3 = s.save(changed, FEES, T + 120_000, URL)
    assert (r1.written, r2.written, r3.written) == (True, False, True)
    assert r1.previous is None  # toute première version
    assert r2.previous is None and r2.snapshot.path == r1.snapshot.path
    assert r3.previous is not None and r3.previous.path == r1.snapshot.path
    assert [p.name for p in s.paths()] == ["20240101T000000Z.json.zst", "20240101T000200Z.json.zst"]
    assert r1.snapshot.fetched_ms == T  # arrondi à la seconde = heure du nom de fichier
    assert not list(s.dir.glob(".tmp-*"))  # écriture atomique : pas de reste
    assert s.dir == tmp_path / "meta" / "exchange_info" / "binance"


def test_round_trip_and_compression(tmp_path: Path) -> None:
    s = store(tmp_path)
    saved = s.save(info(), FEES, T, URL).snapshot
    assert s.load(saved.path) == saved
    doc = json.loads(zstandard.ZstdDecompressor().decompress(saved.path.read_bytes()))
    assert doc["url"] == URL and doc["source"] == "binance"


def test_filters_and_symbols(tmp_path: Path) -> None:
    snap = store(tmp_path).save(info(), FEES, T, URL).snapshot
    f = snap.filters("BTCEUR")
    assert (f.tick_size, f.step_size, f.min_notional) == (
        Decimal("0.01"),
        Decimal("0.00001"),
        Decimal("5"),
    )
    assert [x["symbol"] for x in snap.symbols()] == ["BTCEUR", "ETHEUR", "BTCUSDT"]
    assert len(snap.symbols(None)) == 4  # retirées comprises
    with pytest.raises(DataError, match="absent du snapshot"):
        snap.filters("NOPE")


def test_snapshot_at_point_in_time(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.save(info(), FEES, T, URL)
    changed = info()
    changed["symbols"][0]["filters"] = filters(min_notional="10")
    s.save(changed, FEES, T + DAY, URL)
    before, extrapolated = s.snapshot_at(T - 1)  # backtest avant le 1er snapshot
    assert extrapolated and before.fetched_ms == T
    first, extrapolated = s.snapshot_at(T)  # borne incluse
    assert not extrapolated and first.filters("BTCEUR").min_notional == 5
    assert s.snapshot_at(T + DAY - 1)[0].fetched_ms == T
    last, extrapolated = s.snapshot_at(T + 10 * DAY)
    assert not extrapolated and last.filters("BTCEUR").min_notional == 10


# --- différences -------------------------------------------------------------------------


def test_diff_by_hand(tmp_path: Path) -> None:
    """Nouvelle version : +SOLEUR, BTCUSDT disparue, ETHEUR retirée, filtres BTCEUR, frais."""
    s = store(tmp_path)
    old = s.save(info(), FEES, T, URL).snapshot
    new_info = info()
    new_info["symbols"] = [x for x in new_info["symbols"] if x["symbol"] != "BTCUSDT"]
    new_info["symbols"][0]["filters"] = filters(min_notional="10")
    new_info["symbols"][1]["status"] = "BREAK"
    new_info["symbols"].append(
        {
            "symbol": "SOLEUR",
            "status": "TRADING",
            "baseAsset": "SOL",
            "quoteAsset": "EUR",
            "filters": filters(),
        }
    )
    new = s.save(new_info, {**FEES, "taker_frac": 0.00075}, T + DAY, URL).snapshot
    d = diff_snapshots(old, new)
    assert d.added == ("SOLEUR",)
    assert d.removed == ("BTCUSDT",)
    assert d.status_changed == (("ETHEUR", "TRADING", "BREAK"),)
    assert d.filters_changed == ("BTCEUR",)
    assert d.fees_changed and not d.is_empty
    assert d.touching(("BTCEUR", "ETHEUR", "SOLEUR", "XRPEUR")) == ("BTCEUR", "ETHEUR", "SOLEUR")


def test_diff_identical_is_empty(tmp_path: Path) -> None:
    snap = store(tmp_path).save(info(), FEES, T, URL).snapshot
    d = diff_snapshots(snap, snap)
    assert d.is_empty and d.touching(("BTCEUR",)) == ()


# --- cas limites et erreurs --------------------------------------------------------------


def test_empty_store(tmp_path: Path) -> None:
    s = store(tmp_path)
    assert s.paths() == [] and s.latest() is None
    with pytest.raises(DataError, match="aucun snapshot"):
        s.snapshot_at(T)


def test_foreign_files_ignored(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.save(info(), FEES, T, URL)
    (s.dir / "notes.txt").write_text("x")
    (s.dir / ".tmp-abc").write_bytes(b"partiel")  # reste d'un crash
    assert len(s.paths()) == 1


@pytest.mark.parametrize("fetched", [T - 1000, T + 999])  # plus ancien ; même seconde
def test_not_newer_rejected(tmp_path: Path, fetched: int) -> None:
    s = store(tmp_path)
    s.save(info(), FEES, T, URL)
    changed = info()
    changed["symbols"][0]["status"] = "BREAK"
    with pytest.raises(DataError, match="pas plus récent"):
        s.save(changed, FEES, fetched, URL)


def test_tampered_snapshot_detected(tmp_path: Path) -> None:
    s = store(tmp_path)
    snap = s.save(info(), FEES, T, URL).snapshot
    doc = json.loads(zstandard.ZstdDecompressor().decompress(snap.path.read_bytes()))
    doc["exchange_info"]["symbols"][0]["filters"][2]["minNotional"] = "0.01"  # falsifié
    snap.path.write_bytes(zstandard.ZstdCompressor().compress(json.dumps(doc).encode()))
    with pytest.raises(DataError, match="altéré"):
        s.load(snap.path)


def test_renamed_snapshot_detected(tmp_path: Path) -> None:
    s = store(tmp_path)
    snap = s.save(info(), FEES, T, URL).snapshot
    snap.path.rename(s.dir / "20250101T000000Z.json.zst")  # date du nom ≠ date du contenu
    with pytest.raises(DataError, match="incohérents"):
        s.latest()


def test_corrupt_snapshot(tmp_path: Path) -> None:
    s = store(tmp_path)
    snap = s.save(info(), FEES, T, URL).snapshot
    snap.path.write_bytes(b"pas du zstd")
    with pytest.raises(DataError, match="illisible"):
        s.latest()
