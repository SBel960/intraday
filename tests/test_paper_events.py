"""Tests de qlab.live.paper_events : signaux, refus motivés, clôtures calculées à la main."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qlab.core.config import load_config
from qlab.core.paths import DataPaths
from qlab.core.records import append_record
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.live import paper_events as pe

H = MS_PER_HOUR
T = date_to_ms("2026-10-15") + 10 * 60_000  # annonce reçue à 00:10 ⇒ entrée à 01:00


class FakeMarket:
    def __init__(self, bars: dict[str, list[list[Any]]], volumes: dict[str, float]) -> None:
        self.bars, self.volumes = bars, volumes

    def volume_24h(self, symbol: str) -> float | None:
        return self.volumes.get(symbol)

    def hour_bars(self, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
        return [b for b in self.bars.get(symbol, []) if start_ms <= b[0] <= end_ms]

    def funding(self, symbol: str, start_ms: int, end_ms: int) -> float:
        return 0.001


def _news(title: str, catalog: str, received: int | None, sid: int) -> dict[str, Any]:
    return {
        "source_id": sid,
        "title": title,
        "catalog": catalog,
        "published_ms": T - 1000,
        "received_ms": received,
        "backfilled": received is None,
    }


def _bars(prices: list[tuple[float, float, float]]) -> list[list[Any]]:
    """(ouverture, plus haut, clôture) par heure à partir de 01:00."""
    entry = date_to_ms("2026-10-15") + H
    return [[entry + i * H, o, h, 0.0, c] for i, (o, h, c) in enumerate(prices)]


@pytest.fixture
def cfg(config_dir: Path) -> Any:
    return load_config(config_dir).events.paper  # vente 3 h sur retrait, 2 lignes, 0,2 %/jambe


def test_signals_and_reasons(cfg: Any) -> None:
    """AA liquide ⇒ en attente ; BB sans contrat ; CC illiquide ; DD : lignes pleines (2) ;
    annonce rechargée de l'historique (pas de réception) : ignorée."""
    news = [
        _news("Binance Will Delist AA, BB, CC on 2026-11-01", "delisting", T, 1),
        _news("Binance Lists EE", "listing", T, 2),
        _news("Binance Lists DD", "listing", T + 1, 3),
        _news("Binance Will Delist OLD on 2026-11-01", "delisting", None, 4),
    ]
    market = FakeMarket({}, {"AAUSDT": 5e6, "CCUSDT": 1e5, "EEUSDT": 2e6, "DDUSDT": 9e6})
    out = pe.step(cfg, news, [], market, T)
    got = {r["symbol"]: (r["decision"], r["reason"]) for r in out}
    assert got["AAUSDT"] == ("pending", "")
    assert got["BBUSDT"] == ("skipped", "aucun contrat perpétuel")
    assert got["CCUSDT"][1].startswith("illiquide")
    assert got["EEUSDT"] == ("pending", "")
    assert got["DDUSDT"] == ("skipped", "lignes pleines")
    assert "OLDUSDT" not in got
    aa = next(r for r in out if r["symbol"] == "AAUSDT")
    assert (aa["entry_ms"], aa["exit_ms"], aa["side"]) == (
        T - 10 * 60_000 + H,
        T - 10 * 60_000 + 4 * H,
        -1,
    )
    assert [r["id"] for r in out] == [1, 2, 3, 4, 5]
    assert pe.step(cfg, news, out, market, T) == []  # rien en double


def test_short_closed_by_hand(cfg: Any) -> None:
    """Vente à 10, sortie à l'ouverture de 04:00 : 8 ⇒ +20 % ; financement 0,1 % reçu par le
    vendeur ⇒ +20,1 % ; coût 2 × 0,2 % ⇒ +19,7 % ; mise 50 / 2 = 25 ⇒ +4,925."""
    news = [_news("Binance Will Delist AA on 2026-11-01", "delisting", T, 1)]
    market = FakeMarket(
        {"AAUSDT": _bars([(10, 11, 9), (9, 9.5, 8.5), (8.5, 9, 8), (8, 8, 8)])}, {"AAUSDT": 5e6}
    )
    journal = pe.step(cfg, news, [], market, T)
    assert pe.step(cfg, news, journal, market, T + 3 * H) == []  # pas encore échue
    close = pe.step(cfg, news, journal, market, T + 4 * H)[0]
    assert close["return_frac"] == pytest.approx(0.197)
    assert close["net_quote"] == pytest.approx(25 * 0.197)
    assert not close["liquidated"]


def test_liquidation_and_retired_contract(cfg: Any) -> None:
    """Plus haut 21 ≥ 2 × 10 : liquidé (−100 % − coûts). Contrat retiré avant la sortie :
    sortie à la dernière clôture connue."""
    news = [_news("Binance Will Delist AA on 2026-11-01", "delisting", T, 1)]
    squeeze = FakeMarket(
        {"AAUSDT": _bars([(10, 21, 12), (12, 12, 12), (12, 12, 12), (12, 12, 12)])}, {"AAUSDT": 5e6}
    )
    journal = pe.step(cfg, news, [], squeeze, T)
    close = pe.step(cfg, news, journal, squeeze, T + 4 * H)[0]
    assert close["liquidated"] and close["return_frac"] == pytest.approx(-1.004)
    retired = FakeMarket({"AAUSDT": _bars([(10, 10, 9), (9, 9, 6)])}, {"AAUSDT": 5e6})
    close = pe.step(cfg, news, pe.step(cfg, news, [], retired, T), retired, T + 4 * H)[0]
    assert close["exit_price"] == 6


def test_status_prints_trades(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = DataPaths(config_dir.parent / "data")
    path = paths.paper_journal(pe.JOURNAL)
    append_record(
        path,
        {
            "type": "signal",
            "id": 1,
            "kind": "delisting",
            "base": "AA",
            "symbol": "AAUSDT",
            "received_ms": T,
            "decision": "pending",
            "reason": "",
        },
    )
    append_record(path, {"type": "close", "id": 1, "return_frac": 0.1, "net_quote": 2.5})
    assert pe.main(["--config", str(config_dir), "status"]) == 0
    out = capsys.readouterr().out
    assert "AAUSDT" in out and "+10.0%" in out and "1 trade(s) clos, résultat +2.50 USDT" in out
