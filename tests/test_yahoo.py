"""Tests de qlab.data.yahoo : lecture de la réponse, ajustement, erreurs, commande (sans réseau)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from qlab.core import http
from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import date_to_ms
from qlab.data import yahoo

DAY0 = date_to_ms("2024-01-02")


def _blob(closes: list[float | None], adj: list[float | None]) -> bytes:
    n = len(closes)
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        # 14:30 UTC : ouverture de New York, ramenée à minuit UTC
                        "timestamp": [(DAY0 + i * 86_400_000) // 1000 + 52_200 for i in range(n)],
                        "indicators": {
                            "quote": [
                                {
                                    "open": closes,
                                    "high": closes,
                                    "low": closes,
                                    "close": closes,
                                    "volume": [10] * n,
                                }
                            ],
                            "adjclose": [{"adjclose": adj}],
                        },
                    }
                ],
                "error": None,
            }
        }
    ).encode()


def test_parse_adjusts_prices_and_drops_empty_days() -> None:
    """Facteur adjclose / close = ½ le 1er jour (dividende ensuite), 1 après ; jour 2 sans
    clôture : retiré."""
    got = yahoo.parse(_blob([100.0, None, 110.0], [50.0, None, 110.0]), "SPY")
    assert got["t_ms"].to_list() == [DAY0, DAY0 + 2 * 86_400_000]
    assert got["close"].to_list() == pytest.approx([50.0, 110.0])
    assert got["open"].to_list() == pytest.approx([50.0, 110.0])
    assert got.columns == list(yahoo.COLUMNS)


def test_refused_or_malformed_answers() -> None:
    refused = json.dumps({"chart": {"result": None, "error": {"code": "Not Found"}}}).encode()
    with pytest.raises(DataError, match="refusée"):
        yahoo.parse(refused, "ZZZ")
    with pytest.raises(DataError, match="sans série"):
        yahoo.parse(json.dumps({"chart": {"result": [], "error": None}}).encode(), "ZZZ")
    broken = json.dumps({"chart": {"result": [{"timestamp": [1]}], "error": None}}).encode()
    with pytest.raises(DataError, match="forme"):
        yahoo.parse(broken, "ZZZ")


def test_url(config_dir: Path) -> None:
    cfg = load_config(config_dir).markets
    url = yahoo.chart_url(cfg, "BTC-USD", date_to_ms("2024-01-01"))
    assert url.startswith("https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?")
    assert f"period1={date_to_ms('2020-01-01') // 1000}" in url and "interval=1d" in url


def test_fetch_command_writes_one_file_per_market(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_get(url: str, **_: object) -> http.HttpResult:
        seen.append(url)
        return http.HttpResult(_blob([1.0, 2.0], [1.0, 2.0]), {}, 0, 0, 1)

    monkeypatch.setattr(http, "get", fake_get)
    assert yahoo.main(["--config", str(config_dir), "fetch", "--only", "AAA,BBB"]) == 0
    paths = DataPaths(config_dir.parent / "data")
    assert len(seen) == 2
    assert pl.read_parquet(paths.markets_daily("AAA"))["close"].to_list() == [1.0, 2.0]
    assert not paths.markets_daily("CCC").exists()
