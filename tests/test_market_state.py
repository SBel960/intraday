"""Tests de qlab.longterm.market_state : indicateurs à la main, agrégation, commande."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import market_state as ms

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")


def _bars(days: list[int], closes: list[float], volume: float = 10.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time_ms": [T0 + d * D for d in days],
            "close": [float(c) for c in closes],
            "volume_quote": [volume] * len(days),
        }
    )


def test_pair_indicators_by_hand() -> None:
    """Clôtures 1 2 3 2 1, moyenne 3 jours : nulle aux jours 0-1 (moins de 3 jours), puis
    3 > 2 ; 2 < 7/3 ; 1 < 2 ⇒ [–, –, vrai, faux, faux]. Rendement du jour 1 : ln 2."""
    ind = ms.pair_indicators(_bars([0, 1, 2, 3, 4], [1, 2, 3, 2, 1]), [3])
    assert ind["above_ma_3"].to_list() == [None, None, True, False, False]
    assert ind["ret_1d"][0] is None
    assert ind["ret_1d"][1] == pytest.approx(math.log(2))


def test_gap_blocks_average_and_return() -> None:
    """Jour 3 absent : fenêtres des jours 4 et 5 incomplètes, rendement du jour 4 absent."""
    ind = ms.pair_indicators(_bars([0, 1, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6]), [3])
    assert ind["above_ma_3"].to_list() == [None, None, True, None, None, True]
    assert ind["ret_1d"].is_null().to_list() == [True, False, False, True, False, False]


def _write(paths: DataPaths, symbol: str, bars: pl.DataFrame) -> None:
    out = paths.lt_klines("1d", symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(out)


def test_market_state_aggregates_members_only(tmp_path: Path) -> None:
    """Jour 2 : BTC (1 → 2 → 4, volume 30) et SOL (1 → 1 → 1,5) au-dessus de leur moyenne 2 j,
    ETH (4 → 2) en dessous ⇒ largeur 2/3. Rendements ln 2, ln 1,5, ln 0,5 ⇒ dispersion =
    leur écart-type. Part de BTC : 30 / (30 + 10 + 10). DOGE en hausse mais hors univers."""
    paths = DataPaths(tmp_path)
    _write(paths, "BTCUSDT", _bars([0, 1, 2], [1, 2, 4], volume=30.0))
    _write(paths, "SOLUSDT", _bars([0, 1, 2], [1, 1, 1.5]))
    _write(paths, "ETHUSDT", _bars([0, 1, 2], [4, 4, 2]))
    _write(paths, "DOGEUSDT", _bars([0, 1, 2], [1, 5, 9]))
    members = pl.DataFrame(
        {
            "date_ms": [T0 + 2 * D] * 3,
            "base": ["BTC", "SOL", "ETH"],
            "symbol": ["BTCUSDT", "SOLUSDT", "ETHUSDT"],
        }
    )
    state = ms.market_state(paths, members, [2])
    assert state.columns == [
        "date_ms",
        "n_assets",
        "breadth_2",
        "dispersion",
        "btc_volume_share",
    ]
    row = state.row(0, named=True)
    assert row["n_assets"] == 3 and row["breadth_2"] == pytest.approx(2 / 3)
    rets = pl.Series([math.log(2), math.log(1.5), math.log(0.5)])
    assert row["dispersion"] == pytest.approx(rets.std())
    assert row["btc_volume_share"] == pytest.approx(0.6)


@pytest.mark.parametrize("ma_days", [[], [0]])
def test_bad_lengths_rejected(ma_days: list[int]) -> None:
    with pytest.raises(DataError, match="ma_days"):
        ms.pair_indicators(_bars([0], [1.0]), ma_days)


def test_empty_universe_rejected(tmp_path: Path) -> None:
    empty = pl.DataFrame(schema={"date_ms": pl.Int64, "base": pl.String, "symbol": pl.String})
    with pytest.raises(DataError, match="univers vide"):
        ms.market_state(DataPaths(tmp_path), empty, [50])


def test_cli(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = DataPaths(config_dir.parent / "data")
    assert ms.main(["--config", str(config_dir), "show"]) == 1  # pas encore de snapshot
    SnapshotStore(paths, "binance").save(
        exchange_info([binance_symbol(s, "USDT") for s in ("BTCUSDT", "ETHUSDT")]),
        FALLBACK_FEES,
        T0,
        "https://x",
    )
    for s, step in (("BTCUSDT", 1.01), ("ETHUSDT", 0.99)):
        closes = [100 * step**d for d in range(60)]
        _write(paths, s, _bars(list(range(60)), closes, volume=2e6))
    assert ms.main(["--config", str(config_dir), "show", "--ma-days", "10"]) == 0
    out = capsys.readouterr().out
    assert "(30 jours)" in out  # jours 30 à 59 : après la chauffe de 30 jours
    assert "breadth_10" in out and "0.5" in out  # BTC monte, ETH baisse
