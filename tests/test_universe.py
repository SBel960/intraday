"""Tests de qlab.longterm.universe : cas calculés à la main, marché fictif, commande."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.config import UniverseConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import universe as un

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")
CFG = UniverseConfig(
    warmup_days=2,
    reference_quotes=("USDT", "BUSD"),
    excluded_bases=("USDC",),
    leveraged_suffixes=("UP", "DOWN"),
    volume_lookback_days=3,
    min_volume_quote=3.0,
    quote_min_volume=20.0,
)
KNOWN = {
    s: {"baseAsset": b, "quoteAsset": q}
    for s, b, q in [
        ("BTCUSDT", "BTC", "USDT"),
        ("BTCBUSD", "BTC", "BUSD"),
        ("ETHBUSD", "ETH", "BUSD"),
        ("ETHUSDT", "ETH", "USDT"),
        ("ETHBTC", "ETH", "BTC"),
        ("BTCUPUSDT", "BTCUP", "USDT"),
        ("JUPUSDT", "JUP", "USDT"),
        ("USDCUSDT", "USDC", "USDT"),
        ("DEADUSDT", "DEAD", "USDT"),
        ("BTCEUR", "BTC", "EUR"),
        ("SHIBEUR", "SHIB", "EUR"),
        ("USDCEUR", "USDC", "EUR"),
    ]
}


def _bars(days: list[int], volumes: list[float] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time_ms": [T0 + d * D for d in days],
            "volume_quote": volumes or [10.0] * len(days),
        }
    )


def _days(dates: pl.Series) -> list[int]:
    return [(t - T0) // D for t in dates.to_list()]


def test_pair_of() -> None:
    assert un.pair_of("ETHBTC", KNOWN, ["USDT"]) == un.Pair("ETHBTC", "ETH", "BTC")
    assert un.pair_of("NBTUSDT", KNOWN, ["BUSD", "USDT"]) == un.Pair("NBTUSDT", "NBT", "USDT")
    assert un.pair_of("SKYBNB", KNOWN, ["USDT"]) is None
    assert un.pair_of("USDT", KNOWN, ["USDT"]) is None


def test_is_asset() -> None:
    bases = {"BTC", "BTCUP", "JUP", "USDC"}
    assert un.is_asset("BTC", bases, CFG)
    assert not un.is_asset("USDC", bases, CFG)  # stablecoin
    assert not un.is_asset("BTCUP", bases, CFG)  # BTC + UP : token à levier
    assert un.is_asset("JUP", bases, CFG)  # « J » n'est pas une base


def test_warmup_by_hand() -> None:
    """Première bougie au jour 0, chauffe 2 jours ⇒ éligible dès le jour 2."""
    assert _days(un.eligible_dates(_bars([0, 1, 2, 3]), warmup_days=2)) == [2, 3]
    assert _days(un.eligible_dates(_bars([0, 1]), warmup_days=5)) == []
    assert un.eligible_dates(_bars([]), warmup_days=0).len() == 0


def test_liquidity_by_hand() -> None:
    """Volumes 1 1 1 5 5 5 1, médiane des 3 derniers jours (jour compris) :
    1 1 1 1 5 5 5 ⇒ seuil 3 franchi aux jours 4, 5, 6."""
    bars = _bars(list(range(7)), [1, 1, 1, 5, 5, 5, 1])
    dates = un.eligible_dates(bars, warmup_days=0, lookback_days=3, min_volume=3.0)
    assert _days(dates) == [4, 5, 6]


def test_liquidity_window_is_in_days_not_rows() -> None:
    """Jour 5 absent : la fenêtre du jour 6 couvre les jours 4..6 = volumes (5, 1) ⇒ médiane 3."""
    bars = _bars([0, 1, 2, 3, 4, 6], [1, 1, 1, 5, 5, 1])
    dates = un.eligible_dates(bars, warmup_days=0, lookback_days=3, min_volume=3.0)
    assert _days(dates) == [4, 6]


def test_liquidity_never_looks_ahead() -> None:
    """Ajouter des jours futurs ne change pas l'éligibilité des jours passés."""
    short = _bars(list(range(5)), [1, 1, 1, 5, 5])
    longer = _bars(list(range(8)), [1, 1, 1, 5, 5, 99, 99, 99])

    def days(bars: pl.DataFrame) -> list[int]:
        return _days(un.eligible_dates(bars, warmup_days=0, lookback_days=3, min_volume=3.0))

    assert [d for d in days(longer) if d < 5] == days(short) == [4]


def _write(paths: DataPaths, symbol: str, bars: pl.DataFrame) -> None:
    out = paths.lt_klines("1d", symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(out)


def _market(paths: DataPaths) -> None:
    """BTC en USDT et BUSD ; ETH en BUSD (jours 0-9, retiré), puis en USDT (dès le jour 5) ;
    ETHBTC hors dollar, BTCUP à levier, USDC stablecoin, DEAD illiquide, SKYBNB inconnu."""
    _write(paths, "BTCUSDT", _bars(list(range(12))))
    _write(paths, "BTCBUSD", _bars(list(range(12))))
    _write(paths, "ETHBUSD", _bars(list(range(10))))
    _write(paths, "ETHUSDT", _bars(list(range(5, 12))))
    for s in ("ETHBTC", "BTCUPUSDT", "USDCUSDT", "SKYBNB"):
        _write(paths, s, _bars(list(range(12))))
    _write(paths, "DEADUSDT", _bars(list(range(12)), [1.0] * 12))


def test_observed_one_pair_per_asset(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    _market(paths)
    obs = un.observed(paths, KNOWN, CFG)
    assert (obs.candidates, obs.unknown) == (5, 1)  # BTC×2, ETH×2, DEAD ; SKYBNB inconnu
    m = obs.members
    btc = m.filter(pl.col("base") == "BTC")
    assert set(btc["symbol"]) == {"BTCUSDT"} and _days(btc["date_ms"]) == list(range(2, 12))
    eth_rows = m.filter(pl.col("base") == "ETH")
    eth = dict(zip(_days(eth_rows["date_ms"]), eth_rows["symbol"], strict=True))
    # ETHUSDT listé au jour 5, chauffe 2 ⇒ prioritaire dès le jour 7 ; avant : ETHBUSD
    assert eth == {d: "ETHBUSD" for d in range(2, 7)} | {d: "ETHUSDT" for d in range(7, 12)}
    assert set(m["base"]) == {"BTC", "ETH"}


def test_traded_uses_warmup_only(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    _write(paths, "BTCEUR", _bars(list(range(4)), [0.0] * 4))  # volume nul : pas de filtre
    _write(paths, "SOLEUR", _bars([3]))
    t = un.traded(paths, ["BTCEUR", "SOLEUR"], warmup_days=2)
    assert t.columns == ["date_ms", "symbol"]
    assert [(_days(pl.Series([d]))[0], s) for d, s in t.rows()] == [(2, "BTCEUR"), (3, "BTCEUR")]
    with pytest.raises(DataError, match="absentes"):
        un.traded(paths, ["ETHEUR"], warmup_days=2)


def test_cli(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = DataPaths(config_dir.parent / "data")
    assert un.main(["--config", str(config_dir), "show"]) == 1  # pas encore de snapshot
    SnapshotStore(paths, "binance").save(
        exchange_info([binance_symbol(s, "USDT") for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]),
        FALLBACK_FEES,
        T0,
        "https://x",
    )
    volume = [2e6] * 40
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        _write(paths, s, _bars(list(range(40)), volume))
    assert un.main(["--config", str(config_dir), "show", "--date", "2024-02-05"]) == 0
    out = capsys.readouterr().out
    assert "Observé : 3 paires en dollar candidates, 0 inconnues" in out
    assert "2024 : actifs par jour min 3, médiane 3, max 3" in out
    assert "Tradé BTCUSDT : éligible dès le 2024-01-31" in out
    assert "2024-02-05 : 3 actifs : BTCUSDT ETHUSDT SOLUSDT" in out


def test_quoted_universe_uses_its_own_volume_threshold(tmp_path: Path) -> None:
    """Paires EUR : BTCEUR à 30 €/jour (≥ 20) éligible dès la fin de la chauffe ; SHIBEUR à
    10 €/jour (< 20) jamais ; USDCEUR : stablecoin exclu ; paires USDT ignorées."""
    paths = DataPaths(tmp_path)
    _market(paths)
    _write(paths, "BTCEUR", _bars(list(range(6)), [30.0] * 6))
    _write(paths, "SHIBEUR", _bars(list(range(6)), [10.0] * 6))
    _write(paths, "USDCEUR", _bars(list(range(6)), [99.0] * 6))
    eur = un.quoted(paths, KNOWN, CFG, "EUR")
    assert eur.candidates == 2  # BTCEUR, SHIBEUR (USDCEUR exclu)
    assert set(eur.members["symbol"]) == {"BTCEUR"}
    assert _days(eur.members["date_ms"]) == [2, 3, 4, 5]
