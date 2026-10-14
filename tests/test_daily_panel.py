"""Tests de qlab.markets.daily_panel : jours ouvrés, pointes, fériés, éligibilité — à la main."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fakes import markets_series

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.markets import daily_panel as dp

MON = date_to_ms("2024-01-01")  # lundi


def _day(i: int) -> int:
    return MON + i * MS_PER_DAY


def test_business_bars_are_consecutive_over_weekends() -> None:
    """1970-01-01 jeudi ⇒ barre 3 ; lundi 05 ⇒ 5 ; vendredi 09 ⇒ 9 ; lundi 12 ⇒ 10."""
    t = pl.DataFrame({"t_ms": [0, 4 * MS_PER_DAY, 8 * MS_PER_DAY, 11 * MS_PER_DAY]})
    got = t.select(dp.business_bar(pl.col("t_ms")), dp.weekday(pl.col("t_ms")).alias("w"))
    assert got["t_ms"].to_list() == [3, 5, 9, 10]
    assert got["w"].to_list() == [3, 0, 4, 0]


def test_grid_drops_weekends_and_marks_missing_days() -> None:
    """Lundi, mardi, jeudi, samedi (crypto) : mercredi devient un trou, samedi disparaît."""
    bars = pl.DataFrame(
        {"symbol": ["X"] * 4, "t_ms": [_day(0), _day(1), _day(3), _day(5)], "close": [1.0, 2, 3, 4]}
    )
    grid = dp.complete_grid(bars)
    assert grid["t_ms"].to_list() == [_day(i) for i in range(4)]
    assert grid["close"].to_list() == [1.0, 2.0, None, 3.0]
    assert grid["bar"].diff().drop_nulls().to_list() == [1, 1, 1]


def test_round_trip_spike_becomes_a_hole() -> None:
    """100 → 150 → 100 : ±40 % deux jours de suite, en sens opposés ⇒ erreur de cotation.
    100 → 150 → 150 : vrai mouvement, gardé."""
    panel = pl.DataFrame(
        {
            "symbol": ["A"] * 4 + ["B"] * 4,
            "open": [100.0, 150, 100, 101, 100, 150, 150, 151],
            "high": [100.0, 150, 100, 101, 100, 150, 150, 151],
            "low": [100.0, 150, 100, 101, 100, 150, 150, 151],
            "close": [100.0, 150, 100, 101, 100, 150, 150, 151],
            "volume_quote": [1.0] * 8,
        }
    )
    got = dp.remove_spikes(panel, 0.25)
    assert got["close"].to_list() == [100.0, None, 100, 101, 100, 150, 150, 151]
    assert got["volume_quote"].null_count() == 1


def test_holidays_carry_the_close_without_entry(config_dir: Path) -> None:
    """Mercredi férié : clôture de mardi reportée, plus haut = plus bas = clôture, pas
    d'ouverture (aucune entrée), ``traded`` faux. Éligible après 5 séances réelles (config
    de test) et un volume médian des 3 séances précédentes ≥ 1 000 $."""
    cfg = load_config(config_dir).markets
    days = [0, 1, 3, 4, 7, 8, 9, 10]  # mercredi 3 janvier manquant
    bars = pl.DataFrame(
        {
            "symbol": ["X"] * len(days),
            "t_ms": [_day(i) for i in days],
            "open": [10.0 + i for i in range(len(days))],
            "high": [11.0 + i for i in range(len(days))],
            "low": [9.0 + i for i in range(len(days))],
            "close": [10.0 + i for i in range(len(days))],
            "volume": [200.0] * len(days),  # 200 × ~10 $ ≥ 1 000 $
        }
    )
    p = dp.assemble(bars, cfg)
    wed = p.row(2, named=True)
    assert (wed["open"], wed["close"], wed["high"], wed["low"], wed["traded"]) == (
        None,
        11.0,
        11.0,
        11.0,
        False,
    )
    assert p["eligible"].to_list() == [False] * 6 + [True] * 3  # 6e séance réelle : mardi 9 janvier
    assert p["funding"].to_list() == [0.0] * 9


def test_load_reads_every_market_or_says_what_is_missing(config_dir: Path) -> None:
    cfg = load_config(config_dir).markets
    paths = DataPaths(config_dir.parent / "data")
    with pytest.raises(DataError, match="absentes"):
        dp.load(paths, cfg)
    markets_series(paths)
    p = dp.load(paths, cfg)
    assert sorted(p["symbol"].unique()) == ["AAA", "BBB", "BTC-USD", "CCC"]
    assert p.filter(dp.weekday(pl.col("t_ms")) >= dp.WEEKDAYS).is_empty()
