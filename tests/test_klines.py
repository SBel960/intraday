"""Tests de qlab.longterm.klines : CSV construits à la main, unités, contrôle qualité, trous."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest
from fakes import kline_csv, kline_row

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.longterm import klines as kl

D = MS_PER_DAY
T0 = date_to_ms("2024-12-30")  # deux jours en ms, puis 2025 en µs


def test_parse_ms_and_us_to_ms() -> None:
    ms = kl.parse_csv(kline_csv(kline_row(T0), kline_row(T0 + D)))
    us = kl.parse_csv(kline_csv(kline_row(T0, us=True), kline_row(T0 + D, us=True)))
    assert ms.equals(us)
    assert ms.columns == [*kl.COLUMNS, "close_time_ms"]
    assert ms["open_time_ms"].to_list() == [T0, T0 + D]
    assert ms["close_time_ms"].to_list() == [T0 + D - 1, T0 + 2 * D - 1]


def test_parse_header_and_empty() -> None:
    header = "open_time,open,high,low,close,volume,close_time,qv,count,tbb,tbq,ignore"
    assert kl.parse_csv(kline_csv(header, kline_row(T0))).height == 1
    assert kl.parse_csv(b"").height == 0
    assert kl.parse_csv(b"").columns == [*kl.COLUMNS, "close_time_ms"]


def test_mixed_units_are_converted_row_by_row() -> None:
    """Archive réelle KLAYBTC 2024-10 : dernière ligne en µs, les autres en ms."""
    mixed = kl.parse_csv(kline_csv(kline_row(T0), kline_row(T0 + D, us=True)))
    assert mixed["open_time_ms"].to_list() == [T0, T0 + D]
    assert mixed["close_time_ms"].to_list() == [T0 + D - 1, T0 + 2 * D - 1]


@pytest.mark.parametrize("t", [5, 4_200_000_000_000, 4_200_000_000_000_000])
def test_implausible_timestamps_rejected(t: int) -> None:
    """1970, an 2103 en ms, an 2103 en µs."""
    with pytest.raises(DataError, match="hors 2000–2100"):
        kl.parse_csv(kline_csv(f"{t},1,1,1,1,1,{t + 1},1,1,1,1,0"))


def test_quality_control_drops_and_counts_bad_bars() -> None:
    """5 bougies fautives sur 7 : décalée, clôture ≠ ouverture + 1 j − 1, high < close,
    low > open, volume négatif."""
    shifted = kline_row(T0 + 1000)  # ouverture pas sur une frontière de jour
    wrong_close_time = kline_row(T0 + 2 * D).replace(f",{T0 + 3 * D - 1},", f",{T0 + 3 * D},")
    frame = kl.parse_csv(
        kline_csv(
            kline_row(T0),
            kline_row(T0 + D),
            shifted,
            wrong_close_time,
            kline_row(T0 + 3 * D, close=120.0),  # high 110 < close 120
            kline_row(T0 + 4 * D, low=101.0),
            kline_row(T0 + 5 * D, vol=-1.0),
        )
    )
    merged = kl.merge([frame], D)
    assert merged.bars["open_time_ms"].to_list() == [T0, T0 + D]
    assert (merged.anomalies, merged.conflicts) == (5, 0)


def test_truncated_bar_is_kept_and_flagged() -> None:
    """Dernière bougie avant un retrait : clôture à 03:00 au lieu de 23:59:59,999."""
    last = kline_row(T0 + D).replace(f",{T0 + 2 * D - 1},", f",{T0 + D + 3 * 3_600_000 - 1},")
    merged = kl.merge([kl.parse_csv(kline_csv(kline_row(T0), last))], D)
    assert merged.bars["partial"].to_list() == [False, True]
    assert merged.anomalies == 0


def test_duplicates_and_monthly_priority() -> None:
    """Même bougie dans la mensuelle et la quotidienne : un seul exemplaire ; valeurs
    différentes : la mensuelle (premier cadre) fait foi, un conflit compté."""
    monthly = kl.parse_csv(kline_csv(kline_row(T0), kline_row(T0 + D, close=101.0)))
    daily = kl.parse_csv(
        kline_csv(kline_row(T0), kline_row(T0 + D, close=105.0), kline_row(T0 + 2 * D))
    )
    merged = kl.merge([monthly, daily], D)
    assert merged.bars["open_time_ms"].to_list() == [T0, T0 + D, T0 + 2 * D]
    assert merged.bars["close"].to_list() == [100.0, 101.0, 100.0]
    assert (merged.anomalies, merged.conflicts) == (0, 1)
    assert kl.merge([], D).bars.columns == list(kl.OUTPUT)


def test_gaps_by_hand() -> None:
    """Jours 0 1 2 5 6 9 : trous de 2 bougies à partir des jours 3 et 7."""
    times = pl.Series([T0 + d * D for d in (0, 1, 2, 5, 6, 9)])
    g = kl.gaps(times, D)
    assert g["start_ms"].to_list() == [T0 + 3 * D, T0 + 7 * D]
    assert g["missing"].to_list() == [2, 2]
    assert kl.gaps(pl.Series([T0, T0 + D]), D).height == 0


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="absentes"):
        kl.load(DataPaths(tmp_path), "1d", "BTCEUR")
    with pytest.raises(DataError, match="non géré"):
        kl.interval_to_ms("4h")
    bad = tmp_path / "two.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("a.csv", kline_row(T0))
        z.writestr("b.csv", kline_row(T0))
    with pytest.raises(DataError, match="un seul fichier"):
        kl.read_archive(bad)
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(broken, "w") as z:
        z.writestr("x.csv", kline_row(T0).replace(",7,", ",sept,"))
    with pytest.raises(DataError, match=r"broken\.zip : CSV illisible"):
        kl.read_archive(broken)
