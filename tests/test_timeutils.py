"""Tests de qlab.core.timeutils : valeurs calculées à la main, cas limites, cas d'erreur.

Repères (calculés à la main) :
- 2024-01-01T00:00:00.000Z = 1 704 067 200 000 ms, un lundi.
- 2024-01-01T13:47:12.345Z = 1 704 067 200 000 + 13·3 600 000 + 47·60 000 + 12 345
  = 1 704 116 832 345 ms ; minute du jour 13·60 + 47 = 827 ⇒ tranche 5 min n° 165.
- 2024-02-29T23:59:59.999Z = 1 704 067 200 000 + 59·86 400 000 + 86 399 999
  = 1 709 251 199 999 ms, un jeudi (année bissextile).
- −1 ms = 1969-12-31T23:59:59.999Z, un mercredi, dernière tranche de la journée.
"""

from __future__ import annotations

import time

import polars as pl
import pytest

from qlab.core import timeutils as tu
from qlab.core.errors import DataError

T_2024 = 1_704_067_200_000
T_AFTERNOON = 1_704_116_832_345
T_LEAP_END = 1_709_251_199_999


# --- cas nominal : valeurs à la main -----------------------------------------------------


def test_constants() -> None:
    assert tu.MS_PER_DAY == 86_400_000
    assert tu.US_PER_S == 1_000_000


def test_conversions_ms_us() -> None:
    assert tu.ms_to_us(T_2024) == 1_704_067_200_000_000
    assert tu.us_to_ms(1_704_067_200_000_999) == T_2024  # troncature vers −∞
    assert tu.us_to_ms(tu.ms_to_us(T_AFTERNOON)) == T_AFTERNOON  # aller-retour exact


def test_slot_and_hour() -> None:
    assert tu.slot_of_day(T_AFTERNOON, 5) == 165
    assert tu.slot_of_day(T_AFTERNOON, 60) == 13
    assert tu.hour_of_day(T_AFTERNOON) == 13
    assert tu.slot_of_day(T_2024, 5) == 0
    assert tu.slot_of_day(T_LEAP_END, 5) == 287  # dernière tranche : 1440/5 − 1
    assert tu.slot_of_day(T_LEAP_END, 1440) == 0  # tranche = journée entière


@pytest.mark.parametrize(
    ("ts_ms", "expected"),
    [
        (0, 3),  # 1970-01-01, jeudi
        (T_2024, 0),  # lundi
        (T_2024 + 6 * tu.MS_PER_DAY, 6),  # 2024-01-07, dimanche
        (T_2024 + 7 * tu.MS_PER_DAY, 0),  # lundi suivant
        (T_LEAP_END, 3),  # 2024-02-29, jeudi
    ],
)
def test_weekday(ts_ms: int, expected: int) -> None:
    assert tu.weekday(ts_ms) == expected


def test_floor_and_day_start() -> None:
    assert tu.day_start_ms(T_AFTERNOON) == T_2024
    assert tu.floor_ms(T_AFTERNOON, tu.MS_PER_MIN) == T_AFTERNOON - 12_345
    assert tu.floor_ms(T_AFTERNOON, 1) == T_AFTERNOON


def test_dates_and_iso() -> None:
    assert tu.date_str(T_AFTERNOON) == "2024-01-01"
    assert tu.date_str(T_LEAP_END) == "2024-02-29"
    assert tu.date_to_ms("2024-01-01") == T_2024
    assert tu.date_to_ms("2024-02-29") == T_LEAP_END + 1 - tu.MS_PER_DAY
    assert tu.ms_to_iso(T_AFTERNOON) == "2024-01-01T13:47:12.345Z"
    assert tu.ms_to_iso(T_LEAP_END) == "2024-02-29T23:59:59.999Z"


def test_binance_archive_units() -> None:
    """Archives spot Binance : ms avant 2025, µs à partir du 2025-01-01."""
    ts_2025_us = 1_735_689_600_000_000  # 2025-01-01T00:00:00Z en µs
    assert tu.infer_epoch_unit(T_2024) == "ms"
    assert tu.infer_epoch_unit(ts_2025_us) == "us"
    assert tu.to_us(T_2024, "ms") == tu.ms_to_us(T_2024)
    assert tu.to_us(ts_2025_us, "us") == ts_2025_us
    assert tu.date_str(tu.us_to_ms(ts_2025_us)) == "2025-01-01"
    assert tu.weekday(tu.us_to_ms(ts_2025_us)) == 2  # mercredi


def test_now_is_integer_and_consistent() -> None:
    before_ns = time.time_ns()
    ms, us = tu.now_ms(), tu.now_us()
    assert isinstance(ms, int) and isinstance(us, int)
    assert before_ns // 1_000_000 <= ms <= us // 1_000 + 1
    assert tu.infer_epoch_unit(ms) == "ms"
    assert tu.infer_epoch_unit(us) == "us"


# --- cas limites -------------------------------------------------------------------------


def test_before_epoch_floors_toward_minus_infinity() -> None:
    assert tu.us_to_ms(-1) == -1
    assert tu.day_start_ms(-1) == -tu.MS_PER_DAY
    assert tu.date_str(-1) == "1969-12-31"
    assert tu.weekday(-1) == 2  # mercredi
    assert tu.slot_of_day(-1, 5) == 287
    assert tu.hour_of_day(-1) == 23
    assert tu.ms_to_iso(-1) == "1969-12-31T23:59:59.999Z"


def test_epoch_zero() -> None:
    assert tu.date_str(0) == "1970-01-01"
    assert tu.ms_to_iso(0) == "1970-01-01T00:00:00.000Z"
    assert tu.slot_of_day(0, 5) == 0


def test_infer_unit_bounds() -> None:
    assert tu.infer_epoch_unit(946_684_800_000) == "ms"  # 2000-01-01 pile
    assert tu.infer_epoch_unit(4_102_444_800_000 - 1) == "ms"
    assert tu.infer_epoch_unit(946_684_800_000_000) == "us"


def test_polars_expr_match_scalar() -> None:
    values = [-1, 0, T_2024, T_AFTERNOON, T_LEAP_END]
    df = pl.DataFrame({"ts_ms": values}, schema={"ts_ms": pl.Int64}).select(
        tu.slot_of_day_expr(pl.col("ts_ms"), 5).alias("slot"),
        tu.weekday_expr(pl.col("ts_ms")).alias("wd"),
        tu.floor_ms_expr(pl.col("ts_ms"), tu.MS_PER_DAY).alias("day"),
    )
    assert df["slot"].to_list() == [tu.slot_of_day(v, 5) for v in values]
    assert df["wd"].to_list() == [tu.weekday(v) for v in values]
    assert df["day"].to_list() == [tu.day_start_ms(v) for v in values]
    assert df.schema == {"slot": pl.Int64, "wd": pl.Int64, "day": pl.Int64}  # pas de flottant


def test_polars_empty_column() -> None:
    df = pl.DataFrame({"ts_ms": []}, schema={"ts_ms": pl.Int64})
    out = df.select(tu.slot_of_day_expr(pl.col("ts_ms"), 5))
    assert out.height == 0


# --- cas d'erreur ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1.0, True, "1704067200000", None])
def test_non_integer_rejected(bad: object) -> None:
    with pytest.raises(TypeError, match="entier attendu"):
        tu.weekday(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entier attendu"):
        tu.ms_to_us(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("slot", [0, -5, 7, 1441])
def test_bad_slot_rejected(slot: int) -> None:
    with pytest.raises(ValueError, match="diviser 1440"):
        tu.slot_of_day(T_2024, slot)
    with pytest.raises(ValueError, match="diviser 1440"):
        tu.slot_of_day_expr(pl.col("ts_ms"), slot)


@pytest.mark.parametrize("step", [0, -1])
def test_bad_step_rejected(step: int) -> None:
    with pytest.raises(ValueError, match="step_ms doit être > 0"):
        tu.floor_ms(T_2024, step)
    with pytest.raises(ValueError, match="step_ms doit être > 0"):
        tu.floor_ms_expr(pl.col("ts_ms"), step)


@pytest.mark.parametrize("ts", [0, 17, 946_684_800_000 - 1, 4_102_444_800_000_000, -T_2024])
def test_infer_unit_out_of_range(ts: int) -> None:
    with pytest.raises(ValueError, match="hors plage"):
        tu.infer_epoch_unit(ts)


def test_to_us_unknown_unit() -> None:
    with pytest.raises(ValueError, match="unité inconnue"):
        tu.to_us(T_2024, "s")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["2024-1-5", "2024/01/01", "2024-02-30", "", "2024-01-01T00"])
def test_bad_date_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD attendue"):
        tu.date_to_ms(bad)


# --- non-régression du break test (2026-09-25) --------------------------------------------


@pytest.mark.parametrize("ts", [10**15, -(10**15), 2**63, 1_735_689_600_000_000])
def test_out_of_range_dates_raise_data_error(ts: int) -> None:
    """Des µs lues comme des ms (ex. 2025-01-01 en µs) donnent l'an ~57 000 : erreur claire."""
    with pytest.raises(DataError, match="µs confondues"):
        tu.date_str(ts)
    with pytest.raises(DataError, match="hors des années 1–9999"):
        tu.ms_to_iso(ts)
