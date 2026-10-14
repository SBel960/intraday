"""Tests de qlab.timing.hourly_signals : valeurs calculées à la main, causalité, erreurs."""

from __future__ import annotations

import math

import polars as pl
import pytest

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.timing import hourly_signals as sg

H = MS_PER_HOUR
T0 = date_to_ms("2024-01-01")  # lundi 00:00 UTC


def _panel(closes: list[float], symbol: str = "AUSDT", **cols: list[float]) -> pl.DataFrame:
    n = len(closes)
    data: dict[str, object] = {
        "symbol": [symbol] * n,
        "t_ms": [T0 + i * H for i in range(n)],
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume_quote": [10.0] * n,
        "taker_buy_quote": [5.0] * n,
        "funding": [0.0] * n,
    }
    data.update(cols)
    return pl.DataFrame(data)


def _values(df: pl.DataFrame, name: str, k: float = 2, horizon: int = 1) -> list[float | None]:
    return sg.compute(df, name, k, horizon).to_list()


def test_momentum_and_reversal_by_hand() -> None:
    df = _panel([100, 110, 121, 100])
    mom = _values(df, "momentum")
    assert mom[:2] == [None, None]
    assert mom[2] == pytest.approx(math.log(121 / 100))
    assert mom[3] == pytest.approx(math.log(100 / 110))
    assert _values(df, "reversal")[3] == pytest.approx(-math.log(100 / 110))


def test_flow_by_hand_and_zero_volume() -> None:
    """Achats agressifs 8 et 2 sur volumes 10 et 10 : (16 − 10 + 4 − 10) / 20 = 0 ; puis
    2 et 9 : (4 − 10 + 18 − 10) / 20 = 0,1. Volume nul sur la fenêtre : pas de signal."""
    df = _panel(
        [1, 1, 1, 1],
        volume_quote=[10.0, 10.0, 10.0, 0.0],
        taker_buy_quote=[8.0, 2.0, 9.0, 0.0],
    )
    flow = _values(df, "flow")
    assert flow[1] == pytest.approx(0.0)
    assert flow[2] == pytest.approx((4 - 10 + 18 - 10) / 20)
    assert _values(df, "flow_reversal")[2] == pytest.approx(-0.1)
    zero = _panel([1, 1], volume_quote=[0.0, 0.0], taker_buy_quote=[0.0, 0.0])
    assert _values(zero, "flow")[1] is None


def test_funding_breakout_low_volatility_by_hand() -> None:
    df = _panel([100, 110, 99, 120], funding=[0.001, 0.002, -0.001, 0.0])
    assert _values(df, "funding_contrarian")[2] == pytest.approx(-0.001)  # −(0,002 − 0,001)
    brk = _values(df, "breakout")
    assert brk[3] == pytest.approx(120 / 110 - 1)  # plus haut des 2 h précédentes : 110
    assert brk[2] == pytest.approx(99 / 110 - 1)
    vol = _values(df, "low_volatility")
    r1, r2 = math.log(110 / 100), math.log(99 / 110)
    mean = (r1 + r2) / 2
    expected = math.sqrt(((r1 - mean) ** 2 + (r2 - mean) ** 2) / 1)
    assert vol[2] == pytest.approx(-expected)


def test_channel_and_aliases_by_hand() -> None:
    """Canal sur 2 h : t1 [100, 110] clôture 110 ⇒ +½ ; t2 [110, 99] clôture 99 ⇒ −½ ;
    t3 [99, 120] ⇒ +½. Range nul ⇒ pas de signal. un alias calcule la formule de sa cible."""
    df = _panel([100, 110, 99, 120])
    assert _values(df, "channel")[1:] == pytest.approx([0.5, -0.5, 0.5])
    assert _values(_panel([1.0, 1.0]), "channel") == [None, None]
    for alias, target in sg.ALIASES.items():
        assert sg.resolve(alias) == target
        assert _values(df, alias) == _values(df, target)


def test_volume_shock_by_hand() -> None:
    """Base : 30 jours à 10/h. Puis 2 h à 30/h, prix en hausse : ln(60 / (2 · 10)) = ln 3."""
    n = sg.BASELINE_H + 2
    closes = [100.0] * sg.BASELINE_H + [101.0, 102.0]
    df = _panel(closes, volume_quote=[10.0] * sg.BASELINE_H + [30.0, 30.0])
    shock = _values(df, "volume_shock")
    assert shock[n - 1] == pytest.approx(math.log(3))
    assert shock[sg.BASELINE_H] is None  # moyenne des 30 j précédents incomplète


def test_btc_catchup_by_hand() -> None:
    btc = _panel([100.0, 100.0, 110.0], symbol="BTCUSDT")
    alt = _panel([10.0, 10.0, 10.5], symbol="AUSDT")
    df = sg.with_reference(pl.concat([alt, btc]), "BTCUSDT")
    got = df.with_columns(sg.compute(df, "btc_catchup", 2, 1).alias("s"))
    alt_s = got.filter(pl.col("symbol") == "AUSDT")["s"].to_list()
    assert alt_s[2] == pytest.approx(math.log(1.1) - math.log(1.05))
    assert got.filter(pl.col("symbol") == "BTCUSDT")["s"].to_list()[2] == pytest.approx(0.0)
    with pytest.raises(DataError, match="référence absent"):
        sg.with_reference(alt, "BTCUSDT")


def test_seasonality_uses_only_past_weeks() -> None:
    """3 semaines ; le lundi 00:00, rendement 1 h de +1 % en semaine 1, +3 % en semaine 2.
    Signal du lundi 00:00 : semaine 1 rien, semaine 2 → 1 %, semaine 3 → 2 %."""
    week = sg.HOURS_PER_WEEK
    opens = [100.0] * (3 * week + 3)
    opens[2] = 101.0  # entrée open(1)=100, sortie open(2)=101 : +1 % décidé en t=0
    opens[week + 2] = 103.0
    df = _panel(opens)
    s = sg.compute(df, "seasonality", None, 1).to_list()
    assert s[0] is None
    assert s[week] == pytest.approx(0.01)
    assert s[2 * week] == pytest.approx(0.02)
    with pytest.raises(DataError, match="trop long"):
        sg.compute(df, "seasonality", None, week)


@pytest.mark.parametrize("name", [n for n in sg.NAMES if n != "btc_catchup"])
def test_causal_future_change_does_not_move_signal(name: str) -> None:
    """Modifier les bougies après t ne change aucun signal jusqu'à t (sauf seasonality, qui lit
    des rendements passés d'au moins une semaine : vérifiée sur la même règle)."""
    n = sg.BASELINE_H + 30
    closes = [100.0 + (i % 7) for i in range(n)]
    base = _panel(closes, funding=[0.0001 * (i % 3) for i in range(n)]).with_columns(
        pl.lit(None, pl.Int64).alias("delisting_age"),  # causalité des annonces : test_event_book
        pl.lit(None, pl.Int64).alias("listing_age"),
    )
    cut = n - 5
    changed = base.with_columns(
        pl.when(pl.col("t_ms") > T0 + cut * H)
        .then(pl.col("close") * 3)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("t_ms") > T0 + cut * H)
        .then(pl.col("open") * 3)
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.when(pl.col("t_ms") > T0 + cut * H)
        .then(1e6)
        .otherwise(pl.col("volume_quote"))
        .alias("volume_quote"),
    )
    horizon = 2
    a = _values(base, name, 3, horizon)
    b = _values(changed, name, 3, horizon)
    limit = cut + 1 - (horizon + 1 if sg.resolve(name) == "seasonality" else 0)
    assert a[:limit] == b[:limit]


def test_bad_parameters() -> None:
    df = _panel([1.0, 2.0])
    with pytest.raises(DataError, match="signal inconnu"):
        sg.compute(df, "magie", 1, 1)
    with pytest.raises(DataError, match="entière"):
        sg.compute(df, "momentum", 1.5, 1)
