"""Tests de qlab.timing.event_book : barre de signal, éligibilité, âge des annonces."""

from __future__ import annotations

import polars as pl

from qlab.core.timeutils import MS_PER_HOUR
from qlab.timing import event_book as eb
from qlab.timing import hourly_signals as hs

H = MS_PER_HOUR
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC


def _panel() -> pl.DataFrame:
    """AUSDT éligible toutes les heures ; BUSDT jamais (illiquide)."""
    rows = [
        {"symbol": s, "t_ms": T0 + i * H, "eligible": s == "AUSDT", "close": 1.0}
        for s in ("AUSDT", "BUSDT")
        for i in range(8)
    ]
    return pl.DataFrame(rows)


def _events(*rows: tuple[str, str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"kind": k, "base": b, "effective_ms": e} for k, b, e in rows],
        schema={"kind": pl.String, "base": pl.String, "effective_ms": pl.Int64},
    )


def test_signal_bar_is_the_first_close_after_the_announcement() -> None:
    """Annonce connue à 02:10 : barre 02:00 (clôture 03:00) ; à 03:00 pile : barre 02:00.
    B illiquide : ignoré."""
    ev = _events(
        ("delisting", "A", T0 + 2 * H + 600_000),
        ("delisting", "B", T0 + 2 * H),
        ("listing", "A", T0 + 3 * H),
    )
    got = eb.signal_times(_panel(), ev, "delisting", "USDT")
    assert got.rows() == [("AUSDT", T0 + 2 * H)]
    assert eb.signal_times(_panel(), ev, "listing", "USDT").rows() == [("AUSDT", T0 + 2 * H)]


def test_ages_and_signals() -> None:
    """Retrait de A à la barre 2 : âge nul avant, 0 à la barre 2, 5 à la barre 7 ; signal
    −1 pendant 3 barres (2, 3, 4), entrée à l'ouverture de la barre 3."""
    ev = _events(("delisting", "A", T0 + 2 * H + 1))
    p = eb.with_event_ages(_panel(), ev, "USDT")
    a = p.filter(pl.col("symbol") == "AUSDT")
    assert a["delisting_age"].to_list() == [None, None, 0, 1, 2, 3, 4, 5]
    assert a["listing_age"].null_count() == 8
    assert p["symbol"].to_list() == _panel()["symbol"].to_list()  # ordre conservé
    sig = hs.compute(p, "delisting_short", 3, 1)
    assert sig.to_list()[:8] == [0, 0, -1, -1, -1, 0, 0, 0]
    assert hs.compute(p, "listing_long", 3, 1).to_list()[:8] == [0.0] * 8


def test_fomc_hours_and_pre_fomc_signal() -> None:
    """Communiqué à 06:00 : BTC reçoit ``fomc_to`` = 6, 5, … ; pre_fomc(2) achète aux barres
    de signal 03:00 et 04:00 (à 3 et 2 h), donc tenu de 04:00 à 06:00, sortie au communiqué."""
    p = _panel().with_columns(pl.col("symbol").replace("AUSDT", "BTCUSDT"))
    got = eb.with_fomc(p, [T0 + 6 * H], "BTCUSDT")
    btc = got.filter(pl.col("symbol") == "BTCUSDT")
    assert btc["fomc_to"].to_list() == [6, 5, 4, 3, 2, 1, 0, None]
    assert got.filter(pl.col("symbol") == "BUSDT")["fomc_to"].null_count() == 8
    sig = hs.compute(got.with_columns(pl.lit(None).alias("delisting_age")), "pre_fomc", 2, 1)
    assert sig.to_list()[:8] == [0, 0, 0, 1, 1, 0, 0, 0]
