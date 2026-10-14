"""Tests de qlab.timing.panel : grille, financement, univers point-in-time, rendements futurs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, MS_PER_HOUR, date_to_ms
from qlab.timing import panel as pn

H = MS_PER_HOUR
T0 = date_to_ms("2024-01-01")
UNI = pn.Universe("USDT", frozenset({"USDC"}), 1, 2, 100.0)


def _bars(
    symbol: str, opens: list[float | None], t0: int = T0, volume: float = 10.0
) -> pl.DataFrame:
    rows = [
        {
            "symbol": symbol,
            "t_ms": t0 + i * H,
            "open": o,
            "high": o,
            "low": o,
            "close": o,
            "volume_quote": volume,
            "taker_buy_quote": volume / 2,
        }
        for i, o in enumerate(opens)
        if o is not None
    ]
    return pl.DataFrame(rows)


NO_FUNDING = pl.DataFrame(
    schema={"symbol": pl.String, "funding_time_ms": pl.Int64, "rate": pl.Float64}
)


def test_grid_marks_gaps_and_shifts_are_hours() -> None:
    """Heure 2 absente : ligne présente, prix nul. Décision en t, entrée open(t+1).
    h=1 : t=0 sortie open(2) manquante ⇒ dernière clôture connue en t+h=1 (110) ⇒ 0 ;
    t=1 entrée sur le trou ⇒ nul ; t=2 : 126/120 − 1 ; t=3 : fin de série ⇒ dernière clôture."""
    p = pn.assemble(_bars("AUSDT", [100, 110, None, 120, 126]), NO_FUNDING, UNI)
    assert p["t_ms"].to_list() == [T0 + i * H for i in range(5)]
    assert p["open"].to_list() == [100, 110, None, 120, 126]
    fwd = p.select(pn.forward_return(1).alias("r"))["r"].to_list()
    assert fwd[0] == pytest.approx(0.0)
    assert fwd[1] is None
    assert fwd[2] == pytest.approx(126 / 120 - 1)
    assert fwd[3] == pytest.approx(0.0)
    two = p.select(pn.forward_return(2).alias("r"))["r"].to_list()
    assert two[0] == pytest.approx(120 / 110 - 1)  # open(1) → open(3), le trou est sauté
    assert two[2] == pytest.approx(126 / 120 - 1)  # sortie hors série ⇒ dernière clôture


def test_delisted_contract_closed_at_last_price() -> None:
    """Contrat retiré après une chute : la position décidée avant le retrait est soldée au
    dernier prix (−50 %), pas ignorée ; ``complete_only`` l'exclurait (fin des données)."""
    p = pn.assemble(_bars("AUSDT", [100, 100, 50]), NO_FUNDING, UNI)
    assert p.select(pn.forward_return(4).alias("r"))["r"].to_list()[0] == pytest.approx(-0.5)
    assert pn.complete_only(p, 1)["t_ms"].to_list() == [T0]


def test_forward_return_by_hand() -> None:
    p = pn.assemble(_bars("AUSDT", [100, 110, 121, 133.1]), NO_FUNDING, UNI)
    fwd = p.select(pn.forward_return(2).alias("r"))["r"].to_list()
    assert fwd[0] == pytest.approx(133.1 / 110 - 1)  # entrée open(1), sortie open(3)
    assert fwd[3] is None  # pas d'entrée


def test_funding_summed_per_hour_and_over_holding() -> None:
    """Deux règlements dans l'heure 1 (0,01 % + 0,02 %), un dans l'heure 3 (−0,05 %).
    Détention 2 h décidée en t=0 : heures 1 et 2 ⇒ 0,03 % ; décidée en t=1 : heures 2, 3."""
    funding = pl.DataFrame(
        {
            "symbol": ["AUSDT"] * 3,
            "funding_time_ms": [T0 + H + 1, T0 + H + 2, T0 + 3 * H + 5],
            "rate": [0.0001, 0.0002, -0.0005],
        }
    )
    p = pn.assemble(_bars("AUSDT", [1, 1, 1, 1, 1, 1]), funding, UNI)
    assert p["funding"].to_list() == pytest.approx([0, 0.0003, 0, -0.0005, 0, 0])
    held = p.select(pn.forward_funding(2).alias("f"))["f"].to_list()
    assert held[:3] == pytest.approx([0.0003, -0.0005, -0.0005])


def test_eligibility_is_point_in_time() -> None:
    """Volume 10/heure ⇒ 240/jour. Seuil 100, médiane sur 2 jours précédents, âge ≥ 1 jour :
    jour 0 non (pas d'historique), jour 1 oui (médiane du jour 0 = 240)."""
    p = pn.assemble(_bars("AUSDT", [1.0] * 48), NO_FUNDING, UNI)
    assert p["eligible"].to_list() == [False] * 24 + [True] * 24
    thin = pn.assemble(_bars("BUSDT", [1.0] * 48, volume=1.0), NO_FUNDING, UNI)
    assert not thin["eligible"].any()  # 24/jour < 100


def test_eligibility_ignores_future_volume() -> None:
    """Changer le volume du jour 1 ne change pas l'éligibilité du jour 1 (jours précédents)."""
    base = _bars("AUSDT", [1.0] * 48)
    loud = base.with_columns(
        pl.when(pl.col("t_ms") >= T0 + MS_PER_DAY)
        .then(1e9)
        .otherwise(pl.col("volume_quote"))
        .alias("volume_quote")
    )
    a = pn.assemble(base, NO_FUNDING, UNI)["eligible"]
    b = pn.assemble(loud, NO_FUNDING, UNI)["eligible"]
    assert a.to_list() == b.to_list()


def test_in_sample_stops_before_holdout() -> None:
    p = pn.assemble(_bars("AUSDT", [1.0] * 10), NO_FUNDING, UNI)
    cut = pn.bar_at(p, T0 + 6 * H)
    assert cut == (T0 + 6 * H) // H  # barre = heure depuis l'époque
    kept = pn.in_sample(p, cut, 2)["t_ms"].to_list()
    assert kept == [T0 + i * H for i in range(3)]  # sortie open(t+3) < t=6 ⇒ t ≤ 2
    with pytest.raises(DataError, match="aucune barre"):
        pn.bar_at(p, T0 + 10 * H)


def _write(paths: DataPaths, symbol: str, df: pl.DataFrame) -> None:
    out = paths.lt_futures("klines_1h", symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.rename({"t_ms": "open_time_ms"}).drop("symbol").write_parquet(out)


def test_load_from_disk_filters_quote_and_bases(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    with pytest.raises(DataError, match="absentes"):
        pn.contract_symbols(paths, UNI)
    for s in ("AUSDT", "USDCUSDT", "BTCUSDC"):
        _write(paths, s, _bars(s, [1.0, 2.0]))
    assert pn.contract_symbols(paths, UNI) == ["AUSDT"]
    rates = paths.lt_futures("funding", "AUSDT")
    rates.parent.mkdir(parents=True)
    pl.DataFrame({"funding_time_ms": [T0], "interval_hours": [8], "rate": [0.001]}).write_parquet(
        rates
    )
    p = pn.load(paths, UNI)
    assert p["symbol"].unique().to_list() == ["AUSDT"]
    assert p["funding"].to_list() == pytest.approx([0.001, 0.0])
    with pytest.raises(DataError, match="aucun contrat"):
        pn.load(paths, UNI, symbols=[])
