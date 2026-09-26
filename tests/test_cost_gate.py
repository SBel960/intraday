"""Tests de qlab.costs.cost_gate : scénarios construits à la main, valeurs attendues écrites.

Scénario de base : une cotation par seconde ; mid multiplié par 1,001 chaque seconde ; spread
relatif 0,0002 (bid = m (1 − 0,0001), ask = m (1 + 0,0001)) ; f_taker 0,1 %, slip 0 ⇒
c = 2 × 0,001 + 0,0002 = 0,0022.
- h = 5 s  : |ln 1,001⁵| = 5 ln 1,001 = 0,0049975 ⇒ ratio 2,27 < 3
- h = 10 s : 10 ln 1,001 = 0,0099950 ⇒ ratio 4,54 > 3 ⇒ plus petit horizon rentable : 10 s
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.costs.cost_gate import ALL_SLOTS, GateParams, run_gate, sample_segment, summarize

T = 1_704_067_200_000  # 2024-01-01T00:00:00Z : tranche horaire 0
HOUR = 3_600_000
PARAMS = GateParams(
    fee_taker=0.001,
    slippage=0.0,
    horizons_s=(5, 10),
    min_ratio=3.0,
    slot_minutes=60,
    step_ms=1000,
    max_staleness_ms=2000,
)


def _quotes(
    start_ms: int, seconds: int, growth: float, mid0: float = 100.0, spread: float = 0.0002
) -> pl.DataFrame:
    """Une cotation par seconde, mid × ``growth`` à chaque seconde, spread relatif fixe."""
    mids = [mid0 * growth**k for k in range(seconds + 1)]
    half = spread / 2
    return pl.DataFrame(
        {
            "ts_ms": [start_ms + 1000 * k for k in range(seconds + 1)],
            "bid": [m * (1 - half) for m in mids],
            "ask": [m * (1 + half) for m in mids],
        },
        schema={"ts_ms": pl.Int64, "bid": pl.Float64, "ask": pl.Float64},
    )


def _row(result_table: pl.DataFrame, slot: int, horizon: int) -> dict[str, float]:
    rows = result_table.filter((pl.col("slot") == slot) & (pl.col("horizon_s") == horizon))
    assert rows.height == 1
    return rows.row(0, named=True)


def test_by_hand() -> None:
    r = run_gate(_quotes(T, 30, 1.001), PARAMS)
    h5, h10 = _row(r.table, 0, 5), _row(r.table, 0, 10)
    assert h5["cost_median"] == pytest.approx(0.0022)
    assert h5["move_median"] == pytest.approx(5 * math.log(1.001))
    assert h5["ratio"] == pytest.approx(5 * math.log(1.001) / 0.0022)  # 2,27
    assert h10["ratio"] == pytest.approx(10 * math.log(1.001) / 0.0022)  # 4,54
    assert (h5["n"], h10["n"]) == (26, 21)  # 31 instants ; t + h doit rester dans les données
    assert r.min_horizon_s == {0: 10, ALL_SLOTS: 10}
    assert r.verdict() == "Plus petit horizon rentable (ratio > 3) : 10 s."
    assert (r.quotes_used, r.quotes_excluded) == (31, 0)


def test_nothing_passes_verdict() -> None:
    r = run_gate(_quotes(T, 30, 1.0001), PARAMS)  # mouvements 10× plus petits
    assert r.overall_min_horizon_s is None
    assert (
        r.verdict() == "Aucun horizon ≤ 10 s ne dépasse le ratio 3 : aucune stratégie "
        "intraday n'est testée (§5)."
    )


def test_hourly_slots() -> None:
    """Heure 0 agitée (×1,002/s), heure 1 immobile : seule la tranche 0 passe."""
    quotes = pl.concat([_quotes(T, 30, 1.002), _quotes(T + HOUR, 30, 1.0)])
    r = run_gate(quotes, PARAMS)
    assert r.min_horizon_s[0] == 5  # 5 ln 1,002 / 0,0022 = 4,54
    assert r.min_horizon_s[1] is None
    assert _row(r.table, 1, 5)["move_median"] == 0.0


def test_gap_is_excluded_not_interpolated() -> None:
    """Trou de 20 s entre deux blocs : les instants du trou (cotation > 2 s) sont exclus."""
    quotes = pl.concat([_quotes(T, 10, 1.001), _quotes(T + 31_000, 10, 1.001, mid0=200.0)])
    r = run_gate(quotes, PARAMS)
    h5 = _row(r.table, 0, 5)
    # bloc 1 : t = 0…5 (6) ; t = 6, 7 : mid(t+5) = cotation de 10 s, âgée de 1 et 2 s ≤ 2 s
    # (tolérance de fraîcheur) ⇒ gardés ; t = 8…30 : cotation de plus de 2 s ⇒ trou, exclus ;
    # bloc 2 : t = 31…36 (6). Total 6 + 2 + 6 = 14.
    assert h5["n"] == 14
    assert h5["move_median"] == pytest.approx(5 * math.log(1.001))  # jamais 200 / 100


def test_invalid_quotes_excluded_and_counted() -> None:
    quotes = _quotes(T, 30, 1.001).with_columns(
        pl.when(pl.col("ts_ms") == T + 3000)
        .then(pl.col("bid") * 2)
        .otherwise(pl.col("bid"))
        .alias("bid")
    )  # une cotation croisée (bid > ask)
    r = run_gate(quotes, PARAMS)
    assert (r.quotes_used, r.quotes_excluded) == (30, 1)


def test_deterministic() -> None:
    q = _quotes(T, 30, 1.001)
    assert run_gate(q, PARAMS).table.equals(run_gate(q, PARAMS).table)


def test_params_from_config(config_dir: Path) -> None:
    p = GateParams.from_config(load_config(config_dir), fee_taker=0.00095)
    assert p.horizons_s == (5, 30, 60, 300, 900)
    assert (p.min_ratio, p.slot_minutes, p.step_ms, p.max_staleness_ms) == (3.0, 60, 1000, 2000)
    assert p.fee_taker == 0.00095


@pytest.mark.parametrize(
    ("quotes", "msg"),
    [
        (pl.DataFrame({"ts_ms": [T], "bid": [1.0]}), "colonnes manquantes"),
        (
            pl.DataFrame(schema={"ts_ms": pl.Int64, "bid": pl.Float64, "ask": pl.Float64}),
            "aucune ligne",
        ),
        (pl.DataFrame({"ts_ms": [T + 1000, T], "bid": [1.0, 1.0], "ask": [1.1, 1.1]}), "trié"),
        (
            pl.DataFrame({"ts_ms": [T, T + 1000], "bid": [2.0, 2.0], "ask": [1.0, 1.0]}),
            "aucune valide",
        ),
        (_quotes(T, 3, 1.001), "pas assez de données"),
    ],
)
def test_errors(quotes: pl.DataFrame, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        run_gate(quotes, PARAMS)


def test_segments_pool_samples_not_medians() -> None:
    """Deux journées séparées de 30 jours : pas de grille entre elles, médianes sur l'ensemble.
    Jour 1 : ×1,001/s (5 ln 1,001) ; jour 2 : ×1,003/s (5 ln 1,003) ; 26 échantillons chacun à
    5 s ⇒ médiane des 52 = moyenne des deux valeurs centrales."""
    day = 86_400_000
    seg1 = sample_segment(_quotes(T, 30, 1.001), PARAMS)
    seg2 = sample_segment(_quotes(T + 30 * day, 30, 1.003), PARAMS)
    r = summarize([seg1, seg2], PARAMS)
    h5 = _row(r.table, ALL_SLOTS, 5)
    assert h5["n"] == 52  # aucune grille entre les deux journées
    assert h5["move_median"] == pytest.approx((5 * math.log(1.001) + 5 * math.log(1.003)) / 2)
    assert (r.quotes_used, r.quotes_excluded) == (62, 0)


def test_summarize_nothing() -> None:
    with pytest.raises(DataError, match="pas assez de données"):
        summarize([], PARAMS)
