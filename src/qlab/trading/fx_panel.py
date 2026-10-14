"""Panel horaire du volet trading : devises, indices, matières premières.

Même forme que le panel des perpétuels (``timing/panel.py``) : une ligne par (instrument,
heure) entre la première et la dernière heure échangée, ``bar`` = heure depuis l'époque. Les
heures sans échange (nuits de certains indices, week-ends, fériés) sont des barres fermées
(``timing.panel.carry_closed`` : clôture reportée, aucune entrée ni sortie).

- ``spread_frac`` : spread complet supposé de l'instrument (``data/histdata.py``, config),
  facturé à moitié à chaque trade par le portefeuille (``timing/portfolio.py``) ;
- ``eligible`` : instrument coté depuis au moins ``warmup_bars`` heures échangées, heure
  échangée, heure UTC hors de ``excluded_hours_utc`` (bascule quotidienne : spreads écartés) ;
- ``funding`` = 0 : le coût de portage d'une nuit (swap) n'est pas modélisé (spec §3).
"""

from __future__ import annotations

import polars as pl

from qlab.core.config import TradingConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_HOUR
from qlab.timing.hourly_signals import HOURS_PER_DAY
from qlab.timing.panel import carry_closed, complete_grid

COLUMNS = ("open", "high", "low", "close", "spread_frac", "volume")


def assemble(bars: pl.DataFrame, cfg: TradingConfig) -> pl.DataFrame:
    """Barres horaires échangées (``symbol``, ``t_ms``, prix moyens, spread, volume) → panel."""
    panel = carry_closed(complete_grid(bars)).with_columns(
        (pl.col("t_ms") // MS_PER_HOUR).alias("bar")
    )
    age = pl.col("traded").cast(pl.Int64).cum_sum().over("symbol")
    hour = (pl.col("t_ms") // MS_PER_HOUR) % HOURS_PER_DAY
    eligible = (
        pl.col("traded") & (age > cfg.warmup_bars) & ~hour.is_in(list(cfg.excluded_hours_utc))
    )
    return panel.select(
        "symbol",
        "t_ms",
        "bar",
        "open",
        "high",
        "low",
        "close",
        pl.col("volume").alias("volume_quote"),
        "spread_frac",
        "traded",
        pl.lit(0.0).alias("funding"),
        eligible.fill_null(False).alias("eligible"),
    )


def load(paths: DataPaths, cfg: TradingConfig) -> pl.DataFrame:
    """Panel des instruments de la config dont les barres sont construites."""
    present = [i.name for i in cfg.instruments if paths.trading_hourly(i.name).exists()]
    if not present:
        raise DataError("aucune barre horaire construite (data.dukascopy download puis build)")
    bars = pl.concat(
        pl.read_parquet(paths.trading_hourly(name))
        .select("t_ms", *COLUMNS)
        .with_columns(pl.lit(name).alias("symbol"))
        for name in present
    )
    return assemble(bars, cfg)
