"""Panel horaire des contrats perpétuels USDⓈ-M : grille complète, financement, univers liquide.

Une ligne par (contrat, heure) entre la première et la dernière bougie du contrat ; une heure
sans bougie reste une ligne à prix nul (trou marqué, jamais interpolé). Les décalages d'une
série valent donc exactement des heures : ``shift(k)`` sur un contrat = k heures plus tôt.

- contrats : ``lt/futures/klines_1h`` cotés en ``quote`` (USDT), base absente des bases exclues
  (stablecoins, devises : ``longterm.universe.excluded_bases``) ;
- ``funding`` : somme des taux de financement réglés pendant l'heure (positif : l'acheteur
  paie) ; 0 sans règlement ;
- ``eligible`` (point-in-time, sans biais du survivant) : coté depuis au moins ``warmup_days``
  et volume quotidien médian des ``volume_lookback_days`` **jours précédents** ≥
  ``min_volume_quote``. Le volume du jour en cours n'entre pas : décision possible à l'heure.

Rendements sur l'ouverture : la décision prise à la clôture de l'heure ``t`` s'exécute à
l'ouverture de l'heure ``t + 1`` (``forward_return``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, MS_PER_HOUR

COLUMNS = ("symbol", "t_ms", "open", "high", "low", "close", "volume_quote", "taker_buy_quote")
INTERVAL = "1h"


@dataclass(frozen=True, slots=True)
class Universe:
    """Règles de l'univers point-in-time (``timing.yaml`` + bases exclues du long terme)."""

    quote: str
    excluded_bases: frozenset[str]
    warmup_days: int
    volume_lookback_days: int
    min_volume_quote: float


def contract_symbols(paths: DataPaths, universe: Universe) -> list[str]:
    """Contrats construits sur disque, cotés en ``quote``, base non exclue ; triés."""
    folder = paths.lt_futures(f"klines_{INTERVAL}", "X").parent
    if not folder.is_dir():
        raise DataError("bougies des contrats absentes : klines_build build --market um")
    names = sorted(p.stem for p in folder.glob("*.parquet"))
    q = universe.quote
    return [s for s in names if s.endswith(q) and s.removesuffix(q) not in universe.excluded_bases]


def _read(path: Path, symbol: str) -> pl.LazyFrame:
    return pl.scan_parquet(path).select(
        pl.lit(symbol).alias("symbol"),
        pl.col("open_time_ms").alias("t_ms"),
        *COLUMNS[2:],
    )


def complete_grid(bars: pl.DataFrame) -> pl.DataFrame:
    """Une ligne par heure entre la première et la dernière bougie de chaque contrat."""
    grid = (
        bars.group_by("symbol")
        .agg(pl.int_ranges(pl.min("t_ms"), pl.max("t_ms") + 1, MS_PER_HOUR).alias("t_ms"))
        .explode("t_ms", empty_as_null=True)
    )
    return grid.join(bars, on=["symbol", "t_ms"], how="left").sort("symbol", "t_ms")


def hourly_funding(funding: pl.DataFrame) -> pl.DataFrame:
    """Règlements (``symbol``, ``funding_time_ms``, ``rate``) → somme par heure."""
    return (
        funding.with_columns((pl.col("funding_time_ms") // MS_PER_HOUR * MS_PER_HOUR).alias("t_ms"))
        .group_by("symbol", "t_ms")
        .agg(pl.col("rate").sum().alias("funding"))
    )


def eligibility(panel: pl.DataFrame, universe: Universe) -> pl.Series:
    """Masque point-in-time : âge ≥ warmup et médiane des volumes quotidiens passés ≥ seuil."""
    daily = (
        panel.with_columns((pl.col("t_ms") // MS_PER_DAY).alias("day"))
        .group_by("symbol", "day")
        .agg(pl.col("volume_quote").sum().alias("vol"))
        .sort("symbol", "day")
        .with_columns(
            pl.col("vol")
            .rolling_median(universe.volume_lookback_days, min_samples=1)
            .shift(1)
            .over("symbol")
            .alias("median_vol"),
            pl.col("day").min().over("symbol").alias("first_day"),
        )
    )
    ok = (pl.col("median_vol") >= universe.min_volume_quote) & (
        pl.col("day") - pl.col("first_day") >= universe.warmup_days
    )
    days = daily.select("symbol", "day", ok.fill_null(False).alias("eligible"))
    keyed = panel.select("symbol", (pl.col("t_ms") // MS_PER_DAY).alias("day"))
    return keyed.join(days, on=["symbol", "day"], how="left", maintain_order="left")["eligible"]


def assemble(bars: pl.DataFrame, funding: pl.DataFrame, universe: Universe) -> pl.DataFrame:
    """Bougies + règlements de plusieurs contrats → panel complet (grille, funding, eligible)."""
    panel = complete_grid(bars)
    panel = panel.join(hourly_funding(funding), on=["symbol", "t_ms"], how="left")
    panel = panel.with_columns(pl.col("funding").fill_null(0.0))
    return panel.with_columns(eligibility(panel, universe).fill_null(False).alias("eligible"))


def load(
    paths: DataPaths, universe: Universe, symbols: Iterable[str] | None = None
) -> pl.DataFrame:
    """Panel de tous les contrats de l'univers (ou de ``symbols``) lus sur disque."""
    chosen = contract_symbols(paths, universe) if symbols is None else list(symbols)
    if not chosen:
        raise DataError("aucun contrat dans l'univers")
    bars = pl.concat(
        [_read(paths.lt_futures(f"klines_{INTERVAL}", s), s) for s in chosen]
    ).collect()
    rates = [
        pl.scan_parquet(p).select(pl.lit(s).alias("symbol"), "funding_time_ms", "rate")
        for s in chosen
        if (p := paths.lt_futures("funding", s)).exists()
    ]
    empty = pl.DataFrame(
        schema={"symbol": pl.String, "funding_time_ms": pl.Int64, "rate": pl.Float64}
    )
    funding = pl.concat(rates).collect() if rates else empty
    return assemble(bars, funding, universe)


def forward_return(horizon_h: int) -> pl.Expr:
    """Rendement simple de l'ouverture de ``t + 1`` à la sortie, ``h`` heures plus tard (par
    contrat). Sortie : ouverture de ``t + 1 + h`` ; si elle manque (trou, retrait de la cote),
    dernière clôture connue à ``t + h``, puis dernière clôture du contrat : une position sur un
    contrat retiré est soldée à son dernier prix, jamais ignorée (biais du survivant). Les
    dernières heures d'un contrat encore coté en fin de données sont à exclure par l'appelant
    (``in_sample``, ``complete_only``)."""
    entry = pl.col("open").shift(-1).over("symbol")
    exit_ = pl.coalesce(
        pl.col("open").shift(-(1 + horizon_h)).over("symbol"),
        pl.col("close").forward_fill().shift(-horizon_h).over("symbol"),
        pl.col("close").drop_nulls().last().over("symbol"),
    )
    return exit_ / entry - 1


def complete_only(panel: pl.DataFrame, horizon_h: int) -> pl.DataFrame:
    """Lignes dont la sortie tombe avant la fin des données (``t + 1 + h`` ≤ dernière heure)."""
    last = panel["t_ms"].max()
    if not isinstance(last, int):
        return panel
    return panel.filter(pl.col("t_ms") + (1 + horizon_h) * MS_PER_HOUR <= last)


def forward_funding(horizon_h: int) -> pl.Expr:
    """Financement réglé pendant la détention : heures ``t + 1`` à ``t + h`` (par contrat). Un
    règlement à l'instant même de l'entrée est compté (prudent), celui de la sortie non."""
    total = pl.col("funding").rolling_sum(horizon_h).shift(-horizon_h).over("symbol")
    return total.fill_null(0.0)


def in_sample(panel: pl.DataFrame, holdout_start_ms: int, horizon_h: int) -> pl.DataFrame:
    """Lignes dont la détention se termine avant le hold-out (aucune donnée du hold-out lue)."""
    return panel.filter(pl.col("t_ms") + (1 + horizon_h) * MS_PER_HOUR < holdout_start_ms)
