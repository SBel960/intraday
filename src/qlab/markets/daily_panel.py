"""Panel quotidien des marchés du volet multi-actifs : grille des jours ouvrés, univers liquide.

Même forme que le panel horaire des perpétuels (``timing/panel.py``), pour réutiliser ses
fenêtres à venir, les signaux et le portefeuille : une ligne par (marché, jour ouvré) entre la
première et la dernière séance du marché ; ``bar`` = numéro du jour ouvré depuis l'époque
(lundi → vendredi consécutifs : ``shift(k)`` = k jours ouvrés). Les séances du week-end
(cryptos) sont retirées : leur mouvement entre dans le rendement du lundi.

Jour sans séance (férié, erreur retirée) : ``traded`` faux, ouverture nulle (aucune entrée ni
sortie ce jour-là), **clôture reportée** de la veille, plus haut et plus bas égaux à cette
clôture : le marché était fermé, son dernier prix reste le prix connu. Sans ce report, chaque
férié vide les fenêtres des signaux, des canaux et de la volatilité (un férié toutes les
quelques semaines).

- **pointe aller-retour** (erreur de cotation) : |ln(c_t / c_{t−1})| et |ln(c_{t+1} / c_t)| tous
  deux > ``spike_log_return``, de signes opposés ⇒ la séance ``t`` devient un trou ;
- ``eligible`` (point-in-time) : au moins ``warmup_days`` séances réelles déjà cotées et
  volume médian en dollars des ``volume_lookback_days`` séances **précédentes**
  ≥ ``min_volume_usd`` ;
- ``volume_quote`` ≈ clôture × volume (dollars) ; pour les cryptos la source donne déjà des
  dollars : le volume est surestimé, sans effet (le seuil est de toute façon franchi) ;
- ``funding`` = 0 (coût d'emprunt des ventes à découvert non modélisé, cf. la spec §3).
"""

from __future__ import annotations

import polars as pl

from qlab.core.config import MarketsConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY

DAYS_PER_WEEK = 7
WEEKDAYS = 5
_EPOCH_WEEKDAY = 3  # le 1970-01-01 est un jeudi (lundi = 0)
PRICES = ("open", "high", "low", "close")


def weekday(t_ms: pl.Expr) -> pl.Expr:
    """Jour de la semaine d'un minuit UTC (lundi = 0)."""
    return (t_ms // MS_PER_DAY + _EPOCH_WEEKDAY) % DAYS_PER_WEEK


def business_bar(t_ms: pl.Expr) -> pl.Expr:
    """Numéro du jour ouvré (lundi → vendredi consécutifs) d'un minuit UTC de semaine."""
    shifted = t_ms // MS_PER_DAY + _EPOCH_WEEKDAY
    return shifted // DAYS_PER_WEEK * WEEKDAYS + shifted % DAYS_PER_WEEK


def complete_grid(bars: pl.DataFrame) -> pl.DataFrame:
    """Une ligne par jour ouvré entre la première et la dernière séance de chaque marché."""
    days = bars.filter(weekday(pl.col("t_ms")) < WEEKDAYS)
    grid = (
        days.group_by("symbol")
        .agg(pl.int_ranges(pl.min("t_ms"), pl.max("t_ms") + 1, MS_PER_DAY).alias("t_ms"))
        .explode("t_ms", empty_as_null=True)
        .filter(weekday(pl.col("t_ms")) < WEEKDAYS)
    )
    out = grid.join(days, on=["symbol", "t_ms"], how="left").sort("symbol", "t_ms")
    return out.with_columns(business_bar(pl.col("t_ms")).alias("bar"))


def remove_spikes(panel: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Séances en pointe aller-retour → trou (prix et volume nuls)."""
    up = (pl.col("close") / pl.col("close").shift(1)).log().over("symbol")
    down = (pl.col("close").shift(-1) / pl.col("close")).log().over("symbol")
    spike = (up.abs() > threshold) & (down.abs() > threshold) & (up * down < 0)
    flagged = panel.with_columns(spike.fill_null(False).alias("_spike"))
    return flagged.with_columns(
        pl.when(pl.col("_spike")).then(None).otherwise(pl.col(c)).alias(c)
        for c in (*PRICES, "volume_quote")
    ).drop("_spike")


def eligibility(panel: pl.DataFrame, cfg: MarketsConfig) -> pl.Expr:
    """Masque point-in-time : ancienneté et volume médian des séances précédentes."""
    traded = pl.col("traded").cast(pl.Int64).cum_sum().over("symbol")
    median = (
        pl.col("volume_quote")
        .rolling_median(cfg.volume_lookback_days, min_samples=1)
        .shift(1)
        .over("symbol")
    )
    return ((traded > cfg.warmup_days) & (median >= cfg.min_volume_usd)).fill_null(False)


def assemble(bars: pl.DataFrame, cfg: MarketsConfig) -> pl.DataFrame:
    """Séances (``symbol``, ``t_ms``, prix, ``volume``) → panel complet du volet."""
    dollars = bars.with_columns((pl.col("close") * pl.col("volume")).alias("volume_quote"))
    panel = remove_spikes(complete_grid(dollars.drop("volume")), cfg.spike_log_return)
    panel = panel.with_columns(
        pl.col("close").is_not_null().alias("traded"),
        pl.col("close").forward_fill().over("symbol"),
    ).with_columns(pl.col("high", "low").fill_null(pl.col("close")))
    return panel.with_columns(
        pl.lit(0.0).alias("funding"), eligibility(panel, cfg).alias("eligible")
    ).select("symbol", "t_ms", "bar", *PRICES, "volume_quote", "traded", "funding", "eligible")


def load(paths: DataPaths, cfg: MarketsConfig) -> pl.DataFrame:
    """Panel de tous les marchés de la config (séries de ``data/yahoo.py``)."""
    missing = [a.ticker for a in cfg.assets if not paths.markets_daily(a.ticker).exists()]
    if missing:
        raise DataError(f"séries absentes : {', '.join(missing)} (data.yahoo fetch)")
    bars = pl.concat(
        pl.read_parquet(paths.markets_daily(a.ticker)).with_columns(
            pl.lit(a.ticker).alias("symbol")
        )
        for a in cfg.assets
    )
    return assemble(bars, cfg)
