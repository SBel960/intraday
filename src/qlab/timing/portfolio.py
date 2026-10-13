"""Portefeuille long / short d'un signal : positions toutes les h heures, rendements nets.

À chaque heure de rééquilibrage ``T`` (``T`` multiple de h heures depuis l'époque), parmi les
contrats éligibles ayant un signal et un prix d'entrée : achat des ``n_long`` plus forts
signaux, vente à découvert des ``n_short`` plus faibles, poids égaux de ``1 / (n_long +
n_short)`` (exposition brute 1 : levier 1×). Moins de candidats que de lignes ⇒ à plat.

Rendement d'une période, par ligne : ``w · (r − f)`` (``r`` : ``panel.forward_return``,
``f`` : financement réglé pendant la détention ; l'acheteur paie un taux positif, le vendeur
le reçoit), **plafonné à la perte de la mise** ``−|w|`` : à 1×, chaque ligne a sa marge
isolée ; une vente à découvert est liquidée si le plus haut de la détention double le prix
d'entrée (``panel.forward_peak``), quelle que soit la sortie. Coût : ``|w_T − w_{T−h}| · c``
par contrat, ``c`` = coût d'une jambe taker (``costs/cost_model.one_way_taker`` : frais + ½
spread + glissement). La dérive des poids
pendant la période est ignorée (approximation usuelle, h ≤ quelques jours).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_HOUR
from qlab.costs.cost_model import one_way_taker
from qlab.timing.panel import forward_funding, forward_peak, forward_return


@dataclass(frozen=True, slots=True)
class Book:
    """Règles du portefeuille : horizon (h), lignes de chaque côté, coût d'une jambe par
    contrat (``default_cost`` pour un contrat sans mesure)."""

    horizon_h: int
    n_long: int
    n_short: int
    costs: Mapping[str, float]
    default_cost: float

    def __post_init__(self) -> None:
        if self.horizon_h < 1 or self.n_long < 0 or self.n_short < 0:
            raise DataError("portefeuille : horizon ≥ 1 h et nombres de lignes ≥ 0")
        if self.n_long + self.n_short == 0:
            raise DataError("portefeuille : au moins une ligne")


def leg_costs(
    symbols: list[str],
    spreads: Mapping[str, float],
    *,
    taker_frac: float,
    fallback_spread_frac: float,
    slippage_frac: float,
) -> dict[str, float]:
    """Coût d'une jambe taker par contrat ; spread mesuré, sinon l'hypothèse de repli."""
    spread = np.array([spreads.get(s, fallback_spread_frac) for s in symbols])
    leg = one_way_taker(
        np.full(len(symbols), taker_frac), spread, np.full(len(symbols), slippage_frac)
    )
    return dict(zip(symbols, leg.tolist(), strict=True))


def rebalance_times(panel: pl.DataFrame, horizon_h: int) -> pl.Series:
    """Heures de rééquilibrage couvertes par le panel (grille complète, même à plat)."""
    step = horizon_h * MS_PER_HOUR
    lo, hi = panel["t_ms"].min(), panel["t_ms"].max()
    if not isinstance(lo, int) or not isinstance(hi, int):
        raise DataError("portefeuille : panel vide")
    first = -(-lo // step) * step
    return pl.Series("t_ms", range(first, hi + 1, step), dtype=pl.Int64)


def weights(panel: pl.DataFrame, signal: pl.Series, book: Book) -> pl.DataFrame:
    """(``t_ms``, ``symbol``, ``w``, ``r``, ``f``) des lignes tenues à chaque rééquilibrage."""
    h, lines = book.horizon_h, book.n_long + book.n_short
    step = h * MS_PER_HOUR
    cand = (
        panel.select(
            "t_ms",
            "symbol",
            "eligible",
            signal.alias("s"),
            forward_return(h).alias("r"),
            forward_funding(h).alias("f"),
            forward_peak(h).alias("peak"),
        )
        .filter(
            (pl.col("t_ms") % step == 0)
            & pl.col("eligible")
            & pl.col("s").is_not_null()
            & pl.col("s").is_finite()
            & pl.col("r").is_not_null()
        )
        .with_columns(
            pl.col("s").rank("ordinal", descending=True).over("t_ms").alias("top"),
            pl.col("s").rank("ordinal").over("t_ms").alias("bottom"),
            pl.len().over("t_ms").alias("n"),
        )
        .filter(pl.col("n") >= lines)
    )
    side = (
        pl.when(pl.col("top") <= book.n_long)
        .then(1.0)
        .when(pl.col("bottom") <= book.n_short)
        .then(-1.0)
        .otherwise(0.0)
    )
    held = cand.with_columns((side / lines).alias("w")).filter(pl.col("w") != 0)
    liquidated = (pl.col("w") < 0) & (pl.col("peak") >= 1)
    stake = -pl.col("w").abs()
    capped = pl.max_horizontal(pl.col("w") * (pl.col("r") - pl.col("f")), stake)
    pnl = pl.when(liquidated).then(stake).otherwise(capped)
    held = held.with_columns(pnl.alias("pnl"), liquidated.alias("liquidated"))
    return held.select("t_ms", "symbol", "w", "r", "f", "pnl", "liquidated").sort("t_ms", "symbol")


def period_returns(panel: pl.DataFrame, signal: pl.Series, book: Book) -> pl.DataFrame:
    """Une ligne par rééquilibrage : ``gross``, ``funding``, ``cost``, ``net``, ``turnover``,
    ``lines`` (0 = à plat), ``liquidations``. Période à plat : 0, moins le coût de fermeture
    éventuel."""
    held = weights(panel, signal, book)
    step = book.horizon_h * MS_PER_HOUR
    prev = held.select((pl.col("t_ms") + step).alias("t_ms"), "symbol", pl.col("w").alias("w0"))
    trades = held.select("t_ms", "symbol", "w").join(
        prev, on=["t_ms", "symbol"], how="full", coalesce=True
    )
    cost_of = pl.col("symbol").replace_strict(book.costs, default=book.default_cost)
    trades = trades.with_columns(
        (pl.col("w").fill_null(0.0) - pl.col("w0").fill_null(0.0)).abs().alias("dw")
    ).with_columns((pl.col("dw") * cost_of).alias("c"))
    per_trade = trades.group_by("t_ms").agg(
        pl.col("c").sum().alias("cost"), pl.col("dw").sum().alias("turnover")
    )
    per_hold = held.group_by("t_ms").agg(
        pl.col("pnl").sum().alias("pnl"),
        (pl.col("w") * pl.col("f")).sum().alias("funding"),
        pl.len().alias("lines"),
        pl.col("liquidated").sum().alias("liquidations"),
    )
    grid = rebalance_times(panel, book.horizon_h).to_frame()
    out = grid.join(per_hold, on="t_ms", how="left").join(per_trade, on="t_ms", how="left")
    out = out.with_columns(
        pl.col("pnl", "funding", "cost", "turnover").fill_null(0.0),
        pl.col("lines", "liquidations").fill_null(0).cast(pl.Int64),
    )
    # brut = résultat des prix (pnl, financement compris, plafonné à la mise) hors financement
    return (
        out.with_columns(
            (pl.col("pnl") + pl.col("funding")).alias("gross"),
            (pl.col("pnl") - pl.col("cost")).alias("net"),
        )
        .drop("pnl")
        .sort("t_ms")
    )


def buy_and_hold(panel: pl.DataFrame, symbol: str, horizon_h: int) -> pl.DataFrame:
    """Rendement de la détention simple de ``symbol`` sur chaque période (sans coûts) ;
    0 avant sa cotation."""
    own = panel.filter(pl.col("symbol") == symbol).select(
        "t_ms", forward_return(horizon_h).alias("bench")
    )
    grid = rebalance_times(panel, horizon_h).to_frame()
    return grid.join(own, on="t_ms", how="left").with_columns(pl.col("bench").fill_null(0.0))
