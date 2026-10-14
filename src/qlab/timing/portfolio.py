"""Portefeuille long / short d'un signal : positions toutes les h barres, rendements nets.

Une **barre** est une ligne de la grille du panel (colonne ``bar`` : une heure pour les
contrats perpétuels, un jour ouvré pour les marchés de ``markets/daily_panel.py``). À chaque
barre de rééquilibrage ``T`` (``T`` multiple de h), parmi les actifs éligibles ayant un signal et un
prix d'entrée : achat des ``n_long`` plus forts signaux, vente à découvert des ``n_short``
plus faibles. Moins de candidats que de lignes ⇒ à plat.

- Portefeuille **signé** (``Book.signed``) : un achat exige un signal > 0, une vente un signal
  < 0 ; tout le marché peut être acheté, vendu ou laissé, selon le signe, et le nombre de
  lignes peut manquer.
- Poids **égaux** (``weighting="equal"``) : ``1 / (n_long + n_short)`` par ligne (exposition
  brute 1, une ligne manquante reste en cash). Poids **inverses de la volatilité**
  (``"inverse_vol"``) : ∝ 1 / σ, σ = écart-type des rendements log des ``vol_lookback``
  barres passées, normalisés à Σ|w| = 1 sur les lignes tenues. Jamais de levier.

Rendement d'une période, par ligne : ``w · (r − f)`` (``r`` : ``panel.forward_return``,
``f`` : financement réglé pendant la détention ; l'acheteur paie un taux positif, le vendeur
le reçoit), **plafonné à la perte de la mise** ``−|w|`` : à 1×, chaque ligne a sa marge
isolée ; une vente à découvert est liquidée si le plus haut de la détention double le prix
d'entrée (``panel.forward_peak``), quelle que soit la sortie. Coût : ``|w_T − w_{T−h}| · c``
par actif, ``c`` = coût d'une jambe (``costs/cost_model.one_way_taker`` : frais + ½ spread +
glissement). La dérive des poids pendant la période est ignorée (approximation usuelle).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from qlab.core.errors import DataError
from qlab.costs.cost_model import one_way_taker
from qlab.timing.panel import forward_funding, forward_peak, forward_return


@dataclass(frozen=True, slots=True)
class Book:
    """Règles du portefeuille : horizon (h, en barres), lignes de chaque côté, coût d'une
    jambe par actif (``default_cost`` pour un actif sans mesure) ; ``signed``, ``weighting`` et
    ``vol_lookback`` : voir le module."""

    horizon: int
    n_long: int
    n_short: int
    costs: Mapping[str, float]
    default_cost: float
    signed: bool = False
    weighting: Literal["equal", "inverse_vol"] = "equal"
    vol_lookback: int = 60

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.n_long < 0 or self.n_short < 0 or self.vol_lookback < 2:
            raise DataError("portefeuille : horizon ≥ 1, lignes ≥ 0, volatilité sur ≥ 2 barres")
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


def rebalance_times(panel: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """(``bar``, ``t_ms``) des barres de rééquilibrage couvertes par le panel (même à plat)."""
    if panel.is_empty():
        raise DataError("portefeuille : panel vide")
    return (
        panel.select("bar", "t_ms").unique("bar").filter(pl.col("bar") % horizon == 0).sort("bar")
    )


def _volatility(lookback: int) -> pl.Expr:
    return pl.col("close").log().diff().rolling_std(lookback).over("symbol")


def weights(panel: pl.DataFrame, signal: pl.Series, book: Book) -> pl.DataFrame:
    """(``bar``, ``t_ms``, ``symbol``, ``w``, ``r``, ``f``, ``pnl``, ``liquidated``) des lignes
    tenues à chaque rééquilibrage."""
    h, lines = book.horizon, book.n_long + book.n_short
    cand = panel.select(
        "bar",
        "t_ms",
        "symbol",
        "eligible",
        signal.alias("s"),
        forward_return(h).alias("r"),
        forward_funding(h).alias("f"),
        forward_peak(h).alias("peak"),
        *([_volatility(book.vol_lookback).alias("vol")] if book.weighting == "inverse_vol" else []),
    ).filter(
        (pl.col("bar") % h == 0)
        & pl.col("eligible")
        & pl.col("s").is_not_null()
        & pl.col("s").is_finite()
        & pl.col("r").is_not_null()
    )
    if book.weighting == "inverse_vol":
        cand = cand.filter(pl.col("vol") > 0)
    cand = cand.with_columns(
        pl.col("s").rank("ordinal", descending=True).over("bar").alias("top"),
        pl.col("s").rank("ordinal").over("bar").alias("bottom"),
        pl.len().over("bar").alias("n"),
    )
    long_ok, short_ok = pl.col("top") <= book.n_long, pl.col("bottom") <= book.n_short
    if book.signed:
        long_ok, short_ok = long_ok & (pl.col("s") > 0), short_ok & (pl.col("s") < 0)
    else:
        cand = cand.filter(pl.col("n") >= lines)
    side = pl.when(long_ok).then(1.0).when(short_ok).then(-1.0).otherwise(0.0)
    held = cand.with_columns(side.alias("side")).filter(pl.col("side") != 0)
    if book.weighting == "inverse_vol":
        risk = 1 / pl.col("vol")
        w = pl.col("side") * risk / risk.sum().over("bar")
    else:
        w = pl.col("side") / lines
    held = held.with_columns(w.alias("w"))
    liquidated = (pl.col("w") < 0) & (pl.col("peak") >= 1)
    stake = -pl.col("w").abs()
    capped = pl.max_horizontal(pl.col("w") * (pl.col("r") - pl.col("f")), stake)
    pnl = pl.when(liquidated).then(stake).otherwise(capped)
    held = held.with_columns(pnl.alias("pnl"), liquidated.alias("liquidated"))
    columns = ("bar", "t_ms", "symbol", "w", "r", "f", "pnl", "liquidated")
    return held.select(columns).sort("bar", "symbol")


def period_returns(panel: pl.DataFrame, signal: pl.Series, book: Book) -> pl.DataFrame:
    """Une ligne par rééquilibrage (``bar``, ``t_ms``) : ``gross``, ``funding``, ``cost``,
    ``net``, ``turnover``, ``lines`` (0 = à plat), ``liquidations``. Période à plat : 0, moins
    le coût de fermeture éventuel."""
    held = weights(panel, signal, book)
    prev = held.select(
        (pl.col("bar") + book.horizon).alias("bar"), "symbol", pl.col("w").alias("w0")
    )
    trades = held.select("bar", "symbol", "w").join(
        prev, on=["bar", "symbol"], how="full", coalesce=True
    )
    cost_of = pl.col("symbol").replace_strict(book.costs, default=book.default_cost)
    trades = trades.with_columns(
        (pl.col("w").fill_null(0.0) - pl.col("w0").fill_null(0.0)).abs().alias("dw")
    ).with_columns((pl.col("dw") * cost_of).alias("c"))
    per_trade = trades.group_by("bar").agg(
        pl.col("c").sum().alias("cost"), pl.col("dw").sum().alias("turnover")
    )
    per_hold = held.group_by("bar").agg(
        pl.col("pnl").sum().alias("pnl"),
        (pl.col("w") * pl.col("f")).sum().alias("funding"),
        pl.len().alias("lines"),
        pl.col("liquidated").sum().alias("liquidations"),
    )
    grid = rebalance_times(panel, book.horizon)
    out = grid.join(per_hold, on="bar", how="left").join(per_trade, on="bar", how="left")
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
        .sort("bar")
    )


def buy_and_hold(panel: pl.DataFrame, symbol: str, horizon: int) -> pl.DataFrame:
    """(``bar``, ``t_ms``, ``bench``) : rendement de la détention simple de ``symbol`` sur
    chaque période (sans coûts) ; 0 avant sa cotation."""
    own = panel.filter(pl.col("symbol") == symbol).select(
        "bar", forward_return(horizon).alias("bench")
    )
    grid = rebalance_times(panel, horizon)
    return grid.join(own, on="bar", how="left").with_columns(pl.col("bench").fill_null(0.0))


def basket(panel: pl.DataFrame, horizon: int, vol_lookback: int) -> pl.DataFrame:
    """(``bar``, ``t_ms``, ``bench``) : achat de **tous** les actifs éligibles, pondérés par
    l'inverse de leur volatilité (Σ w = 1), sans coûts — la référence passive diversifiée."""
    book = Book(horizon, panel["symbol"].n_unique(), 0, {}, 0.0, True, "inverse_vol", vol_lookback)
    out = period_returns(panel, pl.Series([1.0] * panel.height), book)
    return out.select("bar", "t_ms", pl.col("gross").alias("bench"))
