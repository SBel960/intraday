"""LE COST GATE (SPEC_INTRADAY §5) : un signal intraday est-il seulement *possible* après coûts ?

Entrée : cotations top-of-book d'une paire (``ts_ms`` epoch ms, ``bid``, ``ask``), triées.
Pour chaque tranche horaire UTC (``slot_minutes``) et au total :

1. distribution du coût aller-retour taker  c = 2 f_taker + s̃ + 2 slip  (``cost_model``) ;
2. mouvement absolu médian du mid  |ln(m(t+h) / m(t))|  à chaque horizon h ;
3. ratio  mouvement médian / coût médian ;
4. **plus petit horizon** où ce ratio dépasse ``min_move_cost_ratio`` (3).

Échantillonnage sur une **grille régulière** (``sample_step_ms``) : chaque seconde pèse autant,
qu'il y ait beaucoup de cotations ou peu. Le mid en t est la dernière cotation ≤ t ; si elle a
plus de ``max_staleness_ms`` (``intraday.risk.freshness_max_ms``), le point est un **trou** :
exclu, jamais interpolé. Cotations invalides (croisées, ≤ 0, non finies) : exclues et comptées.

Si aucun horizon ne passe, le verdict le dit : aucune stratégie intraday n'est testée.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from qlab.core.config import QlabConfig
from qlab.core.errors import DataError
from qlab.core.timeutils import floor_ms, slot_of_day_expr
from qlab.costs.cost_model import round_trip_taker, valid_quotes

ALL_SLOTS = -1  # ligne « toutes tranches confondues » dans la table de résultats


@dataclass(frozen=True, slots=True)
class GateParams:
    """Paramètres du gate ; ``fee_taker`` : frais réels de la paire (``effective_params``)."""

    fee_taker: float
    slippage: float
    horizons_s: tuple[int, ...]
    min_ratio: float
    slot_minutes: int
    step_ms: int
    max_staleness_ms: int

    @classmethod
    def from_config(cls, config: QlabConfig, fee_taker: float) -> GateParams:
        gate = config.intraday.gate
        return cls(
            fee_taker,
            gate.slippage_frac,
            gate.horizons_s,
            gate.min_move_cost_ratio,
            gate.slot_minutes,
            gate.sample_step_ms,
            config.intraday.risk.freshness_max_ms,
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    """``table`` : une ligne par (tranche, horizon) + tranche ``ALL_SLOTS`` ; colonnes ``slot,
    horizon_s, n, cost_median, cost_p90, move_median, ratio``. ``min_horizon_s`` : tranche →
    plus petit horizon qui passe (``None`` : aucun)."""

    table: pl.DataFrame
    min_horizon_s: dict[int, int | None]
    quotes_used: int
    quotes_excluded: int
    min_ratio: float

    @property
    def overall_min_horizon_s(self) -> int | None:
        return self.min_horizon_s.get(ALL_SLOTS)

    def verdict(self) -> str:
        h = self.overall_min_horizon_s
        if h is None:
            longest = max(self.table["horizon_s"].to_list(), default=0)
            return (
                f"Aucun horizon ≤ {longest} s ne dépasse le ratio {self.min_ratio:g} : "
                "aucune stratégie intraday n'est testée (§5)."
            )
        return f"Plus petit horizon rentable (ratio > {self.min_ratio:g}) : {h} s."


def _clean_quotes(quotes: pl.DataFrame, p: GateParams) -> tuple[pl.DataFrame, int]:
    """Cotations valides avec ``mid`` et ``cost`` ; renvoie aussi le nombre d'exclues."""
    missing = {"ts_ms", "bid", "ask"} - set(quotes.columns)
    if missing:
        raise DataError(f"cotations : colonnes manquantes {sorted(missing)}")
    if quotes.height == 0:
        raise DataError("cotations : aucune ligne")
    ts = quotes["ts_ms"]
    if ts.dtype != pl.Int64 or ts.null_count() or not ts.is_sorted():
        raise DataError("cotations : ts_ms doit être Int64, sans null, trié par ordre croissant")
    bid = quotes["bid"].cast(pl.Float64).to_numpy()
    ask = quotes["ask"].cast(pl.Float64).to_numpy()
    ok = valid_quotes(bid, ask)
    bid, ask = bid[ok], ask[ok]
    mid = (bid + ask) / 2
    cost = round_trip_taker(p.fee_taker, (ask - bid) / mid, p.slippage)
    clean = pl.DataFrame({"q_ts": ts.to_numpy()[ok], "mid": mid, "cost": cost})
    return clean, int((~ok).sum())


def _at(
    grid: pl.DataFrame, clean: pl.DataFrame, offset_ms: int, max_staleness_ms: int, suffix: str
) -> pl.DataFrame:
    """Dernière cotation ≤ t + offset ; ``mid``/``cost`` nuls si elle est trop vieille (trou)."""
    target = grid.with_columns((pl.col("t") + offset_ms).alias("target"))
    joined = target.join_asof(clean, left_on="target", right_on="q_ts", strategy="backward")
    fresh = (pl.col("target") - pl.col("q_ts")) <= max_staleness_ms
    return joined.select(
        "t",
        pl.when(fresh).then(pl.col("mid")).alias(f"mid{suffix}"),
        pl.when(fresh).then(pl.col("cost")).alias(f"cost{suffix}"),
    )


def _samples(clean: pl.DataFrame, p: GateParams) -> pl.DataFrame:
    """Une ligne par (instant de la grille, horizon) valide : ``slot, horizon_s, cost, move``."""
    first = floor_ms(int(clean["q_ts"][0]), p.step_ms)
    last = int(clean["q_ts"][-1])
    grid = pl.DataFrame({"t": np.arange(first, last + 1, p.step_ms, dtype=np.int64)})
    now = _at(grid, clean, 0, p.max_staleness_ms, "0")
    parts = []
    for h in p.horizons_s:
        later = _at(grid, clean, h * 1000, p.max_staleness_ms, "h").select("t", "midh")
        parts.append(
            now.join(later, on="t")
            .filter(pl.col("t") + h * 1000 <= last)  # pas d'horizon au-delà des données
            .drop_nulls(["mid0", "cost0", "midh"])
            .select(
                slot_of_day_expr(pl.col("t"), p.slot_minutes).cast(pl.Int64).alias("slot"),
                pl.lit(h, dtype=pl.Int64).alias("horizon_s"),
                pl.col("cost0").alias("cost"),
                (pl.col("midh") / pl.col("mid0")).log().abs().alias("move"),
            )
        )
    return pl.concat(parts)


def _aggregate(samples: pl.DataFrame) -> pl.DataFrame:
    stats = [
        pl.len().cast(pl.Int64).alias("n"),
        pl.col("cost").median().alias("cost_median"),
        pl.col("cost").quantile(0.9, interpolation="nearest").alias("cost_p90"),
        pl.col("move").median().alias("move_median"),
    ]
    by_slot = samples.group_by("slot", "horizon_s").agg(stats)
    overall = (
        samples.group_by("horizon_s")
        .agg(stats)
        .with_columns(pl.lit(ALL_SLOTS, dtype=pl.Int64).alias("slot"))
    )
    table = pl.concat([by_slot, overall.select(by_slot.columns)])
    return table.with_columns((pl.col("move_median") / pl.col("cost_median")).alias("ratio")).sort(
        "slot", "horizon_s"
    )


def run_gate(quotes: pl.DataFrame, params: GateParams) -> GateResult:
    """Exécute le gate sur les cotations d'une paire (fonction pure, déterministe)."""
    clean, excluded = _clean_quotes(quotes, params)
    if clean.height == 0:
        raise DataError(f"cotations : aucune valide ({excluded} exclues)")
    samples = _samples(clean, params)
    if samples.height == 0:
        raise DataError("pas assez de données pour le plus petit horizon (ou uniquement des trous)")
    table = _aggregate(samples)
    passing = table.filter(pl.col("ratio") > params.min_ratio)
    min_h: dict[int, int | None] = dict.fromkeys(table["slot"].unique().to_list())
    for slot, horizon in passing.group_by("slot").agg(pl.col("horizon_s").min()).iter_rows():
        min_h[slot] = horizon
    return GateResult(table, min_h, clean.height, excluded, params.min_ratio)
