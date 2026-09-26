"""Gate de coûts long terme (LT.4) : une stratégie coûte-t-elle trop cher pour être testée ?

On rejoue les **poids** d'une stratégie (``signals``) avec sa politique de rééquilibrage
(``allocation``), jour par jour, en laissant les poids dériver avec les prix, **sans calculer
de performance** (le gate ne doit rien révéler du résultat). On mesure, à un palier de capital :

1. turnover annuel TO = Σ |Δw| / années ;
2. drag annuel = Σ |Δw_i| c_i / années, avec c_i = frais taker réels + ½ spread + slippage
   (+ conversion de devise), spread mesuré sinon hypothèse prudente de la config (signalée) ;
3. ordres rejetés par ``minNotional`` : échanges voulus sous δ_min = minNotional / V ;
4. rapport au seuil de la fiche : drag / edge minimal (``per_year``) ou coût d'un aller-retour
   moyen / edge minimal (``per_trade``). Au-delà de ``max_drag_edge_fraction`` : **non testée**.

Simplification assumée (écrite dans le rapport) : V reste égal au capital du palier pour δ_min
(le gate est un filtre préalable ; le backtest suit la vraie valeur). Un actif sans prix un jour
donné (trou, retrait) garde sa valeur de la veille ce jour-là.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import polars as pl

from qlab.core.config import CapitalTier, LtCostsConfig, QlabConfig
from qlab.core.errors import DataError
from qlab.exchange import effective_params
from qlab.exchange.snapshots import Snapshot
from qlab.longterm import allocation
from qlab.longterm.signals import DATE
from qlab.research.hypothesis import Hypothesis


@dataclass(frozen=True, slots=True)
class AssetCost:
    """Coût d'un ordre (fraction, aller simple) et δ_min (fraction de V) d'une paire."""

    cost_frac: float
    delta_min: float
    spread_measured: bool


def tier_costs(
    config: QlabConfig,
    snapshot: Snapshot,
    tier: CapitalTier,
    spreads: Mapping[str, float],
) -> dict[str, AssetCost]:
    """Coûts des paires tradées au capital ``tier`` : frais réels et δ_min du snapshot
    (``effective_params``), spread mesuré si fourni, sinon ``fallback_spread_frac``."""
    cfg = config.longterm.costs
    capital = Decimal(str(tier.capital_quote))
    params = effective_params.compute(
        config, snapshot, at_ms=snapshot.fetched_ms, net_deposits_quote=capital
    )
    out = {}
    for pair in params.pairs:
        spread = spreads.get(pair.symbol, cfg.fallback_spread_frac)
        cost = float(pair.fee_taker_frac) + spread / 2 + cfg.slippage_frac
        out[pair.symbol] = AssetCost(
            cost + cfg.eur_conversion_cost_frac, float(pair.delta_min_frac), pair.symbol in spreads
        )
    return out


@dataclass(frozen=True, slots=True)
class CostResult:
    years: float
    turnover_annual: float
    drag_annual: float
    orders: int
    rejected: int
    mean_cost_frac: float  # coût moyen d'un aller simple, pondéré par le volume échangé

    @property
    def rejected_share(self) -> float:
        wanted = self.orders + self.rejected
        return self.rejected / wanted if wanted else 0.0


def simulate(
    weights: pl.DataFrame,
    closes: pl.DataFrame,
    policy: allocation.Policy,
    costs: Mapping[str, AssetCost],
    periods_per_year: int,
) -> CostResult:
    """Rejoue ``weights`` (grille de ``signals``) sur ``closes`` (même grille)."""
    assets = [c for c in weights.columns if c != DATE]
    if weights[DATE].to_list() != closes[DATE].to_list() or not set(assets) <= set(closes.columns):
        raise DataError("poids et prix : même grille de dates et mêmes actifs attendus")
    missing = sorted(set(assets) - set(costs))
    if missing or periods_per_year < 1 or weights.height < 2:
        raise DataError(f"coûts inconnus {missing}, ou periods_per_year < 1, ou < 2 dates")
    w_target = weights.select(assets).to_numpy()
    prices = closes.select(assets).to_numpy()
    step = prices[1:] / prices[:-1] - 1
    returns = np.where(np.isfinite(step), step, 0.0)
    dates = weights[DATE].to_list()
    delta_min = {a: costs[a].delta_min for a in assets}
    held: dict[str, float] = {}
    traded = cost = 0.0
    orders = rejected = 0
    for t, day in enumerate(dates):
        target = {a: float(x) for a, x in zip(assets, w_target[t], strict=True) if x > 0}
        decision = allocation.decide(
            policy,
            date_ms=day,
            anchor_ms=dates[0],
            current=held,
            target=target,
            delta_min=delta_min,
        )
        held = allocation.apply(held, decision)
        orders, rejected = orders + len(decision.trades), rejected + len(decision.skipped)
        traded += decision.turnover
        cost += sum(abs(x.delta_w) * costs[x.asset].cost_frac for x in decision.trades)
        if t + 1 < len(dates) and held:
            r = dict(zip(assets, returns[t], strict=True))
            held = allocation.drift(held, {a: r[a] for a in held})
    years = len(dates) / periods_per_year
    return CostResult(
        years, traded / years, cost / years, orders, rejected, cost / traded if traded else 0.0
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    ratio: float  # part de l'edge minimal consommée par les coûts
    detail: str


def gate(result: CostResult, hypothesis: Hypothesis, max_fraction: float) -> Verdict:
    """Compare les coûts à l'edge minimal de la fiche (base annuelle ou par trade)."""
    edge = hypothesis.min_edge_frac
    if hypothesis.edge_basis == "per_year":
        ratio = result.drag_annual / edge
        detail = f"drag {result.drag_annual:.2%}/an pour un edge visé de {edge:.2%}/an"
    else:
        ratio = 2 * result.mean_cost_frac / edge
        detail = (
            f"aller-retour {2 * result.mean_cost_frac:.2%} pour un edge visé de {edge:.2%}/trade"
        )
    return Verdict(ratio <= max_fraction, ratio, f"{detail} ({ratio:.0%} consommé)")


@dataclass(frozen=True, slots=True)
class Row:
    """Une ligne du rapport : essai (fiche + paramètres) × palier."""

    trial: str
    tier: str
    result: CostResult
    verdict: Verdict


def render(rows: Sequence[Row], cfg: LtCostsConfig, spread_measured: bool) -> str:
    """Tableau Markdown du gate LT ; rappelle les hypothèses et simplifications."""
    lines = [
        "| Essai | Palier | TO/an | Drag/an | Ordres | Rejetés (minNotional) | Coûts/edge "
        "| Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        res, v = r.result, r.verdict
        lines.append(
            f"| {r.trial} | {r.tier} | {res.turnover_annual:.1f} | {res.drag_annual:.2%} | "
            f"{res.orders} | {res.rejected} ({res.rejected_share:.0%}) | {v.ratio:.0%} | "
            f"{'testée' if v.passed else '**non testée**'} |"
        )
    spread = (
        "spreads mesurés"
        if spread_measured
        else f"spread supposé {cfg.fallback_spread_frac:.2%} (hypothèse prudente, non mesuré)"
    )
    return "\n".join(
        [
            *lines,
            "",
            f"Seuil : coûts ≤ {cfg.max_drag_edge_fraction:.0%} de l'edge visé. {spread} ; "
            f"slippage {cfg.slippage_frac:.2%}. δ_min calculé au capital du palier (V constant).",
        ]
    )
