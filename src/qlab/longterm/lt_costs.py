"""Gate de coûts long terme (LT.4) : une stratégie coûte-t-elle trop cher pour être testée ?

On rejoue les **poids** d'une stratégie (``signals``) avec sa politique de rééquilibrage
(``allocation``), jour par jour, en laissant les poids dériver avec les prix, **sans calculer
de performance** (le gate ne doit rien révéler du résultat). On mesure, à un palier de capital :

1. turnover annuel TO = Σ |Δw| / années ;
2. drag annuel = Σ |Δw_i| c_i / années, avec c_i = frais taker réels + ½ spread + slippage
   (+ conversion de devise), spread mesuré sinon hypothèse prudente de la config (signalée) ;
3. ordres rejetés par ``minNotional`` : échanges voulus sous δ_min = minNotional / V ;
4. rapport au seuil de la fiche : drag / edge minimal (``per_year``) ou coût d'un aller-retour
   moyen / edge minimal (``per_trade``). Au-delà de ``max_drag_edge_fraction`` : **non testée** ;
5. plus de ``max_rejected_share`` du **volume** voulu (Σ|Δw|) rejeté : **non réalisable** à ce
   palier (à petit capital, les rejets font baisser le turnover mesuré : sans cette règle, le
   verdict serait flatteur). En volume et non en nombre d'ordres : les nombreuses petites
   corrections de dérive rejetées pèsent peu (mesuré le 2026-09-26 : 79 % des ordres mais 25 %
   du volume pour ts_momentum à 50 €).

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
from qlab.exchange.snapshots import Snapshot
from qlab.longterm import allocation, lt_backtest
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
    symbols: Sequence[str] | None = None,
) -> dict[str, AssetCost]:
    """Coûts de ``symbols`` (défaut : paires tradées) au capital ``tier`` : coût d'un ordre
    = frais + impact de ``lt_backtest.pair_rules`` ; δ_min = minNotional / capital."""
    capital = Decimal(str(tier.capital_quote))
    rules = lt_backtest.pair_rules(config, snapshot, spreads, symbols)
    return {
        s: AssetCost(
            float(r.fee_frac + r.impact_frac),
            float(r.filters.min_notional / capital),
            s in spreads,
        )
        for s, r in rules.items()
    }


@dataclass(frozen=True, slots=True)
class CostResult:
    years: float
    turnover_annual: float
    drag_annual: float
    orders: int
    rejected: int
    mean_cost_frac: float  # coût moyen d'un aller simple, pondéré par le volume échangé
    rejected_turnover_annual: float  # Σ |Δw| voulu mais rejeté (minNotional), par an

    @property
    def rejected_share(self) -> float:
        """Part des ordres voulus rejetés (en nombre, pour information)."""
        wanted = self.orders + self.rejected
        return self.rejected / wanted if wanted else 0.0

    @property
    def rejected_volume_share(self) -> float:
        """Part du volume voulu rejeté : critère de réalisabilité."""
        wanted = self.turnover_annual + self.rejected_turnover_annual
        return self.rejected_turnover_annual / wanted if wanted else 0.0


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
    traded = cost = refused = 0.0
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
        refused += sum(abs(x.delta_w) for x in decision.skipped)
        cost += sum(abs(x.delta_w) * costs[x.asset].cost_frac for x in decision.trades)
        if t + 1 < len(dates) and held:
            r = dict(zip(assets, returns[t], strict=True))
            held = allocation.drift(held, {a: r[a] for a in held})
    years = len(dates) / periods_per_year
    return CostResult(
        years,
        traded / years,
        cost / years,
        orders,
        rejected,
        cost / traded if traded else 0.0,
        refused / years,
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    ratio: float  # part de l'edge minimal consommée par les coûts
    detail: str
    feasible: bool = True  # faux : trop d'ordres rejetés par minNotional à ce palier

    @property
    def label(self) -> str:
        if not self.feasible:
            return "**non réalisable**"
        return "testée" if self.passed else "**non testée**"


def gate(result: CostResult, hypothesis: Hypothesis, cfg: LtCostsConfig) -> Verdict:
    """Coûts comparés à l'edge minimal de la fiche (base annuelle ou par trade), puis part
    d'ordres rejetés par ``minNotional``."""
    edge = hypothesis.min_edge_frac
    if hypothesis.edge_basis == "per_year":
        ratio = result.drag_annual / edge
        detail = f"drag {result.drag_annual:.2%}/an pour un edge visé de {edge:.2%}/an"
    else:
        trip = 2 * result.mean_cost_frac
        ratio = trip / edge
        detail = f"aller-retour {trip:.2%} pour un edge visé de {edge:.2%}/trade"
    feasible = result.rejected_volume_share <= cfg.max_rejected_share
    return Verdict(
        feasible and ratio <= cfg.max_drag_edge_fraction,
        ratio,
        f"{detail} ({ratio:.0%} consommé), {result.rejected_volume_share:.0%} du volume rejeté",
        feasible,
    )


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
            f"{res.orders} | {res.rejected} ({res.rejected_share:.0%} des ordres, "
            f"{res.rejected_volume_share:.0%} du volume) | {v.ratio:.0%} | "
            f"{v.label} |"
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
            f"Seuils : coûts ≤ {cfg.max_drag_edge_fraction:.0%} de l'edge visé, ordres rejetés "
            f"≤ {cfg.max_rejected_share:.0%} du volume voulu. {spread} ; "
            f"slippage {cfg.slippage_frac:.2%}. δ_min calculé au capital du palier (V constant).",
        ]
    )
