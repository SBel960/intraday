"""Évaluation d'un essai long terme : rendements, références, stabilité, verdict rédigé.

Utilisé par ``lt_wave`` (orchestration de la vague). Pour chaque essai :

- **univers** (``panel``) : les paires tradées, ou l'univers ``trade_eur`` pour une fiche qui
  le déclare (paires EUR liquides point-in-time, membres jour par jour) ;
- **fenêtre** : à partir de la première décision d'investissement (``first_decision``) : la
  chauffe des signaux est exclue, pour l'essai comme pour ses références ;
- **références** : buy & hold du panier équipondéré des paires tradées (même moteur, mêmes
  coûts) — pour un univers qui change, panier équipondéré des membres du jour, rééquilibré
  au rythme de la fiche — et DCA (TWR et MWR) ;
- **actifs** : la même règle sur chaque actif seul, sur chaque sous-panier de 2 pour une
  fiche multi-actifs, ou sur deux moitiés disjointes de l'univers ``trade_eur`` ;
- **oracle** : contrôle anti-fuite du pipeline (``lt_backtest.oracle_weights``).
"""

from __future__ import annotations

import dataclasses
import itertools
from decimal import Decimal

import numpy as np
import polars as pl

from qlab.core.config import CapitalTier, QlabConfig
from qlab.core.errors import DataError
from qlab.core.timeutils import date_str
from qlab.exchange.snapshots import Snapshot
from qlab.longterm import lt_backtest, lt_report, strategies
from qlab.longterm import signals as sg
from qlab.longterm.allocation import Policy
from qlab.research import report as research_report
from qlab.research.hypothesis import Hypothesis
from qlab.research.trials import TrialResult

BUY_HOLD = Policy("buy_hold")
NOTES = (
    "Filtres d'ordre (pas, minimum) du snapshot actuel appliqués à tout l'historique.",
    "Buy & hold : achat unique au départ ; une paire cotée plus tard n'y entre pas.",
)
Kept = tuple[Hypothesis, dict[str, float], Policy, pl.DataFrame]


@dataclasses.dataclass(frozen=True, slots=True)
class Setup:
    """Tout ce qu'il faut pour rejouer les essais à un palier de capital."""

    config: QlabConfig
    snapshot: Snapshot
    market: strategies.Market
    opens: pl.DataFrame  # ouvertures des paires tradées, grille de ``market.closes``
    rules: dict[str, lt_backtest.PairRules]  # paires tradées et univers trade_eur
    tier: CapitalTier
    spreads: dict[str, float]  # médianes mesurées (``exchange/spreads.py``)

    @property
    def capital(self) -> Decimal:
        return Decimal(str(self.tier.capital_quote))


def panel(setup: Setup, hypothesis: Hypothesis) -> strategies.Panel:
    """Univers de la fiche : paires tradées, ou ``trade_eur``."""
    if hypothesis.universe != "trade_eur":
        return strategies.Panel(setup.market.closes, setup.opens)
    if setup.market.quoted is None:
        raise DataError(f"{hypothesis.id} : univers trade_eur non chargé")
    return setup.market.quoted


def returns(
    setup: Setup, weights: pl.DataFrame, policy: Policy, closes: pl.DataFrame, opens: pl.DataFrame
) -> np.ndarray:
    """Rendements TWR du backtest (ouvertures ramenées à la grille et aux paires de ``closes``)."""
    opens = opens.join(closes.select(sg.DATE), on=sg.DATE, how="semi").select(closes.columns)
    res = lt_backtest.run(weights, opens, closes, policy, setup.rules, initial_quote=setup.capital)
    return res.twr_returns()


def benchmark(setup: Setup, hypothesis: Hypothesis, universe: strategies.Panel) -> np.ndarray:
    """Buy & hold équipondéré ; univers qui change : membres du jour au rythme de la fiche."""
    weights = sg.equal_weight(universe.closes, universe.members)
    policy = BUY_HOLD if universe.members is None else strategies.policy_for(hypothesis)
    return returns(setup, weights, policy, universe.closes, universe.opens)


def first_decision(weights: pl.DataFrame) -> int:
    """Indice du premier jour où la stratégie veut être investie : la chauffe de ses signaux
    est exclue de l'évaluation, pour elle comme pour ses références (le rendement d'indice t
    va du jour t au jour t + 1 : il porte la première exécution)."""
    invested = weights.select(pl.sum_horizontal(pl.exclude(sg.DATE)) > 0)[:, 0].to_numpy()
    return int(invested.argmax()) if invested.any() else 0  # jamais investi : toute la période


def _groups(setup: Setup, hypothesis: Hypothesis) -> list[tuple[str, strategies.Market]]:
    """Sous-marchés pour le critère « actifs » (voir l'en-tête du module)."""
    market = setup.market
    if hypothesis.universe == "trade_eur" and market.quoted is not None:
        q = market.quoted
        symbols = [c for c in q.closes.columns if c != sg.DATE]
        halves = (symbols[0::2], symbols[1::2])
        return [
            (f"moitié {i + 1}", dataclasses.replace(market, quoted=_subpanel(q, half)))
            for i, half in enumerate(halves)
        ]
    assets = [c for c in market.closes.columns if c != sg.DATE]
    size = 2 if strategies.strategy_for(hypothesis).multi_asset else 1
    out = []
    for group in itertools.combinations(assets, size):
        both = market.closes.select(sg.DATE, *group)
        dates = both[sg.DATE].to_numpy()
        start = max(int(dates[both[a].is_not_null().to_numpy()].min()) for a in group)
        closes = both.filter(pl.col(sg.DATE) >= start)  # grille complète depuis que tous existent
        out.append(("+".join(group), dataclasses.replace(market, closes=closes)))
    return out


def _subpanel(q: strategies.Panel, symbols: list[str]) -> strategies.Panel:
    members = None if q.members is None else q.members.select(sg.DATE, *symbols)
    return strategies.Panel(q.closes.select(sg.DATE, *symbols), q.opens, members)


def per_asset(
    setup: Setup, hypothesis: Hypothesis, params: dict[str, float], policy: Policy
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """La même règle sur chaque sous-marché, face à sa référence, après sa chauffe."""
    out = {}
    for label, sub in _groups(setup, hypothesis):
        sub_setup = dataclasses.replace(setup, market=sub)
        universe = panel(sub_setup, hypothesis)
        weights = strategies.strategy_for(hypothesis).build(sub, params)
        first = first_decision(weights)
        out[label] = (
            returns(setup, weights, policy, universe.closes, universe.opens)[first:],
            benchmark(setup, hypothesis, universe)[first:],
        )
    return out


def trial_result(r: np.ndarray, ppy: int) -> TrialResult:
    """Moments de l'essai ; jamais investi (rendements constants) ⇒ Sharpe 0 par convention
    (loi normale : asymétrie 0, kurtosis 3) : l'essai compte quand même dans N."""
    if np.ptp(r) == 0:
        return TrialResult(0.0, r.size, 0.0, 3.0, ppy)
    return lt_report.trial_result(r, ppy)


def criteria(config: QlabConfig) -> research_report.Criteria:
    a = config.longterm.acceptance
    return research_report.Criteria(
        a.dsr_min,
        a.confidence,
        a.min_subperiods,
        a.min_assets,
        a.require_bear,
        a.n_boot,
        a.mean_block_days,
        a.seed,
    )


def judge(setup: Setup, trial: Kept, r: np.ndarray, n_trials: int, variance: float) -> str:
    """Verdict rédigé d'un essai sur sa fenêtre d'évaluation ; non évaluable ⇒ dit pourquoi."""
    h, params, policy, weights = trial
    universe, first = panel(setup, h), first_decision(weights)
    name, ppy = strategies.trial_name(h, params), setup.market.days_per_year
    dates = universe.closes[sg.DATE].to_list()
    labels = lt_report.year_labels(dates)[first:]
    window = f"Évaluation du {date_str(dates[first])} au {date_str(dates[-1])} (chauffe exclue)."
    try:
        bench = benchmark(setup, h, universe)[first:]
        evidence = research_report.Evidence(
            name, r, bench, labels, per_asset(setup, h, params, policy), n_trials, variance, ppy
        )
        report = research_report.evaluate(evidence, criteria(setup.config))
        metrics = (lt_report.metrics(r, ppy), lt_report.metrics(bench, ppy))
        dca = _dca(setup, dates[first])
    except DataError as e:  # ex. jamais investi : Sharpe indéfini
        return f"## {name}\n\n**Verdict : {research_report.REJECTED}** (non évaluable : {e})\n"
    notes = [window, spread_note(setup), *NOTES]
    return lt_report.render(report, *metrics, dca, notes)


def spread_note(setup: Setup) -> str:
    cfg = setup.config.longterm.costs
    measured = [s for s in sorted(setup.rules) if s in setup.spreads]
    assumed = [s for s in sorted(setup.rules) if s not in setup.spreads]
    parts = []
    if measured:
        n = cfg.spread_min_samples
        parts.append(f"spreads mesurés (médiane ≥ {n} relevés) : {', '.join(measured)}")
    if assumed:
        parts.append(f"spread supposé {cfg.fallback_spread_frac:.2%} : {', '.join(assumed)}")
    return "Coûts : " + " ; ".join(parts) + " (spreads actuels appliqués à tout l'historique)."


def oracle_check(setup: Setup, bench: np.ndarray) -> str:
    """L'oracle (``lt_backtest.oracle_weights``) doit multiplier le buy & hold (≥ × 2)."""
    daily = Policy("calendar", period_days=1)
    weights = lt_backtest.oracle_weights(setup.opens)
    oracle = returns(setup, weights, daily, setup.market.closes, setup.opens)
    ratio = float(np.prod(1 + oracle) / np.prod(1 + bench))
    suspect = "**SUSPECT : le moteur ne récompense pas la connaissance de l'avenir**"
    verdict = "ok" if ratio > 2 else suspect
    return f"Contrôle anti-fuite du pipeline (oracle / buy & hold = {ratio:.3g}) : {verdict}"


def _dca(setup: Setup, start_ms: int) -> tuple[lt_report.Metrics, float]:
    """DCA de référence (paires tradées) sur la même fenêtre que l'essai (dès ``start_ms``)."""
    cfg = setup.config.longterm.dca
    since = pl.col(sg.DATE) >= start_ms
    closes, opens = setup.market.closes.filter(since), setup.opens.filter(since)
    dates = closes[sg.DATE].to_list()
    flows = lt_report.dca_flows(dates, Decimal(str(cfg.amount_quote)), cfg.period_days)
    policy = Policy("calendar", period_days=cfg.period_days)
    res = lt_backtest.run(
        sg.equal_weight(closes),
        opens,
        closes,
        policy,
        setup.rules,
        initial_quote=Decimal(0),
        flows=flows,
    )
    ppy = setup.market.days_per_year
    return lt_report.metrics(res.twr_returns(), ppy), lt_report.mwr(res.flows, res.equity[-1], ppy)
