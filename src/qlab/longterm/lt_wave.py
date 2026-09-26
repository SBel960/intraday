"""Verdict d'une vague de fiches long terme : gate de coûts, backtest, registre, critères.

Pour un palier de capital (``--tier``, capital de départ du backtest) :

1. chaque essai (fiche × paramètres) passe d'abord le **gate de coûts** (``lt_costs``) ; un
   essai écarté n'est jamais backtesté ;
2. les essais retenus sont backtestés (``lt_backtest``) et **tous enregistrés** dans le registre
   d'essais **avant** tout calcul de verdict : N et la dispersion des Sharpe du DSR comptent
   tout ce qui a été essayé dans le volet ;
3. chaque essai est comparé au **buy & hold** du panier équipondéré (même moteur, mêmes coûts)
   et au **DCA** (TWR et MWR) ; stabilité par année civile ; « actifs » : la même règle rejouée
   sur chaque actif seul, ou sur chaque sous-panier de 2 pour une fiche multi-actifs ;
4. contrôle anti-fuite du pipeline (LT.5), une fois par vague sur les vrais prix : un oracle
   qui connaît le rendement qu'il va détenir (``peek`` d'une règle construite sur l'avenir)
   doit écraser le buy & hold ; sinon le moteur est suspect et le rapport le dit en tête.
   (Un « tricheur » par essai n'est pas un bon test : connaître les poids de demain d'une
   fiche de faible volatilité ne dit rien des rendements) ;
5. critères et verdict (``research/report.py``), rapport ``reports/lt_wave_{date}.md``.

Commande : ``python -m qlab.longterm.lt_wave --config config run [--tier t0]
[--hypotheses hypotheses]``
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.config import CapitalTier, QlabConfig
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.timeutils import date_str, now_ms
from qlab.exchange.snapshots import Snapshot, SnapshotStore
from qlab.longterm import klines, lt_backtest, lt_costs, lt_report, strategies, universe
from qlab.longterm import signals as sg
from qlab.longterm.allocation import Policy
from qlab.research import report as research_report
from qlab.research.hypothesis import Hypothesis, load_all
from qlab.research.trials import TrialRegistry, TrialResult

BUY_HOLD = Policy("buy_hold")
NOTES = (
    "Filtres d'ordre (pas, minimum) du snapshot actuel appliqués à tout l'historique.",
    "Spread supposé (hypothèse prudente de la config), pas encore mesuré sur les paires EUR.",
    "Buy & hold : achat unique au départ ; une paire cotée plus tard n'y entre pas.",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Setup:
    """Tout ce qu'il faut pour rejouer les essais à un palier de capital."""

    config: QlabConfig
    snapshot: Snapshot
    market: strategies.Market
    opens: pl.DataFrame  # même grille que ``market.closes``
    rules: dict[str, lt_backtest.PairRules]
    tier: CapitalTier

    @property
    def capital(self) -> Decimal:
        return Decimal(str(self.tier.capital_quote))


def _returns(
    setup: Setup, weights: pl.DataFrame, policy: Policy, closes: pl.DataFrame
) -> np.ndarray:
    opens = setup.opens.join(closes.select(sg.DATE), on=sg.DATE, how="semi").select(closes.columns)
    res = lt_backtest.run(weights, opens, closes, policy, setup.rules, initial_quote=setup.capital)
    return res.twr_returns()


def _per_asset(
    setup: Setup, hypothesis: Hypothesis, params: dict[str, float], policy: Policy
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """La même règle sur chaque actif seul (ou sous-panier de 2), face à son buy & hold."""
    assets = [c for c in setup.market.closes.columns if c != sg.DATE]
    size = 2 if strategies.strategy_for(hypothesis).multi_asset else 1
    out = {}
    for group in itertools.combinations(assets, size):
        both = setup.market.closes.select(sg.DATE, *group)
        dates = both[sg.DATE].to_numpy()
        start = max(int(dates[both[a].is_not_null().to_numpy()].min()) for a in group)
        closes = both.filter(pl.col(sg.DATE) >= start)  # grille complète depuis que tous existent
        sub = dataclasses.replace(setup.market, closes=closes)
        weights = strategies.strategy_for(hypothesis).build(sub, params)
        first = first_decision(weights)
        out["+".join(group)] = (
            _returns(setup, weights, policy, closes)[first:],
            _returns(setup, sg.equal_weight(closes), BUY_HOLD, closes)[first:],
        )
    return out


def first_decision(weights: pl.DataFrame) -> int:
    """Indice du premier jour où la stratégie veut être investie : la chauffe de ses signaux
    (ex. un an de financement) est exclue de l'évaluation, pour elle comme pour ses références
    (le rendement d'indice t va du jour t au jour t + 1 : il porte la première exécution)."""
    invested = weights.select(pl.sum_horizontal(pl.exclude(sg.DATE)) > 0)[:, 0].to_numpy()
    return int(invested.argmax()) if invested.any() else weights.height - 1


def _criteria(config: QlabConfig) -> research_report.Criteria:
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


Kept = tuple[Hypothesis, dict[str, float], Policy, pl.DataFrame]
Refs = tuple[np.ndarray, int, float]  # rendements du buy & hold (grille entière), N, V[SR]


def _gate(setup: Setup, hypotheses: Sequence[Hypothesis]) -> tuple[list[Kept], list[str]]:
    """Gate de coûts au palier : essais retenus et une ligne de rapport par essai."""
    cfg, market = setup.config, setup.market
    costs = lt_costs.tier_costs(cfg, setup.snapshot, setup.tier, {})
    kept, lines = [], [f"## Gate de coûts au palier {setup.tier.name}", ""]
    for h in hypotheses:
        policy = strategies.policy_for(h)
        for params in h.grid():
            weights = strategies.strategy_for(h).build(market, params)
            cost = lt_costs.simulate(weights, market.closes, policy, costs, market.days_per_year)
            verdict = lt_costs.gate(cost, h, cfg.longterm.costs)
            name = strategies.trial_name(h, params)
            lines.append(f"- {name} : {verdict.label} ({verdict.detail})")
            if verdict.passed:
                kept.append((h, params, policy, weights))
    return kept, lines


def _result(returns: np.ndarray, ppy: int) -> TrialResult:
    """Moments de l'essai ; jamais investi (rendements constants) ⇒ Sharpe 0 par convention
    (loi normale : asymétrie 0, kurtosis 3) : l'essai compte quand même dans N."""
    if np.ptp(returns) == 0:
        return TrialResult(0.0, returns.size, 0.0, 3.0, ppy)
    return lt_report.trial_result(returns, ppy)


def _judge(setup: Setup, trial: Kept, returns: np.ndarray, refs: Refs) -> str:
    """Verdict d'un essai face aux références, sur sa fenêtre d'évaluation (après chauffe) ;
    non évaluable ⇒ dit pourquoi."""
    h, params, policy, weights = trial
    full_bench, n_trials, variance = refs
    first = first_decision(weights)
    bench = full_bench[first:]
    name, ppy = strategies.trial_name(h, params), setup.market.days_per_year
    dates = setup.market.closes[sg.DATE].to_list()
    labels = lt_report.year_labels(dates)[first:]
    window = f"Évaluation du {date_str(dates[first])} au {date_str(dates[-1])} (chauffe exclue)."
    try:
        dca = _dca(setup, first)
        per_asset = _per_asset(setup, h, params, policy)
        evidence = research_report.Evidence(
            name, returns, bench, labels, per_asset, n_trials, variance, ppy
        )
        report = research_report.evaluate(evidence, _criteria(setup.config))
        strategy_metrics = lt_report.metrics(returns, ppy)
    except DataError as e:  # ex. jamais investi : Sharpe indéfini
        return f"## {name}\n\n**Verdict : {research_report.REJECTED}** (non évaluable : {e})\n"
    bench_metrics = lt_report.metrics(bench, ppy)
    return lt_report.render(report, strategy_metrics, bench_metrics, dca, [window, *NOTES])


def run_wave(setup: Setup, hypotheses: Sequence[Hypothesis], registry: TrialRegistry) -> str:
    """Rapport Markdown de la vague (voir l'en-tête)."""
    ppy, closes = setup.market.days_per_year, setup.market.closes
    kept, lines = _gate(setup, hypotheses)
    returns = []
    for h, params, policy, weights in kept:  # 1) tout enregistrer d'abord
        r = _returns(setup, weights, policy, closes)[first_decision(weights) :]
        registry.record(h, params, _result(r, ppy), ts_ms=now_ms(), note=setup.tier.name)
        returns.append(r)
    n_trials = registry.n_trials("longterm")
    variance = float(np.var(registry.sharpes("longterm")))
    bench = _returns(setup, sg.equal_weight(closes), BUY_HOLD, closes)
    lines += [
        "",
        _oracle_check(setup, bench),
        f"N = {n_trials} essais dans le volet ; V[SR] = {variance:.2e}",
        "",
    ]
    for trial, r in zip(kept, returns, strict=True):  # 2) puis juger
        lines.append(_judge(setup, trial, r, (bench, n_trials, variance)))
    return "\n".join(lines)


def _oracle_check(setup: Setup, bench: np.ndarray) -> str:
    """Oracle : chaque jour, tout sur l'actif au meilleur rendement d'ouverture à ouverture sur
    la période qu'il va détenir (s'il est positif). Il doit multiplier le buy & hold."""
    opens = setup.opens.select(pl.exclude(sg.DATE)).to_numpy()
    held = opens[2:] / opens[1:-1] - 1  # décidé en t, détenu de l'ouverture t+1 à t+2
    best = np.nan_to_num(held, nan=-np.inf)
    rows = np.zeros_like(opens)
    pick = best.argmax(axis=1)
    rows[np.arange(held.shape[0]), pick] = (best.max(axis=1) > 0).astype(float)
    assets = [c for c in setup.opens.columns if c != sg.DATE]
    weights = pl.DataFrame(
        {sg.DATE: setup.opens[sg.DATE], **dict(zip(assets, rows.T, strict=True))}
    )
    daily = Policy("calendar", period_days=1)
    oracle = _returns(setup, weights, daily, setup.market.closes)
    ratio = float(np.prod(1 + oracle) / np.prod(1 + bench))
    verdict = (
        "ok"
        if ratio > 2
        else "**SUSPECT : le moteur ne récompense pas la connaissance de l'avenir**"
    )
    return f"Contrôle anti-fuite du pipeline (oracle / buy & hold = {ratio:.3g}) : {verdict}"


def _dca(setup: Setup, first: int) -> tuple[lt_report.Metrics, float]:
    """DCA de référence sur la même fenêtre que l'essai (à partir de ``first``)."""
    cfg = setup.config.longterm.dca
    closes, opens = setup.market.closes[first:], setup.opens[first:]
    dates = closes[sg.DATE].to_list()
    flows = lt_report.dca_flows(dates, Decimal(str(cfg.amount_quote)), cfg.period_days)
    weights = sg.equal_weight(closes)
    policy = Policy("calendar", period_days=cfg.period_days)
    res = lt_backtest.run(
        weights, opens, closes, policy, setup.rules, initial_quote=Decimal(0), flows=flows
    )
    ppy = setup.market.days_per_year
    return lt_report.metrics(res.twr_returns(), ppy), lt_report.mwr(res.flows, res.equity[-1], ppy)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="verdict de la vague : gate, backtest, registre, critères")
    run.add_argument("--tier", default=None, help="palier de capital ; défaut : le premier")
    run.add_argument("--hypotheses", type=Path, default=Path("hypotheses"))
    run.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _action(ctx: Context) -> int:
    config = ctx.config
    snapshot = SnapshotStore(ctx.paths, config.base.exchange(ctx.args.exchange).name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    fiches = [h for h in load_all(ctx.args.hypotheses) if h.volet == "longterm"]
    ma_days = strategies.breadth_lengths(fiches)
    market = strategies.load_market(ctx.paths, config, snapshot, ma_days, ctx.args.exchange)
    bars = {s: klines.load(ctx.paths, universe.INTERVAL, s) for s in config.base.symbols.trade}
    wanted = ctx.args.tier or config.base.capital_tiers[0].name
    tier = next((t for t in config.base.capital_tiers if t.name == wanted), None)
    if tier is None:
        raise DataError(f"palier inconnu : {wanted}")
    rules = lt_backtest.pair_rules(config, snapshot, {})
    setup = Setup(config, snapshot, market, sg.wide(bars, "open"), rules, tier)
    body = run_wave(setup, fiches, TrialRegistry(ctx.paths.trials))
    now = now_ms()
    path = ctx.paths.reports / f"lt_wave_{date_str(now)}.md"
    write_atomic(path, f"# Vague long terme — {date_str(now)}\n\n{body}".encode(), overwrite=True)
    print(f"Rapport : {path}")
    ctx.journal.info("lt_wave.run", {"tier": tier.name, "report": str(path)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.lt_wave",
        description="Verdict d'une vague de fiches long terme",
        component="lt_wave",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
