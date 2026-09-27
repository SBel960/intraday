"""Verdict d'une vague de fiches long terme : gate de coûts, backtest, registre, critères.

Pour un palier de capital (``--tier``, capital de départ du backtest) :

1. chaque essai (fiche × paramètres) passe d'abord le **gate de coûts** (``lt_costs``) ; un
   essai écarté n'est jamais backtesté ;
2. les essais retenus sont backtestés (``lt_backtest``) et **tous enregistrés** dans le registre
   d'essais **avant** tout calcul de verdict : N et la dispersion des Sharpe du DSR comptent
   tout ce qui a été essayé dans le volet ;
3. chaque essai est évalué par ``lt_evaluate`` : univers de la fiche, références (buy & hold,
   DCA), stabilité par année et par actif, fenêtre après la chauffe ;
4. contrôle anti-fuite du pipeline (LT.5), une fois par vague sur les vrais prix : un oracle
   qui connaît le rendement qu'il va détenir doit écraser le buy & hold, sinon le rapport
   l'écrit en tête (un « tricheur » par essai n'est pas un bon test : connaître les poids de
   demain d'une fiche de faible volatilité ne dit rien des rendements) ;
   un essai déjà enregistré (jugé dans une vague précédente) n'est ni regaté ni rejugé ;
5. critères et verdict (``research/report.py``), rapport ``reports/lt_wave_{date}.md``.

Commandes : ``python -m qlab.longterm.lt_wave --config config run [--tier t0]`` (verdict) et
``... costs`` (gate de coûts de tous les paliers, sans backtest), ``... judge --trial "NOM"``
(verdict seul d'un essai enregistré resté sans verdict) ; ``--hypotheses hypotheses``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.timeutils import date_str, now_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.exchange.spreads import medians as spread_medians
from qlab.longterm import klines, lt_backtest, lt_costs, strategies, universe
from qlab.longterm import lt_evaluate as ev
from qlab.longterm import signals as sg
from qlab.research.hypothesis import Hypothesis, load_all
from qlab.research.trials import TrialRegistry


def _gate(
    setup: ev.Setup, hypotheses: Sequence[Hypothesis], registry: TrialRegistry
) -> tuple[list[ev.Kept], list[str]]:
    """Gate de coûts au palier, dans l'univers de chaque fiche : essais retenus et une ligne de
    rapport par essai. Un essai déjà enregistré n'est ni regaté ni rejugé."""
    cfg, market = setup.config, setup.market
    kept, lines = [], [f"## Gate de coûts au palier {setup.tier.name}", ""]
    for h in hypotheses:
        policy, universe_ = strategies.policy_for(h), ev.panel(setup, h)
        symbols = [c for c in universe_.closes.columns if c != sg.DATE]
        costs = lt_costs.tier_costs(cfg, setup.snapshot, setup.tier, setup.spreads, symbols)
        for params in h.grid():
            name = strategies.trial_name(h, params)
            judged = registry.find("longterm", h.id, params)
            if judged is not None:
                lines.append(f"- {name} : déjà jugé le {date_str(judged.ts_ms)}, verdict gardé")
                continue
            weights = strategies.strategy_for(h).build(market, params)
            ppy = market.days_per_year
            cost = lt_costs.simulate(weights, universe_.closes, policy, costs, ppy)
            verdict = lt_costs.gate(cost, h, cfg.longterm.costs)
            lines.append(f"- {name} : {verdict.label} ({verdict.detail})")
            if verdict.passed:
                kept.append((h, params, policy, weights))
    return kept, lines


def run_wave(setup: ev.Setup, hypotheses: Sequence[Hypothesis], registry: TrialRegistry) -> str:
    """Rapport Markdown de la vague (voir l'en-tête)."""
    ppy = setup.market.days_per_year
    kept, lines = _gate(setup, hypotheses, registry)
    measured = []
    for h, params, policy, weights in kept:  # 1) tout enregistrer d'abord
        universe_ = ev.panel(setup, h)
        r = ev.returns(setup, weights, policy, universe_.closes, universe_.opens)
        r = r[ev.first_decision(weights) :]
        registry.record(h, params, ev.trial_result(r, ppy), ts_ms=now_ms(), note=setup.tier.name)
        measured.append(r)
    n_trials = registry.n_trials("longterm")
    variance = float(np.var(registry.sharpes("longterm")))
    trade = strategies.Panel(setup.market.closes, setup.opens)
    bench = ev.returns(setup, sg.equal_weight(trade.closes), ev.BUY_HOLD, trade.closes, trade.opens)
    lines += [
        "",
        ev.oracle_check(setup, bench),
        f"N = {n_trials} essais dans le volet ; V[SR] = {variance:.2e}",
        "",
    ]
    for trial, r in zip(kept, measured, strict=True):  # 2) puis juger
        lines.append(ev.judge(setup, trial, r, n_trials, variance))
    return "\n".join(lines)


def judge_only(
    setup: ev.Setup, hypotheses: Sequence[Hypothesis], registry: TrialRegistry, names: Sequence[str]
) -> str:
    """Verdict seul d'essais **déjà enregistrés** restés sans verdict (ex. défaut du pipeline
    corrigé) : ni gate, ni enregistrement, N inchangé. Le rendement est recalculé et doit
    retrouver le Sharpe du registre (même données, même code de backtest), sinon refus."""
    grid = {strategies.trial_name(h, p): (h, p) for h in hypotheses for p in h.grid()}
    n_trials = registry.n_trials("longterm")
    variance = float(np.var(registry.sharpes("longterm")))
    lines = [f"N = {n_trials} essais dans le volet ; V[SR] = {variance:.2e}", ""]
    for name in names:
        if name not in grid:
            raise DataError(f"essai inconnu : {name!r}")
        h, params = grid[name]
        trial = registry.find("longterm", h.id, params)
        if trial is None or trial.note != setup.tier.name:
            raise DataError(f"{name} : pas enregistré au palier {setup.tier.name}")
        policy = strategies.policy_for(h)
        weights = strategies.strategy_for(h).build(setup.market, params)
        universe_ = ev.panel(setup, h)
        r = ev.returns(setup, weights, policy, universe_.closes, universe_.opens)
        r = r[ev.first_decision(weights) :]
        if not np.isclose(
            ev.trial_result(r, setup.market.days_per_year).sharpe, trial.result.sharpe
        ):
            raise DataError(f"{name} : rendement différent de celui du registre, refus")
        lines.append(ev.judge(setup, (h, params, policy, weights), r, n_trials, variance))
    return "\n".join(lines)


def cost_rows(setup: ev.Setup, hypotheses: Sequence[Hypothesis]) -> list[lt_costs.Row]:
    """Gate de coûts de chaque essai à **chaque** palier, dans l'univers de sa fiche (rapport
    ``lt_costs`` ; n'enregistre rien : aucune performance n'est calculée)."""
    cfg, market = setup.config, setup.market
    rows = []
    for h in hypotheses:
        policy, universe_ = strategies.policy_for(h), ev.panel(setup, h)
        symbols = [c for c in universe_.closes.columns if c != sg.DATE]
        tiers = [
            (t, lt_costs.tier_costs(cfg, setup.snapshot, t, setup.spreads, symbols))
            for t in cfg.base.capital_tiers
        ]
        for params in h.grid():
            weights = strategies.strategy_for(h).build(market, params)
            name = strategies.trial_name(h, params)
            for tier, costs in tiers:
                res = lt_costs.simulate(
                    weights, universe_.closes, policy, costs, market.days_per_year
                )
                verdict = lt_costs.gate(res, h, cfg.longterm.costs)
                rows.append(lt_costs.Row(name, tier.name, res, verdict))
    return rows


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="verdict de la vague : gate, backtest, registre, critères")
    run.add_argument("--tier", default=None, help="palier de capital ; défaut : le premier")
    costs = sub.add_parser("costs", help="gate de coûts de tous les paliers (sans backtest)")
    judge = sub.add_parser("judge", help="verdict seul d'essais enregistrés restés sans verdict")
    judge.add_argument("--tier", default=None, help="palier où les essais ont été enregistrés")
    judge.add_argument("--trial", action="append", required=True, help="nom exact de l'essai")
    for p in (run, costs, judge):
        p.add_argument("--hypotheses", type=Path, default=Path("hypotheses"))
        p.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _prepare(
    ctx: Context, tier_name: str | None, spreads_until_ms: int | None = None
) -> tuple[ev.Setup, list[Hypothesis]]:
    """Données, règles d'ordre et spreads communs aux commandes (spreads relevés jusqu'à
    ``spreads_until_ms`` pour refaire un calcul à l'identique)."""
    config = ctx.config
    snapshot = SnapshotStore(ctx.paths, config.base.exchange(ctx.args.exchange).name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    fiches = [h for h in load_all(ctx.args.hypotheses) if h.volet == "longterm"]
    wanted = tier_name or config.base.capital_tiers[0].name
    tier = next((t for t in config.base.capital_tiers if t.name == wanted), None)
    if tier is None:
        raise DataError(f"palier inconnu : {wanted}")
    quoted = any(h.universe == "trade_eur" for h in fiches)
    market = strategies.load_market(
        ctx.paths,
        config,
        snapshot,
        strategies.breadth_lengths(fiches),
        ctx.args.exchange,
        quoted=quoted,
    )
    bars = {s: klines.load(ctx.paths, universe.INTERVAL, s) for s in config.base.symbols.trade}
    spreads = spread_medians(ctx.paths, config.longterm.costs.spread_min_samples, spreads_until_ms)
    symbols = set(config.base.symbols.trade)
    if market.quoted is not None:
        symbols |= {c for c in market.quoted.closes.columns if c != sg.DATE}
    rules = lt_backtest.pair_rules(config, snapshot, spreads, sorted(symbols))
    return ev.Setup(config, snapshot, market, sg.wide(bars, "open"), rules, tier, spreads), fiches


def _recorded_at(ctx: Context) -> int | None:
    """Pour ``judge`` : instant d'enregistrement le plus récent des essais demandés (leurs
    spreads d'alors) ; ``None`` pour les autres commandes."""
    if ctx.args.cmd != "judge":
        return None
    registry = TrialRegistry(ctx.paths.trials)
    names = set(ctx.args.trial)
    found = [
        t.ts_ms
        for h in load_all(ctx.args.hypotheses)
        for p in h.grid()
        if strategies.trial_name(h, p) in names
        and (t := registry.find("longterm", h.id, p)) is not None
    ]
    return max(found) if found else None


def _action(ctx: Context) -> int:
    setup, fiches = _prepare(ctx, getattr(ctx.args, "tier", None), _recorded_at(ctx))
    now = now_ms()
    if ctx.args.cmd == "costs":
        rows = cost_rows(setup, fiches)
        measured = set(setup.config.base.symbols.trade) <= set(setup.spreads)
        body = lt_costs.render(rows, setup.config.longterm.costs, spread_measured=measured)
        title = (
            f"# Gate de coûts long terme — {date_str(now)} (snapshot {setup.snapshot.path.name})"
        )
        path = ctx.paths.reports / f"lt_costs_{date_str(now)}.md"
        print(body)
    elif ctx.args.cmd == "judge":
        body = judge_only(setup, fiches, TrialRegistry(ctx.paths.trials), ctx.args.trial)
        title = f"# Verdicts complémentaires — {date_str(now)}"
        path = ctx.paths.reports / f"lt_verdict_{date_str(now)}.md"
    else:
        body = run_wave(setup, fiches, TrialRegistry(ctx.paths.trials))
        title = f"# Vague long terme — {date_str(now)}"
        path = ctx.paths.reports / f"lt_wave_{date_str(now)}.md"
    write_atomic(path, f"{title}\n\n{body}\n".encode(), overwrite=True)
    print(f"Rapport : {path}")
    ctx.journal.info(f"lt_wave.{ctx.args.cmd}", {"tier": setup.tier.name, "report": str(path)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.lt_wave",
        description="Gate de coûts et verdict d'une vague de fiches long terme",
        component="lt_wave",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
