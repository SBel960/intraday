"""Volet multi-actifs : étude, balayage, verdicts et hold-out (``docs/SPEC_MULTI_ACTIFS.md``).

Prépare l'étude commune (``timing/study.py``) sur le panel quotidien des 45 marchés
(``markets/daily_panel.py``) : barres = jours ouvrés, coût d'une jambe par classe d'actifs, poids
**inverses de la volatilité** (``vol_lookback_days``), référence = panier qui achète tous les
marchés éligibles avec les mêmes poids (``timing/portfolio.basket``), hold-out à partir de
``markets.holdout_start`` (registre ``DataPaths.markets_holdout``). Fiches
``hypotheses/multi/mk_<signal>.yaml`` (paramètres ``lookback_d``, ``horizon_d``), budget
``markets.max_trials``, N propre au volet ``multi``.

Commandes : ``python -m qlab.markets.multi_asset --config config scan [--only mk_trend]`` ;
``… best [--top 5]`` ; ``… holdout --id mk_x --params lookback_d=126,horizon_d=21``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from qlab.core.cli import Context, run_command
from qlab.core.timeutils import date_to_ms
from qlab.markets import daily_panel as mp
from qlab.research.hypothesis import load_all
from qlab.timing import panel as pn
from qlab.timing import verdict
from qlab.timing.portfolio import basket
from qlab.timing.study import Study, check_budget, run_scan

FICHES = Path("hypotheses/multi")
VOLET = "multi"


def prepare(ctx: Context) -> Study:
    cfg = ctx.config.markets
    panel = mp.load(ctx.paths, cfg)
    costs = {a.ticker: cfg.cost_of(a.ticker) for a in cfg.assets}
    note = (
        f"{len(cfg.assets)} marchés (actions, obligations, devises, matières premières, "
        "immobilier, crypto), jours ouvrés, poids inverses de la volatilité "
        f"({cfg.vol_lookback_days} j), coût d'une jambe par classe "
        f"({min(costs.values()):.2%} à {max(costs.values()):.2%}), emprunt des ventes à "
        "découvert non compté ; référence : panier de tous les marchés éligibles, mêmes poids."
    )
    return Study(
        volet=VOLET,
        prefix="mk_",
        unit="d",
        panel=panel,
        costs=costs,
        default_cost=max(costs.values()),
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        weighting="inverse_vol",
        vol_lookback=cfg.vol_lookback_days,
        holdout_bar=pn.bar_at(panel, date_to_ms(cfg.holdout_start)),
        bars_per_year=cfg.trading_days_per_year,
        bars_per_day=1,
        benchmark=partial(basket, panel, vol_lookback=cfg.vol_lookback_days),
        holdout_path=ctx.paths.markets_holdout,
        note=note,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="essayer toutes les combinaisons déclarées")
    scan.add_argument("--only", help="fiches à essayer, ex. mk_trend,mk_channel")
    best = sub.add_parser("best", help="verdict complet des meilleurs essais du balayage")
    best.add_argument("--top", type=int, default=5)
    hold = sub.add_parser("holdout", help="rejouer UN essai sur le hold-out (une seule fois)")
    hold.add_argument("--id", required=True)
    hold.add_argument("--params", required=True, help="ex. lookback_d=126,horizon_d=21")
    parser.add_argument("--fiches", type=Path, default=FICHES, help="dossier des fiches")


def _action(ctx: Context) -> int:
    fiches = load_all(ctx.args.fiches)
    check_budget(fiches, ctx.config.markets.max_trials)
    study = prepare(ctx)
    if ctx.args.cmd == "scan":
        chosen = ctx.args.only.split(",") if ctx.args.only else [h.id for h in fiches]
        return run_scan(ctx, study, [h for h in fiches if h.id in chosen])
    if ctx.args.cmd == "best":
        return verdict.best(ctx, study, fiches, ctx.args.top)
    return verdict.holdout(ctx, study, fiches, ctx.args.id, ctx.args.params)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.markets.multi_asset",
        description="Volet multi-actifs : balayage, verdicts, hold-out unique",
        component="markets_study",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
