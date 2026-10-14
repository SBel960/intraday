"""Balayage du volet timing : contrats perpétuels USDⓈ-M de Binance, barres horaires.

Prépare l'étude (``timing/study.py``) : panel horaire de tous les contrats déjà éligibles une
fois (univers point-in-time), coûts taker mesurés par contrat, ``n_long`` / ``n_short`` de la
config, poids égaux, référence = buy & hold du contrat ``timing.reference_symbol`` ; âges des
annonces Binance (``timing/event_book.py``) pour les fiches d'événements. Fiches
``hypotheses/timing/tm_<signal>.yaml`` ; budget ``timing.max_trials``.

Commande : ``python -m qlab.timing.scan --config config run [--only tm_x,tm_y]``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.timeutils import date_to_ms
from qlab.events import classify
from qlab.events.news import stored
from qlab.exchange import spreads
from qlab.research.hypothesis import load_all
from qlab.timing import event_book
from qlab.timing import hourly_signals as hs
from qlab.timing import panel as pn
from qlab.timing.portfolio import buy_and_hold, leg_costs
from qlab.timing.study import Study, check_budget, run_scan

FICHES = Path("hypotheses/timing")
VOLET = "timing"


def prepare(ctx: Context) -> Study:
    """Panel des contrats déjà éligibles une fois au moins, coûts d'une jambe par contrat."""
    cfg, base = ctx.config.timing, ctx.config.base
    universe = pn.Universe(
        cfg.quote_asset,
        frozenset(ctx.config.longterm.universe.excluded_bases),
        cfg.warmup_days,
        cfg.volume_lookback_days,
        cfg.min_volume_quote,
    )
    panel = pn.load(ctx.paths, universe)
    panel = panel.filter(pl.col("eligible").any().over("symbol"))
    panel = hs.with_reference(panel, cfg.reference_symbol)
    news = classify.events_table(stored(ctx.paths), ctx.config.events.latency_margin_s)
    panel = event_book.with_event_ages(panel, news, cfg.quote_asset)
    measured = spreads.medians(ctx.paths, cfg.spread_min_samples, market="um")
    symbols = panel["symbol"].unique().sort().to_list()
    leg = dict(slippage_frac=cfg.slippage_frac, fallback_spread_frac=cfg.fallback_spread_frac)
    costs = leg_costs(symbols, measured, taker_frac=cfg.fees.effective_taker_frac, **leg)
    default = leg_costs(["?"], {}, taker_frac=cfg.fees.effective_taker_frac, **leg)["?"]
    known = sum(s in measured for s in symbols)
    note = (
        f"{cfg.n_long} achat(s) / {cfg.n_short} vente(s) à découvert, rendements nets de frais "
        f"taker, spread, glissement et financement. Spreads mesurés pour {known}/{len(symbols)} "
        f"contrats (médiane d'au moins {cfg.spread_min_samples} relevés récents, appliquée à "
        f"tout l'historique) ; repli {cfg.fallback_spread_frac:.2%} pour les autres."
    )
    days = base.exchange(ctx.args.exchange).trading_days_per_year
    return Study(
        volet=VOLET,
        prefix="tm_",
        unit="h",
        panel=panel,
        costs=costs,
        default_cost=default,
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        weighting="equal",
        vol_lookback=2,
        holdout_bar=pn.bar_at(panel, date_to_ms(cfg.holdout_start)),
        bars_per_year=days * hs.HOURS_PER_DAY,
        bars_per_day=hs.HOURS_PER_DAY,
        benchmark=partial(buy_and_hold, panel, cfg.reference_symbol),
        holdout_path=ctx.paths.timing_holdout,
        note=note,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="essayer toutes les combinaisons déclarées")
    run.add_argument("--only", help="fiches à essayer, ex. tm_momentum,tm_flow")
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    parser.add_argument("--fiches", type=Path, default=FICHES, help="dossier des fiches timing")


def _action(ctx: Context) -> int:
    fiches = load_all(ctx.args.fiches)
    check_budget(fiches, ctx.config.timing.max_trials)
    if ctx.args.only:
        fiches = [h for h in fiches if h.id in ctx.args.only.split(",")]
    return run_scan(ctx, prepare(ctx), fiches)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.timing.scan",
        description="Balayage des fiches timing (essais comptés, avant hold-out)",
        component="timing_scan",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
