"""Volet trading : étude, balayage, verdicts et hold-out (``docs/SPEC_TRADING.md``).

Prépare l'étude commune (``timing/study.py``) sur le panel horaire
(``trading/fx_panel.py``) : barres = heures, coût d'une jambe = commission + glissement de la
classe (``trading.classes``) + ½ spread supposé de l'instrument, poids inverses de la
volatilité (``vol_lookback_bars``), référence = panier de tous les instruments éligibles,
hold-out à partir de ``trading.holdout_start`` (registre ``DataPaths.trading_holdout``).
Fiches ``hypotheses/trading/tr_<signal>.yaml`` (``lookback_h``, ``horizon_h``), budget
``trading.max_trials``, N propre au volet ``trading``.

Commandes : ``python -m qlab.trading.intraday_study --config config scan [--only tr_x]`` ;
``… best [--top 5]`` ; ``… holdout --id tr_x --params lookback_h=24,horizon_h=4``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.timeutils import date_to_ms
from qlab.timing import hourly_signals as hs
from qlab.timing import panel as pn
from qlab.timing import verdict
from qlab.timing.portfolio import basket
from qlab.timing.study import Study
from qlab.trading import fx_panel

FICHES = Path("hypotheses/trading")
VOLET = "trading"


def prepare(ctx: Context) -> Study:
    cfg = ctx.config.trading
    panel = fx_panel.load(ctx.paths, cfg)
    names = panel["symbol"].unique().sort().to_list()
    costs = {n: cfg.leg_cost(n) for n in names}
    spreads = panel.group_by("symbol").agg(pl.col("spread_frac").median()).sort("symbol")
    typical = ", ".join(f"{s} {v:.3%}" for s, v in spreads.iter_rows() if v is not None)
    note = (
        f"{len(names)} instruments, barres horaires, poids inverses de la volatilité "
        f"({cfg.vol_lookback_bars} h), coût d'une jambe = commission + glissement "
        f"({min(costs.values()):.3%} à {max(costs.values()):.3%}) + ½ spread supposé "
        f"(spreads complets : {typical}), heures UTC {list(cfg.excluded_hours_utc)} exclues ; "
        "portage d'une nuit non compté ; référence : panier de tous les instruments éligibles."
    )
    return Study(
        volet=VOLET,
        prefix="tr_",
        unit="h",
        panel=panel,
        costs=costs,
        default_cost=max(costs.values()),
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        weighting="inverse_vol",
        vol_lookback=cfg.vol_lookback_bars,
        holdout_bar=pn.bar_at(panel, date_to_ms(cfg.holdout_start)),
        bars_per_year=cfg.days_per_year * hs.HOURS_PER_DAY,
        bars_per_day=hs.HOURS_PER_DAY,
        benchmark=partial(basket, panel, vol_lookback=cfg.vol_lookback_bars),
        holdout_path=ctx.paths.trading_holdout,
        note=note,
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    verdict.add_study_arguments(parser, FICHES, "lookback_h=24,horizon_h=4")


def _action(ctx: Context) -> int:
    return verdict.study_action(ctx, prepare, ctx.config.trading.max_trials)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.trading.intraday_study",
        description="Volet trading : balayage, verdicts, hold-out unique",
        component="trading_study",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
