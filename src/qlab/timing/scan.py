"""Balayage du volet timing : chaque combinaison déclarée dans ``hypotheses/timing`` est essayée,
comptée dans le registre d'essais, et classée — sur la période **avant** le hold-out.

Une fiche ``tm_<signal>`` essaie le signal ``<signal>`` (``timing/hourly_signals.py``) avec ses
``lookback_h`` et ``horizon_h`` ; portefeuille ``timing/portfolio.py`` (``n_long`` / ``n_short``
de la config, univers point-in-time, coûts taker mesurés). Un essai déjà enregistré n'est pas
recompté (le registre garde la dernière mesure). Le budget ``timing.max_trials`` est vérifié
**avant** de lancer quoi que ce soit.

Sorties : tableau Markdown classé (``reports/timing_scan_{date}.md``) et JSON (cockpit).
Verdicts des meilleurs et hold-out : ``timing/verdict.py``.

Commande : ``python -m qlab.timing.scan --config config run [--only tm_x,tm_y]``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.timeutils import date_str, date_to_ms, now_ms
from qlab.exchange import spreads
from qlab.longterm.lt_evaluate import trial_result
from qlab.research.hypothesis import Hypothesis, load_all
from qlab.research.trials import TrialRegistry
from qlab.timing import hourly_signals as hs
from qlab.timing import panel as pn
from qlab.timing.portfolio import Book, leg_costs, period_returns

FICHES = Path("hypotheses/timing")
PREFIX = "tm_"


@dataclass(frozen=True, slots=True)
class Study:
    """Données et règles communes à tous les essais."""

    panel: pl.DataFrame
    costs: dict[str, float]
    default_cost: float
    n_long: int
    n_short: int
    holdout_ms: int
    days_per_year: int
    spread_note: str

    def book(self, horizon_h: int) -> Book:
        return Book(horizon_h, self.n_long, self.n_short, self.costs, self.default_cost)

    def periods_per_year(self, horizon_h: int) -> int:
        return round(self.days_per_year * hs.HOURS_PER_DAY / horizon_h)


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
    measured = spreads.medians(ctx.paths, cfg.spread_min_samples, market="um")
    symbols = panel["symbol"].unique().sort().to_list()
    leg = dict(slippage_frac=cfg.slippage_frac, fallback_spread_frac=cfg.fallback_spread_frac)
    costs = leg_costs(symbols, measured, taker_frac=cfg.fees.effective_taker_frac, **leg)
    default = leg_costs(["?"], {}, taker_frac=cfg.fees.effective_taker_frac, **leg)["?"]
    known = sum(s in measured for s in symbols)
    note = (
        f"Spreads mesurés pour {known}/{len(symbols)} contrats (médiane d'au moins "
        f"{cfg.spread_min_samples} relevés récents, appliquée à tout l'historique) ; "
        f"repli {cfg.fallback_spread_frac:.2%} pour les autres."
    )
    days = base.exchange(ctx.args.exchange).trading_days_per_year
    hold = date_to_ms(cfg.holdout_start)
    return Study(panel, costs, default, cfg.n_long, cfg.n_short, hold, days, note)


def signal_name(hypothesis: Hypothesis) -> str:
    if not hypothesis.id.startswith(PREFIX):
        raise DataError(f"{hypothesis.id} : une fiche timing s'appelle {PREFIX}<signal>")
    return hypothesis.id.removeprefix(PREFIX)


def returns_on(
    study: Study,
    panel: pl.DataFrame,
    hypothesis: Hypothesis,
    params: dict[str, float],
    signal: pl.Series | None = None,
) -> pl.DataFrame:
    """Rendements par période de l'essai sur ``panel`` (toute la période) ; ``signal`` : déjà
    calculé sur ce panel (le même signal sert à plusieurs horizons)."""
    h = int(params["horizon_h"])
    sig = hs.compute(panel, signal_name(hypothesis), params, h) if signal is None else signal
    return period_returns(panel, sig, study.book(h))


def trial_returns(
    study: Study, hypothesis: Hypothesis, params: dict[str, float], signal: pl.Series | None = None
) -> pl.DataFrame:
    """Rendements par période de l'essai, avant le hold-out."""
    out = returns_on(study, study.panel, hypothesis, params, signal)
    return pn.in_sample(out, study.holdout_ms, int(params["horizon_h"]))


def cached_signal(
    study: Study,
    hypothesis: Hypothesis,
    params: dict[str, float],
    cache: dict[tuple[str, float], pl.Series],
) -> pl.Series | None:
    """Signal réutilisable entre horizons (tous sauf la saisonnalité, qui dépend de h)."""
    name = signal_name(hypothesis)
    if name == "seasonality":
        return None
    key = (name, params["lookback_h"])
    if key not in cache:
        cache.clear()  # un seul signal en mémoire (panel de millions de lignes)
        cache[key] = hs.compute(study.panel, name, params, int(params["horizon_h"]))
    return cache[key]


def summary_row(
    hypothesis: Hypothesis, params: dict[str, float], out: pl.DataFrame, ppy: int
) -> dict[str, object]:
    """Résumé d'un essai : Sharpe annualisé, rendement, frottements (par an), exposition."""

    def per_year(column: str) -> float:
        return float(out[column].to_numpy().mean() * ppy)

    r = out["net"].to_numpy()
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    return {
        "id": hypothesis.id,
        "params": {k: int(v) for k, v in params.items()},
        "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        "net_per_year": per_year("net"),
        "gross_per_year": per_year("gross"),
        "cost_per_year": per_year("cost"),
        "funding_per_year": per_year("funding"),
        "invested": float((out["lines"].to_numpy() > 0).mean()),
        "liquidations": int(out["liquidations"].sum()),
        "periods": int(r.size),
        "start": date_str(int(out["t_ms"][0])),
    }


def check_budget(fiches: Sequence[Hypothesis], max_trials: int) -> int:
    total = sum(h.n_trials for h in fiches)
    if total > max_trials:
        raise DataError(f"{total} essais déclarés > budget timing.max_trials = {max_trials}")
    return total


def render(rows: list[dict[str, object]], study: Study, n_trials: int) -> str:
    lines = [
        f"# Volet timing — balayage du {date_str(now_ms())}",
        "",
        f"{n_trials} essais, période avant le hold-out ({date_str(study.holdout_ms)}), "
        f"{study.n_long} achat(s) / {study.n_short} vente(s) à découvert, levier 1×, "
        "rendements nets de frais taker, spread, glissement et financement.",
        "",
        study.spread_note,
        "",
        "| # | Essai | Paramètres | Sharpe an. | Net/an | Brut/an | Frais/an | Financement/an "
        "| Investi | Liquidations | Depuis |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        params = ", ".join(f"{k}={v}" for k, v in r["params"].items())  # type: ignore[attr-defined]
        lines.append(
            f"| {i} | {r['id']} | {params} | {r['sharpe']:+.2f} | {r['net_per_year']:+.1%} | "
            f"{r['gross_per_year']:+.1%} | {r['cost_per_year']:.1%} | "
            f"{r['funding_per_year']:+.1%} | {r['invested']:.0%} | {r['liquidations']} | {r['start']} |"
        )
    return "\n".join(lines) + "\n"


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
    study = prepare(ctx)
    registry = TrialRegistry(ctx.paths.trials)
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, float], pl.Series] = {}
    for hyp in fiches:
        for params in hyp.grid():
            out = trial_returns(study, hyp, params, cached_signal(study, hyp, params, cache))
            ppy = study.periods_per_year(int(params["horizon_h"]))
            registry.record(hyp, params, trial_result(out["net"].to_numpy(), ppy), ts_ms=now_ms())
            rows.append(summary_row(hyp, params, out, ppy))
            print(f"{hyp.id} {params} : Sharpe {rows[-1]['sharpe']:+.2f}", flush=True)
    rows.sort(key=lambda r: -float(r["sharpe"]))  # type: ignore[arg-type]
    n = registry.n_trials("timing")
    day = date_str(now_ms())
    write_atomic(
        ctx.paths.reports / f"timing_scan_{day}.md", render(rows, study, n).encode(), overwrite=True
    )
    blob = {"date": day, "n_trials": n, "rows": rows}
    write_atomic(
        ctx.paths.reports / "timing_scan_latest.json", json.dumps(blob).encode(), overwrite=True
    )
    ctx.journal.info("timing.scan", {"trials": len(rows), "n_trials": n})
    print(f"{len(rows)} essais ; N du volet = {n} ; rapport timing_scan_{day}.md")
    return 0


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
