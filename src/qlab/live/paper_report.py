"""Rapport du paper trading face au backtest : dernier critère d'acceptation (LT.6).

``python -m qlab.live.paper_report --config config --trial "lt_ts_momentum · lookback_days=30"``
écrit ``reports/paper_{date}.md``.

1. **Résultat dans l'intervalle du backtest** : le paper a tenu ``N`` jours (de sa première à sa
   dernière barre) et fait un rendement cumulé ``R``. On tire ``n_boot`` chemins de ``N`` jours
   dans les rendements quotidiens du backtest de l'essai (fenêtre jugée, chauffe exclue ; bootstrap
   stationnaire de ``research/bootstrap.py``, par blocs : tendances et périodes agitées
   gardées). Verdict : **conforme** si ``R`` est dans l'intervalle à ``confidence`` ; **en
   dessous** (rejet) ou **au-dessus** (à examiner : chance, ou exécution qui ne ressemble pas au
   backtest) sinon ; **en cours** tant que ``N`` < ``paper_min_days``.
2. **Fidélité des données** : clôtures et ouvertures notées au journal (API REST) comparées aux
   archives Binance, quand elles sont publiées. Un écart veut dire que le paper n'a pas décidé
   sur les mêmes prix que le backtest.
3. **Exploitation** : barres manquées (PC éteint), décisions par garde-fou, ordres exécutés et
   refusés, frais.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.config import AcceptanceConfig
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.records import Record, read_records
from qlab.core.timeutils import MS_PER_DAY, date_str, now_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.exchange.spreads import medians as spread_medians
from qlab.live.cockpit import trial_by_name
from qlab.live.paper import book_name
from qlab.longterm import klines, lt_backtest, strategies
from qlab.longterm import lt_evaluate as ev
from qlab.longterm import signals as sg
from qlab.research.bootstrap import stationary_indices
from qlab.research.hypothesis import load_all

Floats = npt.NDArray[np.float64]
REL_TOL = 1e-9
DAY = MS_PER_DAY


@dataclass(frozen=True, slots=True)
class PaperRun:
    first_bar_ms: int
    last_bar_ms: int
    days: int  # jours écoulés de la première à la dernière barre
    total_return: float
    decisions: Counter[str]  # trade / suspend / flatten
    missed_bars: int
    filled: int
    refused: int
    fees: Decimal


def paper_run(records: Sequence[Record]) -> PaperRun:
    """Résumé du journal ; ``DataError`` s'il n'a encore aucune barre."""
    days = [r for r in records if r.get("kind") == "day"]
    if not days:
        raise DataError("aucune barre au journal : le paper n'a pas encore tourné")
    first, last = days[0], days[-1]
    executions = [e for r in days for e in r["executions"]]
    return PaperRun(
        int(first["bar_ms"]),
        int(last["bar_ms"]),
        (int(last["bar_ms"]) - int(first["bar_ms"])) // MS_PER_DAY,
        float(Decimal(last["value_close"]) / Decimal(first["value_close"]) - 1),
        Counter(r["risk"]["action"] for r in days),
        sum(int(r.get("missed_bars", 0)) for r in days),
        sum(1 for e in executions if e["accepted"]),
        sum(1 for e in executions if not e["accepted"]),
        sum((Decimal(e["fee"]) for e in executions), Decimal(0)),
    )


@dataclass(frozen=True, slots=True)
class Band:
    low: float
    high: float
    percentile: float  # part des chemins du backtest sous le résultat du paper (0–100)


def backtest_band(backtest: Floats, days: int, paper: float, cfg: AcceptanceConfig) -> Band:
    """Intervalle des rendements cumulés du backtest sur ``days`` jours (voir l'en-tête, 1)."""
    if not 1 <= days <= backtest.size:
        raise DataError(
            f"fenêtre de {days} jours hors de l'historique du backtest ({backtest.size})"
        )
    rng = np.random.default_rng(cfg.seed)
    paths = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        idx = stationary_indices(backtest.size, cfg.mean_block_days, rng)[:days]
        paths[b] = np.prod(1 + backtest[idx]) - 1
    tail = (1 - cfg.confidence) / 2 * 100
    low, high = np.percentile(paths, [tail, 100 - tail])
    return Band(float(low), float(high), float((paths < paper).mean() * 100))


def verdict(run: PaperRun, band: Band, min_days: int) -> str:
    if run.days < min_days:
        return f"en cours ({run.days} / {min_days} jours)"
    if run.total_return < band.low:
        return "en dessous de l'intervalle du backtest : rejet"
    if run.total_return > band.high:
        return "au-dessus de l'intervalle du backtest : à examiner (chance ou exécution différente)"
    return "conforme au backtest"


def data_gaps(
    records: Sequence[Record], archives: Mapping[str, pl.DataFrame]
) -> tuple[int, list[str]]:
    """Prix du journal comparés aux archives : (prix comparés, écarts décrits)."""
    by_open = {s: _prices_by_open(a) for s, a in archives.items()}
    checked, gaps = 0, []
    for r in (r for r in records if r.get("kind") == "day"):
        bar = int(r["bar_ms"])
        noted = (("clôture", r["closes"], bar, 1), ("ouverture", r["next_opens"], bar + DAY, 0))
        for kind, prices, at, pos in noted:
            for s, price in prices.items():
                known = by_open.get(s, {}).get(at)
                if known is None:
                    continue
                checked += 1
                if not math.isclose(price, known[pos], rel_tol=REL_TOL):
                    gaps.append(
                        f"{s} {kind} du {date_str(at)} : journal {price}, archive {known[pos]}"
                    )
    return checked, gaps


def _prices_by_open(archive: pl.DataFrame) -> dict[int, tuple[float, float]]:
    """Barre d'archive → (ouverture, clôture), indexée par son heure d'ouverture."""
    rows = archive.select("open_time_ms", "open", "close").iter_rows()
    return {int(t): (float(o), float(c)) for t, o, c in rows}


def render(
    trial: str, run: PaperRun, band: Band, cfg: AcceptanceConfig, fidelity: tuple[int, list[str]]
) -> str:
    checked, gaps = fidelity
    band_text = (
        f"- Backtest sur {run.days} jours, intervalle à {cfg.confidence:.0%} : "
        f"[{band.low:+.2%} ; {band.high:+.2%}] ({cfg.n_boot} chemins, blocs de "
        f"{cfg.mean_block_days:g} jours) ; le paper fait mieux que "
        f"{band.percentile:.0f} % d'entre eux"
    )
    decisions = ", ".join(f"{a} {n}" for a, n in sorted(run.decisions.items()))
    lines = [
        f"## {trial}",
        "",
        f"**Verdict : {verdict(run, band, cfg.paper_min_days)}**",
        "",
        f"- Période : barres du {date_str(run.first_bar_ms)} au {date_str(run.last_bar_ms)}"
        f" ({run.days} jours)",
        f"- Rendement du paper : {run.total_return:+.2%}",
        band_text,
        f"- Décisions : {decisions}",
        f"- Barres manquées (PC éteint) : {run.missed_bars}",
        f"- Ordres : {run.filled} exécutés, {run.refused} refusés ; frais {run.fees:.4f}",
        f"- Fidélité des données : {checked} prix comparés aux archives, {len(gaps)} écart(s)",
        *(f"  - {g}" for g in gaps[:20]),
    ]
    return "\n".join(lines) + "\n"


def _backtest_returns(ctx: Context, trial: str) -> Floats:
    """Rendements quotidiens du backtest de l'essai, fenêtre jugée (chauffe exclue)."""
    config, ex = ctx.config, ctx.config.base.exchange(ctx.args.exchange)
    snapshot = SnapshotStore(ctx.paths, ex.name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    h, params = trial_by_name(load_all(ctx.args.hypotheses), trial)
    market = strategies.load_market(ctx.paths, config, snapshot, (), ex.name)
    bars = {s: klines.load(ctx.paths, "1d", s) for s in config.base.symbols.trade}
    spreads = spread_medians(ctx.paths, config.longterm.costs.spread_min_samples)
    rules = lt_backtest.pair_rules(config, snapshot, spreads)
    tier = config.base.capital_tiers[0]
    setup = ev.Setup(config, snapshot, market, sg.wide(bars, "open"), rules, tier, spreads)
    weights = strategies.strategy_for(h).build(market, params)
    r = ev.returns(setup, weights, strategies.policy_for(h), market.closes, setup.opens)
    return r[ev.first_decision(weights) :]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial", required=True, help="nom exact de l'essai suivi en paper")
    parser.add_argument("--hypotheses", type=Path, default=Path("hypotheses"))
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _action(ctx: Context) -> int:
    trial = ctx.args.trial
    path = ctx.paths.paper_journal(book_name(trial))
    if not path.exists():
        raise DataError(f"pas de journal de paper pour cet essai : {path}")
    records = read_records(path)
    run = paper_run(records)
    cfg = ctx.config.longterm.acceptance
    band = backtest_band(_backtest_returns(ctx, trial), max(run.days, 1), run.total_return, cfg)
    archives = {s: klines.load(ctx.paths, "1d", s) for s in ctx.config.base.symbols.trade}
    body = render(trial, run, band, cfg, data_gaps(records, archives))
    today = date_str(now_ms())
    out = ctx.paths.reports / f"paper_{today}.md"
    write_atomic(
        out, f"# Paper trading face au backtest — {today}\n\n{body}".encode(), overwrite=True
    )
    print(body + f"\nRapport : {out}")
    ctx.journal.info(
        "paper_report", {"trial": trial, "verdict": verdict(run, band, cfg.paper_min_days)}
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="qlab.live.paper_report",
        description="paper trading face au backtest (critère d'acceptation)",
        component="paper_report",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
