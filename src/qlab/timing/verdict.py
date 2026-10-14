"""Verdict des meilleurs essais d'un volet (timing, multi-actifs), puis test **unique** sur le
hold-out.

``best`` : les ``--top`` meilleurs essais du dernier balayage (``{volet}_scan_latest.json``)
passent les critères complets de ``research/report.py`` (DSR sur N essais du volet, écart de
Sharpe avec la référence de l'étude par bootstrap stationnaire, stabilité par année civile,
deux **moitiés disjointes** de l'univers jouant le rôle des « actifs », MinTRL). Blocs du
bootstrap : ``acceptance.mean_block_days`` converti en périodes de h barres.

``holdout`` : un essai, désigné à l'avance, est rejoué **une seule fois** sur les données à
partir du début du hold-out ; le résultat est inscrit (``Study.holdout_path``) et un second
hold-out est refusé : regarder plusieurs candidats sur ces données les rendrait aussi suspectes
que la période de recherche.

Commandes (perpétuels) : ``python -m qlab.timing.verdict --config config best [--top 5]`` ;
``… holdout --id tm_x --params lookback_h=4,horizon_h=24``.
Multi-actifs : ``markets/multi_asset.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.config import QlabConfig
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.records import append_record, read_records
from qlab.core.timeutils import date_str, now_ms
from qlab.longterm import lt_report
from qlab.longterm.lt_evaluate import criteria
from qlab.research import report as rr
from qlab.research.hypothesis import Hypothesis, load_all
from qlab.research.trials import TrialRegistry
from qlab.timing import panel as pn
from qlab.timing import scan
from qlab.timing.portfolio import period_returns
from qlab.timing.study import Study, returns_on


def halves(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Deux moitiés disjointes de l'univers (contrats triés, un sur deux). La clôture de la
    référence est déjà une colonne du panel : le signal de rattrapage marche dans chaque moitié."""
    symbols = panel["symbol"].unique().sort().to_list()
    return {
        f"moitié {i + 1}": panel.filter(pl.col("symbol").is_in(symbols[i::2])) for i in range(2)
    }


def sharpe_variance(registry: TrialRegistry, volet: str, periods_per_year: int) -> float:
    """V[SR] du DSR à l'échelle de ``periods_per_year``. Les essais du volet n'ont pas la même
    période (1 h à 1 semaine) : chaque Sharpe par période est annualisé avec sa propre fréquence
    puis ramené à celle de l'essai jugé ; sans cela, les essais horaires gonflent SR*."""
    rescaled = [
        t.result.sharpe * np.sqrt(t.result.periods_per_year / periods_per_year)
        for t in registry.latest(volet).values()
    ]
    return float(np.var(rescaled, ddof=1)) if len(rescaled) > 1 else 0.0


def evidence(
    study: Study, hyp: Hypothesis, params: dict[str, float], registry: TrialRegistry
) -> rr.Evidence:
    """Rendements nets avant hold-out, référence de l'étude, moitiés de l'univers."""
    h = study.horizon(params)
    out = pn.in_sample(returns_on(study, study.panel, hyp, params), study.holdout_bar, h)
    bench = out.select("bar").join(study.benchmark(h), on="bar", how="left")
    per_asset = {}
    for label, part in halves(study.panel).items():
        sub = pn.in_sample(returns_on(study, part, hyp, params), study.holdout_bar, h)
        sub = sub.join(bench.select("bar", "bench"), on="bar", how="inner")
        per_asset[label] = (sub["net"].to_numpy(), sub["bench"].fill_null(0.0).to_numpy())
    years = [date_str(t)[:4] for t in out["t_ms"].to_list()]
    return rr.Evidence(
        f"{hyp.id} {', '.join(f'{k}={int(v)}' for k, v in params.items())}",
        out["net"].to_numpy(),
        bench["bench"].fill_null(0.0).to_numpy(),
        years,
        per_asset,
        registry.n_trials(study.volet),
        sharpe_variance(registry, study.volet, study.periods_per_year(h)),
        study.periods_per_year(h),
    )


def oracle_sharpe(study: Study, horizon: int) -> float:
    """Contrôle anti-fuite (§8) : un « signal » qui connaît le rendement à venir doit donner un
    Sharpe énorme avec le même moteur et les mêmes coûts ; sinon le pipeline est suspect."""
    peek = study.panel.select(pn.forward_return(horizon).alias("s"))["s"]
    book = study.book({f"horizon_{study.unit}": horizon})
    out = pn.in_sample(period_returns(study.panel, peek, book), study.holdout_bar, horizon)
    r = out["net"].to_numpy()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(study.periods_per_year(horizon)))


def timing_criteria(config: QlabConfig, bars_per_day: int, horizon: int) -> rr.Criteria:
    """Critères du projet ; longueur des blocs ramenée en périodes de ``horizon`` barres."""
    base = criteria(config)
    block = max(1.0, base.mean_block * bars_per_day / horizon)
    return dataclasses.replace(base, mean_block=block)


def _finite(metrics: object) -> dict[str, float | None]:
    """Mesures d'un ``lt_report.Metrics`` ; un ratio infini (aucune perte) devient ``None``."""
    values = dataclasses.asdict(metrics)  # type: ignore[call-overload]
    return {k: (v if np.isfinite(v) else None) for k, v in values.items()}


def _find(fiches: list[Hypothesis], ident: str) -> Hypothesis:
    for h in fiches:
        if h.id == ident:
            return h
    raise DataError(f"fiche inconnue : {ident}")


def best(ctx: Context, study: Study, fiches: list[Hypothesis], top: int) -> int:
    """Verdicts complets des ``top`` meilleurs essais du dernier balayage du volet."""
    latest = ctx.paths.reports / f"{study.volet}_scan_latest.json"
    if not latest.exists():
        raise DataError(f"aucun balayage du volet {study.volet} : lancer d'abord le balayage")
    rows = json.loads(latest.read_text())["rows"][:top]
    registry = TrialRegistry(ctx.paths.trials)
    parts = [f"# Volet {study.volet} — verdicts du {date_str(now_ms())}", ""]
    verdicts = []
    for row in rows:
        hyp, params = _find(fiches, row["id"]), {k: float(v) for k, v in row["params"].items()}
        ev = evidence(study, hyp, params, registry)
        crit = timing_criteria(ctx.config, study.bars_per_day, study.horizon(params))
        report = rr.evaluate(ev, crit)
        parts += [rr.render(report), ""]
        verdicts.append(
            {"id": hyp.id, "params": row["params"], "verdict": report.verdict, **report.headline}
        )
        print(f"{ev.name} : {report.verdict}", flush=True)
    oracle = oracle_sharpe(study, study.horizon(rows[0]["params"])) if rows else float("nan")
    parts.append(
        f"Contrôle anti-fuite : signal tricheur (rendement à venir) ⇒ Sharpe {oracle:+.1f}."
    )
    parts.append(study.note)
    day = date_str(now_ms())
    write_atomic(
        ctx.paths.reports / f"{study.volet}_verdict_{day}.md",
        "\n".join(parts).encode(),
        overwrite=True,
    )
    blob = json.dumps(
        {"date": day, "verdicts": verdicts, "oracle_sharpe": oracle}, default=float
    ).encode()
    write_atomic(ctx.paths.reports / f"{study.volet}_verdict_latest.json", blob, overwrite=True)
    return 0


def holdout(ctx: Context, study: Study, fiches: list[Hypothesis], ident: str, spec: str) -> int:
    """Rejoue l'essai ``ident`` (``spec`` : ``k=v,k=v``) une seule fois sur le hold-out."""
    previous = read_records(study.holdout_path)
    if previous:
        raise DataError(f"hold-out déjà utilisé ({previous[0]['name']}) : il ne se rejoue pas")
    hyp = _find(fiches, ident)
    params = {k: float(v) for k, v in (p.split("=") for p in spec.split(","))}
    if params not in [dict(g) for g in hyp.grid()]:
        raise DataError(f"{hyp.id} : combinaison {params} non déclarée dans la fiche")
    h = study.horizon(params)
    full = pn.complete_only(returns_on(study, study.panel, hyp, params), h)
    held = full.filter(pl.col("bar") >= study.holdout_bar)
    bench = held.select("bar").join(study.benchmark(h), on="bar", how="left")
    ppy = study.periods_per_year(h)
    m = lt_report.metrics(held["net"].to_numpy(), ppy)
    b = lt_report.metrics(bench["bench"].fill_null(0.0).to_numpy(), ppy)
    record = {
        "ts_ms": now_ms(),
        "name": f"{hyp.id} {spec}",
        "periods": held.height,
        "strategy": _finite(m),
        "benchmark": _finite(b),
    }
    append_record(study.holdout_path, record)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    best = sub.add_parser("best", help="verdict complet des meilleurs essais du balayage")
    best.add_argument("--top", type=int, default=5)
    hold = sub.add_parser("holdout", help="rejouer UN essai sur le hold-out (une seule fois)")
    hold.add_argument("--id", required=True)
    hold.add_argument("--params", required=True, help="ex. lookback_h=4,horizon_h=24")
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    parser.add_argument("--fiches", type=Path, default=scan.FICHES)


def _action(ctx: Context) -> int:
    fiches = load_all(ctx.args.fiches)
    study = scan.prepare(ctx)
    if ctx.args.cmd == "best":
        return best(ctx, study, fiches, ctx.args.top)
    return holdout(ctx, study, fiches, ctx.args.id, ctx.args.params)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.timing.verdict",
        description="Verdicts du volet timing et hold-out unique",
        component="timing_verdict",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
