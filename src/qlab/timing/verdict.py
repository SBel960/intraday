"""Verdict des meilleurs essais du volet timing, puis test **unique** sur le hold-out.

``best`` : les ``--top`` meilleurs essais du dernier balayage (``timing_scan_latest.json``)
passent les critères complets de ``research/report.py`` (DSR sur N essais du volet, écart de
Sharpe avec le buy & hold du contrat de référence par bootstrap stationnaire, stabilité par
année civile, deux **moitiés disjointes** de l'univers jouant le rôle des « actifs », MinTRL).
Blocs du bootstrap : ``acceptance.mean_block_days`` converti en périodes de h heures.

``holdout`` : un essai, désigné à l'avance, est rejoué **une seule fois** sur les données à
partir de ``timing.holdout_start`` ; le résultat est inscrit (``DataPaths.timing_holdout``) et
un second hold-out est refusé : regarder plusieurs candidats sur ces données les rendrait
aussi suspectes que la période de recherche.

Commandes : ``python -m qlab.timing.verdict --config config best [--top 5]`` ;
``… holdout --id tm_x --params lookback_h=4,horizon_h=24``
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
from qlab.timing import hourly_signals as hs
from qlab.timing import panel as pn
from qlab.timing import scan
from qlab.timing.portfolio import buy_and_hold, period_returns

VOLET = "timing"


def halves(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Deux moitiés disjointes de l'univers (contrats triés, un sur deux). La clôture de la
    référence est déjà une colonne du panel : le signal de rattrapage marche dans chaque moitié."""
    symbols = panel["symbol"].unique().sort().to_list()
    return {
        f"moitié {i + 1}": panel.filter(pl.col("symbol").is_in(symbols[i::2])) for i in range(2)
    }


def sharpe_variance(registry: TrialRegistry, periods_per_year: int) -> float:
    """V[SR] du DSR à l'échelle de ``periods_per_year``. Les essais du volet n'ont pas la même
    période (1 h à 1 semaine) : chaque Sharpe par période est annualisé avec sa propre fréquence
    puis ramené à celle de l'essai jugé ; sans cela, les essais horaires gonflent SR*."""
    rescaled = [
        t.result.sharpe * np.sqrt(t.result.periods_per_year / periods_per_year)
        for t in registry.latest(VOLET).values()
    ]
    return float(np.var(rescaled, ddof=1)) if len(rescaled) > 1 else 0.0


def evidence(
    study: scan.Study,
    hyp: Hypothesis,
    params: dict[str, float],
    reference: str,
    registry: TrialRegistry,
) -> rr.Evidence:
    """Rendements nets avant hold-out, buy & hold de ``reference``, moitiés de l'univers."""
    h = int(params["horizon_h"])
    out = pn.in_sample(scan.returns_on(study, study.panel, hyp, params), study.holdout_ms, h)
    bench = out.select("t_ms").join(buy_and_hold(study.panel, reference, h), on="t_ms", how="left")
    per_asset = {}
    for label, part in halves(study.panel).items():
        sub = pn.in_sample(scan.returns_on(study, part, hyp, params), study.holdout_ms, h)
        sub = sub.join(bench, on="t_ms", how="inner")
        per_asset[label] = (sub["net"].to_numpy(), sub["bench"].fill_null(0.0).to_numpy())
    years = [date_str(t)[:4] for t in out["t_ms"].to_list()]
    return rr.Evidence(
        f"{hyp.id} {', '.join(f'{k}={int(v)}' for k, v in params.items())}",
        out["net"].to_numpy(),
        bench["bench"].fill_null(0.0).to_numpy(),
        years,
        per_asset,
        registry.n_trials(VOLET),
        sharpe_variance(registry, study.periods_per_year(h)),
        study.periods_per_year(h),
    )


def oracle_sharpe(study: scan.Study, horizon_h: int) -> float:
    """Contrôle anti-fuite (§8) : un « signal » qui connaît le rendement à venir doit donner un
    Sharpe énorme avec le même moteur et les mêmes coûts ; sinon le pipeline est suspect."""
    peek = study.panel.select(pn.forward_return(horizon_h).alias("s"))["s"]
    out = pn.in_sample(
        period_returns(study.panel, peek, study.book({"horizon_h": horizon_h})),
        study.holdout_ms,
        horizon_h,
    )
    r = out["net"].to_numpy()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(study.periods_per_year(horizon_h)))


def timing_criteria(config: QlabConfig, horizon_h: int) -> rr.Criteria:
    """Critères du projet ; longueur des blocs ramenée en périodes de ``horizon_h`` heures."""
    base = criteria(config)
    block = max(1.0, base.mean_block * hs.HOURS_PER_DAY / horizon_h)
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


def _best(ctx: Context, study: scan.Study, fiches: list[Hypothesis]) -> int:
    latest = ctx.paths.reports / "timing_scan_latest.json"
    if not latest.exists():
        raise DataError("aucun balayage : lancer d'abord « timing.scan run »")
    rows = json.loads(latest.read_text())["rows"][: ctx.args.top]
    registry = TrialRegistry(ctx.paths.trials)
    parts, verdicts = [f"# Volet timing — verdicts du {date_str(now_ms())}", ""], []
    for row in rows:
        hyp, params = _find(fiches, row["id"]), {k: float(v) for k, v in row["params"].items()}
        ev = evidence(study, hyp, params, ctx.config.timing.reference_symbol, registry)
        report = rr.evaluate(ev, timing_criteria(ctx.config, int(params["horizon_h"])))
        parts += [rr.render(report), ""]
        verdicts.append(
            {"id": hyp.id, "params": row["params"], "verdict": report.verdict, **report.headline}
        )
        print(f"{ev.name} : {report.verdict}", flush=True)
    oracle = oracle_sharpe(study, int(rows[0]["params"]["horizon_h"])) if rows else float("nan")
    parts.append(
        f"Contrôle anti-fuite : signal tricheur (rendement à venir) ⇒ Sharpe {oracle:+.1f}."
    )
    parts.append(study.spread_note)
    day = date_str(now_ms())
    write_atomic(
        ctx.paths.reports / f"timing_verdict_{day}.md", "\n".join(parts).encode(), overwrite=True
    )
    blob = json.dumps(
        {"date": day, "verdicts": verdicts, "oracle_sharpe": oracle}, default=float
    ).encode()
    write_atomic(ctx.paths.reports / "timing_verdict_latest.json", blob, overwrite=True)
    return 0


def _holdout(ctx: Context, study: scan.Study, fiches: list[Hypothesis]) -> int:
    previous = read_records(ctx.paths.timing_holdout)
    if previous:
        raise DataError(f"hold-out déjà utilisé ({previous[0]['name']}) : il ne se rejoue pas")
    hyp = _find(fiches, ctx.args.id)
    params = {k: float(v) for k, v in (p.split("=") for p in ctx.args.params.split(","))}
    if params not in [dict(g) for g in hyp.grid()]:
        raise DataError(f"{hyp.id} : combinaison {params} non déclarée dans la fiche")
    h = int(params["horizon_h"])
    full = pn.complete_only(scan.returns_on(study, study.panel, hyp, params), h)
    held = full.filter(pl.col("t_ms") >= study.holdout_ms)
    bench = held.select("t_ms").join(
        buy_and_hold(study.panel, ctx.config.timing.reference_symbol, h), on="t_ms", how="left"
    )
    ppy = study.periods_per_year(h)
    m = lt_report.metrics(held["net"].to_numpy(), ppy)
    b = lt_report.metrics(bench["bench"].fill_null(0.0).to_numpy(), ppy)
    name = f"{hyp.id} {ctx.args.params}"
    record = {
        "ts_ms": now_ms(),
        "name": name,
        "periods": held.height,
        "strategy": _finite(m),
        "buy_hold": _finite(b),
    }
    append_record(ctx.paths.timing_holdout, record)
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
    return _best(ctx, study, fiches) if ctx.args.cmd == "best" else _holdout(ctx, study, fiches)


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
