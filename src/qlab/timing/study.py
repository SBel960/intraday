"""Étude d'un volet « meilleur moment d'achat / vente » : données, règles et balayage des fiches.

Commun aux perpétuels Binance (``timing/scan.py``, barres horaires) et aux marchés
multi-actifs (``markets/multi_asset.py``, jours ouvrés). Une fiche ``<préfixe>_<signal>`` essaie le
signal ``<signal>`` (``timing/hourly_signals.py``) avec ses paramètres ``lookback_<u>`` et
``horizon_<u>`` (``u`` : unité de la barre, ``h`` ou ``d``) ; ``n_long``, ``n_short`` et
``signed`` (0 / 1) de la fiche remplacent ceux de l'étude. Chaque essai est compté dans le
registre (N du DSR du volet) et classé sur la période **avant** le hold-out ; le budget est
vérifié avant de lancer quoi que ce soit.

Sorties : ``reports/{volet}_scan_{date}.md`` (classement) et ``{volet}_scan_latest.json``
(cockpit, verdicts : ``timing/verdict.py``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from qlab.core.cli import Context
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.timeutils import date_str, now_ms
from qlab.longterm.lt_evaluate import trial_result
from qlab.research.hypothesis import Hypothesis
from qlab.research.trials import TrialRegistry
from qlab.timing import hourly_signals as hs
from qlab.timing import panel as pn
from qlab.timing.portfolio import Book, period_returns


@dataclass(frozen=True, slots=True)
class Study:
    """Données et règles communes à tous les essais d'un volet.

    ``benchmark(h)`` : (``bar``, ``t_ms``, ``bench``), rendement de la référence par période ;
    ``bars_per_day`` : 24 (heures) ou 1 (jours ouvrés) ; ``holdout_path`` : registre du hold-out.
    """

    volet: str
    prefix: str
    unit: Literal["h", "d"]
    panel: pl.DataFrame
    costs: Mapping[str, float]
    default_cost: float
    n_long: int
    n_short: int
    weighting: Literal["equal", "inverse_vol"]
    vol_lookback: int
    holdout_bar: int
    bars_per_year: int
    bars_per_day: int
    benchmark: Callable[[int], pl.DataFrame]
    holdout_path: Path
    note: str

    def horizon(self, params: Mapping[str, float]) -> int:
        return int(params[f"horizon_{self.unit}"])

    def book(self, params: Mapping[str, float]) -> Book:
        """Portefeuille d'un essai : horizon de la fiche ; ``n_long``, ``n_short`` et ``signed``
        pris dans ses paramètres s'ils y sont, sinon dans l'étude."""
        return Book(
            self.horizon(params),
            int(params.get("n_long", self.n_long)),
            int(params.get("n_short", self.n_short)),
            self.costs,
            self.default_cost,
            signed=bool(params.get("signed", 0)),
            weighting=self.weighting,
            vol_lookback=self.vol_lookback,
        )

    def periods_per_year(self, horizon: int) -> int:
        return round(self.bars_per_year / horizon)

    def signal(
        self, panel: pl.DataFrame, hypothesis: Hypothesis, params: Mapping[str, float]
    ) -> pl.Series:
        return hs.compute(
            panel,
            self.signal_name(hypothesis),
            params.get(f"lookback_{self.unit}"),
            self.horizon(params),
        )

    def signal_name(self, hypothesis: Hypothesis) -> str:
        if not hypothesis.id.startswith(self.prefix):
            raise DataError(
                f"{hypothesis.id} : une fiche {self.volet} s'appelle {self.prefix}<signal>"
            )
        return hypothesis.id.removeprefix(self.prefix)


def returns_on(
    study: Study,
    panel: pl.DataFrame,
    hypothesis: Hypothesis,
    params: Mapping[str, float],
    signal: pl.Series | None = None,
) -> pl.DataFrame:
    """Rendements par période de l'essai sur ``panel`` (toute la période) ; ``signal`` : déjà
    calculé sur ce panel (le même signal sert à plusieurs horizons)."""
    sig = study.signal(panel, hypothesis, params) if signal is None else signal
    return period_returns(panel, sig, study.book(params))


def trial_returns(
    study: Study,
    hypothesis: Hypothesis,
    params: Mapping[str, float],
    signal: pl.Series | None = None,
) -> pl.DataFrame:
    """Rendements par période de l'essai, avant le hold-out."""
    out = returns_on(study, study.panel, hypothesis, params, signal)
    return pn.in_sample(out, study.holdout_bar, study.horizon(params))


def cached_signal(
    study: Study,
    hypothesis: Hypothesis,
    params: Mapping[str, float],
    cache: dict[tuple[str, float | None], pl.Series],
) -> pl.Series | None:
    """Signal réutilisable entre horizons (tous sauf la saisonnalité, qui dépend de h)."""
    name = hs.resolve(study.signal_name(hypothesis))
    if name == "seasonality":
        return None
    key = (name, params.get(f"lookback_{study.unit}"))
    if key not in cache:
        cache.clear()  # un seul signal en mémoire (panel de millions de lignes)
        cache[key] = study.signal(study.panel, hypothesis, params)
    return cache[key]


def summary_row(
    hypothesis: Hypothesis, params: Mapping[str, float], out: pl.DataFrame, ppy: int
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
        raise DataError(f"{total} essais déclarés > budget de {max_trials} essais du volet")
    return total


def render(rows: list[dict[str, object]], study: Study, n_trials: int) -> str:
    holdout = study.panel.filter(pl.col("bar") == study.holdout_bar)["t_ms"].min()
    lines = [
        f"# Volet {study.volet} — balayage du {date_str(now_ms())}",
        "",
        f"{n_trials} essais, période avant le hold-out "
        f"({date_str(holdout) if isinstance(holdout, int) else '?'}), levier 1×, rendements nets.",
        "",
        study.note,
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
            f"{r['funding_per_year']:+.1%} | {r['invested']:.0%} | {r['liquidations']} | "
            f"{r['start']} |"
        )
    return "\n".join(lines) + "\n"


def run_scan(ctx: Context, study: Study, fiches: Sequence[Hypothesis]) -> int:
    """Essaie chaque combinaison des ``fiches``, l'enregistre, écrit le classement."""
    registry = TrialRegistry(ctx.paths.trials)
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, float | None], pl.Series] = {}
    for hyp in fiches:
        for params in hyp.grid():
            out = trial_returns(study, hyp, params, cached_signal(study, hyp, params, cache))
            ppy = study.periods_per_year(study.horizon(params))
            registry.record(hyp, params, trial_result(out["net"].to_numpy(), ppy), ts_ms=now_ms())
            rows.append(summary_row(hyp, params, out, ppy))
            print(f"{hyp.id} {params} : Sharpe {rows[-1]['sharpe']:+.2f}", flush=True)
    rows.sort(key=lambda r: -float(r["sharpe"]))  # type: ignore[arg-type]
    n = registry.n_trials(study.volet)
    day = date_str(now_ms())
    report = ctx.paths.reports / f"{study.volet}_scan_{day}.md"
    write_atomic(report, render(rows, study, n).encode(), overwrite=True)
    blob = json.dumps({"date": day, "n_trials": n, "rows": rows}).encode()
    write_atomic(ctx.paths.reports / f"{study.volet}_scan_latest.json", blob, overwrite=True)
    ctx.journal.info(f"{study.volet}.scan", {"trials": len(rows), "n_trials": n})
    print(f"{len(rows)} essais ; N du volet = {n} ; rapport {report.name}")
    return 0
