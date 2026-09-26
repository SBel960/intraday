"""Verdict d'une stratégie : critères d'acceptation (§7 et LT.6) évalués, puis rapport Markdown.

Entrées (``Evidence``) : rendements **nets de coûts** de la stratégie et du buy & hold aux
mêmes dates, étiquette de sous-période de chaque observation, résultats par actif, nombre
d'essais du volet et variance de leurs Sharpe (``research/trials.py``). Les seuils
(``Criteria``) viennent de la config de l'appelant ; rien n'est supposé ici, pas même le
calendrier (``periods_per_year`` fourni par le marché).

Critères, tous requis :

1. DSR > ``dsr_min`` (N essais du volet) ;
2. écart de Sharpe stratégie − buy & hold : IC bootstrap stationnaire strictement > 0 ;
3. stabilité : écart de Sharpe > 0 dans **chaque** sous-période, au moins ``min_subperiods``
   sous-périodes, dont un marché baissier si ``require_bear`` (baissier = buy & hold composé
   négatif sur la sous-période) ;
4. écart de Sharpe > 0 sur au moins ``min_assets`` actifs ;
5. historique ≥ MinTRL (contre SR*, le meilleur Sharpe attendu par hasard) ;
6. paper trading dans l'intervalle du backtest — évalué ailleurs, transmis ici (``None`` :
   pas encore fait).

Verdict : paper hors intervalle, critère 2-4 manqué, ou Sharpe ≤ SR* ⇒ **rejeté** ; sinon
historique < MinTRL ⇒ **non concluant** (Sharpe au-dessus du hasard mais pas assez de données :
à même seuil, DSR et MinTRL disent la même chose) ; sinon DSR manqué ⇒ rejeté ; sinon paper non
fait ⇒ **candidat au paper trading** ; sinon **accepté**.

Rendements **simples** par période (le composé du buy & hold décide « baissier »).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from qlab.core.errors import DataError
from qlab.research import stats
from qlab.research.bootstrap import bootstrap, sharpe_difference

Floats = npt.NDArray[np.float64]

_PAPER = {
    None: "pas encore fait",
    True: "dans l'intervalle du backtest",
    False: "hors de l'intervalle du backtest",
}

REJECTED = "rejeté"
INCONCLUSIVE = "non concluant"
CANDIDATE = "candidat au paper trading"
ACCEPTED = "accepté"


@dataclass(frozen=True, slots=True)
class Criteria:
    """Seuils d'acceptation et réglages du bootstrap (config de l'appelant)."""

    dsr_min: float
    confidence: float  # IC bootstrap ; MinTRL au risque 1 − confidence
    min_subperiods: int
    min_assets: int
    require_bear: bool
    n_boot: int
    mean_block: float
    seed: int


@dataclass(frozen=True, slots=True)
class Evidence:
    """Résultats d'une combinaison de paramètres ; toutes les séries sont nettes de coûts."""

    name: str
    strategy: Floats
    benchmark: Floats
    periods: Sequence[str]
    per_asset: Mapping[str, tuple[Floats, Floats]]
    n_trials: int
    sharpe_variance: float
    periods_per_year: int
    paper_ok: bool | None = None


@dataclass(frozen=True, slots=True)
class Check:
    """Un critère : ``passed`` à ``None`` = pas encore évaluable."""

    name: str
    passed: bool | None
    detail: str


@dataclass(frozen=True, slots=True)
class SubResult:
    """Sharpe annualisés stratégie et buy & hold sur une sous-période ou un actif."""

    label: str
    n: int
    sharpe: float
    benchmark_sharpe: float
    bear: bool = False

    @property
    def beats(self) -> bool:
        return self.sharpe > self.benchmark_sharpe


@dataclass(frozen=True, slots=True)
class Report:
    name: str
    verdict: str
    checks: tuple[Check, ...]
    headline: dict[str, float] = field(default_factory=dict)
    subperiods: tuple[SubResult, ...] = ()
    assets: tuple[SubResult, ...] = ()


def _sub(label: str, strat: Floats, bench: Floats, ppy: int) -> SubResult:
    return SubResult(
        label,
        strat.size,
        stats.annualize(stats.moments(strat).sharpe, ppy),
        stats.annualize(stats.moments(bench).sharpe, ppy),
        bear=float(np.prod(1 + bench)) < 1,
    )


def _subperiods(ev: Evidence) -> tuple[SubResult, ...]:
    """Groupes contigus d'étiquettes identiques ; une étiquette qui revient est une erreur."""
    labels = list(ev.periods)
    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    if len({labels[i] for i in starts}) != len(starts):
        raise DataError("sous-périodes : chaque étiquette doit former un seul bloc contigu")
    bounds = [*starts, len(labels)]
    return tuple(
        _sub(labels[a], ev.strategy[a:b], ev.benchmark[a:b], ev.periods_per_year)
        for a, b in pairwise(bounds)
    )


def _validate(ev: Evidence, cr: Criteria) -> None:
    n = np.asarray(ev.strategy).size
    if np.asarray(ev.benchmark).shape != (n,) or len(ev.periods) != n:
        raise DataError("stratégie, buy & hold et sous-périodes : mêmes dates attendues")
    if not 0 < cr.confidence < 1 or cr.min_subperiods < 1 or cr.min_assets < 1:
        raise DataError("critères : confidence dans ]0, 1[, minimums ≥ 1")


def _stability(subs: tuple[SubResult, ...], cr: Criteria) -> Check:
    beaten = sum(s.beats for s in subs)
    bears = sum(s.bear for s in subs)
    ok = beaten == len(subs) >= cr.min_subperiods and (bears > 0 or not cr.require_bear)
    bear_txt = f", dont {bears} baissière(s)" if cr.require_bear else ""
    return Check(
        "stabilité par sous-période",
        ok,
        f"bat le buy & hold dans {beaten}/{len(subs)} sous-périodes "
        f"(minimum {cr.min_subperiods}{bear_txt})",
    )


def evaluate(ev: Evidence, cr: Criteria) -> Report:
    """Calcule tous les critères et le verdict."""
    _validate(ev, cr)
    ppy = ev.periods_per_year
    m = stats.moments(ev.strategy)
    sr_star = stats.expected_max_sharpe(ev.n_trials, ev.sharpe_variance)
    dsr = stats.psr(m.sharpe, sr_star, m.n, m.skew, m.kurtosis)
    min_trl = stats.min_track_record(m.sharpe, sr_star, m.skew, m.kurtosis, 1 - cr.confidence)
    diff = bootstrap(
        sharpe_difference,
        ev.strategy,
        ev.benchmark,
        n_boot=cr.n_boot,
        mean_block=cr.mean_block,
        seed=cr.seed,
    )
    low, high = diff.interval(cr.confidence)
    subs = _subperiods(ev)
    assets = tuple(_sub(k, s, b, ppy) for k, (s, b) in sorted(ev.per_asset.items()))
    winners = sum(a.beats for a in assets)
    checks = (
        Check("DSR", dsr > cr.dsr_min, f"{dsr:.3f} (seuil {cr.dsr_min}, N = {ev.n_trials})"),
        Check(
            "écart de Sharpe vs buy & hold",
            diff.strictly_positive(cr.confidence),
            f"IC {cr.confidence:.0%} par période [{low:+.4f} ; {high:+.4f}]",
        ),
        _stability(subs, cr),
        Check(
            "actifs",
            winners >= cr.min_assets,
            f"bat le buy & hold sur {winners}/{len(assets)} actifs (minimum {cr.min_assets})",
        ),
        Check("historique ≥ MinTRL", m.n >= min_trl, f"{m.n} observations, MinTRL {min_trl:.0f}"),
        Check("paper trading", ev.paper_ok, _PAPER[ev.paper_ok]),
    )
    headline = {
        "sharpe_annualise": stats.annualize(m.sharpe, ppy),
        "sharpe_buy_hold": stats.annualize(stats.moments(ev.benchmark).sharpe, ppy),
        "sharpe_lo": stats.lo_annualized_sharpe(ev.strategy, ppy, stats.newey_west_lags(m.n)),
        "tstat_newey_west": stats.newey_west_tstat(ev.strategy, stats.newey_west_lags(m.n)),
        "psr_0": stats.psr(m.sharpe, 0.0, m.n, m.skew, m.kurtosis),
        "dsr": dsr,
        "sr_star_annualise": stats.annualize(sr_star, ppy),
        "min_trl": min_trl,
    }
    return Report(
        ev.name, _verdict(checks, beats_chance=min_trl < math.inf), checks, headline, subs, assets
    )


def _verdict(checks: tuple[Check, ...], *, beats_chance: bool) -> str:
    dsr, *robustness, history, paper = checks
    if paper.passed is False or not all(c.passed for c in robustness) or not beats_chance:
        return REJECTED
    if not history.passed:
        return INCONCLUSIVE
    if not dsr.passed:
        return REJECTED
    return CANDIDATE if paper.passed is None else ACCEPTED


def _mark(passed: bool | None) -> str:
    return "—" if passed is None else ("oui" if passed else "**non**")


def _table(title: str, rows: tuple[SubResult, ...]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| | n | Sharpe | Buy & hold | Bat | Baissier |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.label} | {r.n} | {r.sharpe:+.2f} | {r.benchmark_sharpe:+.2f} | "
        f"{_mark(r.beats)} | {'oui' if r.bear else ''} |"
        for r in rows
    ]
    return [*lines, ""]


def render(report: Report) -> str:
    """Rapport Markdown : verdict, critères, chiffres clés, sous-périodes et actifs."""
    h = report.headline
    lines = [
        f"## {report.name}",
        "",
        f"**Verdict : {report.verdict}**",
        "",
        "| Critère | Rempli | Détail |",
        "|---|---|---|",
        *(f"| {c.name} | {_mark(c.passed)} | {c.detail} |" for c in report.checks),
        "",
        f"- Sharpe annualisé {h['sharpe_annualise']:+.2f} (Lo 2002 : {h['sharpe_lo']:+.2f}) ; "
        f"buy & hold {h['sharpe_buy_hold']:+.2f}",
        f"- Meilleur Sharpe attendu par hasard (SR*) : {h['sr_star_annualise']:+.2f} ; "
        f"PSR(0) {h['psr_0']:.3f} ; t Newey–West {h['tstat_newey_west']:+.2f}",
        "",
        *_table("Sous-périodes", report.subperiods),
        *_table("Actifs", report.assets),
    ]
    return "\n".join(lines)
