"""Mesures et rapport d'un essai long terme : performances, références, verdict.

Tout part des rendements TWR quotidiens du backtest (apports neutralisés) et du calendrier du
marché (``periods_per_year``) :

- **CAGR** = (Π(1 + r))^(1/années) − 1 ; **volatilité** = σ(r) √q ;
- **MaxDD** = plus forte baisse depuis un sommet de la courbe de valeur, et sa **durée** (plus
  longue période passée sous un ancien sommet, en périodes) ;
- **Calmar** = CAGR / |MaxDD| ; **Sortino** = moyenne / √(moyenne des min(r, 0)²) × √q ;
- **MWR** : rendement de l'investisseur (taux interne), apports compris ; ≠ TWR dès qu'il y a
  des apports (DCA), et les deux sont toujours montrés séparément (LT.2) ;
- ``trial_result`` : moments pour le registre d'essais (DSR) ; ``year_labels`` : sous-périodes
  (années civiles) pour ``research/report.py``.

Le DCA de référence : ``dca_flows`` (montant fixe tous les ``period_days`` jours depuis le début).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import numpy.typing as npt
from scipy import optimize

from qlab.core.errors import DataError
from qlab.core.timeutils import date_str
from qlab.longterm.allocation import is_calendar_day
from qlab.research import report as research_report
from qlab.research import stats
from qlab.research.trials import TrialResult

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Metrics:
    cagr: float
    vol: float
    sharpe: float  # annualisé
    max_drawdown: float  # ≤ 0
    drawdown_periods: int
    calmar: float
    sortino: float


def _returns(returns: npt.ArrayLike) -> Floats:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.size < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise DataError("rendements : série 1D finie d'au moins 2 périodes, chacune > −100 %")
    return r


def metrics(returns: npt.ArrayLike, periods_per_year: int) -> Metrics:
    r = _returns(returns)
    years = r.size / periods_per_year
    curve = np.cumprod(1 + r)
    cagr = float(curve[-1] ** (1 / years) - 1)
    drawdown = curve / np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:] - 1
    max_dd = float(min(drawdown.min(), 0.0))
    under, longest = 0, 0
    for below in drawdown < 0:
        under = under + 1 if below else 0
        longest = max(longest, under)
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    q = math.sqrt(periods_per_year)
    return Metrics(
        cagr=cagr,
        vol=float(r.std(ddof=1)) * q,
        sharpe=stats.annualize(stats.moments(r).sharpe, periods_per_year),
        max_drawdown=max_dd,
        drawdown_periods=longest,
        calmar=cagr / abs(max_dd) if max_dd < 0 else math.inf,
        sortino=float(r.mean()) / downside * q if downside > 0 else math.inf,
    )


def mwr(flows: Sequence[Decimal], final_value: Decimal, periods_per_year: int) -> float:
    """Taux interne annualisé : apports ``flows[t]`` versés à la période t, valeur finale reçue
    à la dernière période. Sans apport ⇒ ``DataError``."""
    paid = [float(f) for f in flows]
    if not any(paid) or any(f < 0 for f in paid):
        raise DataError("MWR : au moins un apport, aucun retrait attendu")
    end = (len(paid) - 1) / periods_per_year

    def npv(rate: float) -> float:
        """Valeur actuelle nette au taux annuel ``rate`` (nulle au taux interne)."""
        discount = [math.pow(1 + rate, -t / periods_per_year) for t in range(len(paid))]
        paid_now = sum(f * d for f, d in zip(paid, discount, strict=True))
        return float(final_value) * math.pow(1 + rate, -end) - paid_now

    return float(optimize.brentq(npv, -0.9999, 1000.0))


def dca_flows(dates: Sequence[int], amount: Decimal, period_days: int) -> dict[int, Decimal]:
    """Apport de ``amount`` à la première date puis tous les ``period_days`` jours."""
    if amount <= 0 or period_days < 1 or not dates:
        raise DataError("DCA : montant > 0, période ≥ 1 jour, au moins une date")
    return {d: amount for d in dates if is_calendar_day(d, dates[0], period_days)}


def trial_result(returns: npt.ArrayLike, periods_per_year: int) -> TrialResult:
    """Moments d'un essai pour le registre (Sharpe par période, asymétrie, kurtosis)."""
    m = stats.moments(_returns(returns))
    return TrialResult(m.sharpe, m.n, m.skew, m.kurtosis, periods_per_year)


def year_labels(dates: Sequence[int]) -> list[str]:
    """Année civile de chaque rendement (le rendement t va de ``dates[t−1]`` à ``dates[t]``)."""
    return [date_str(d)[:4] for d in dates[1:]]


def _row(name: str, m: Metrics, extra: str) -> str:
    return (
        f"| {name} | {m.cagr:+.1%} | {m.vol:.1%} | {m.sharpe:+.2f} | {m.max_drawdown:.1%} | "
        f"{m.drawdown_periods} | {m.calmar:.2f} | {m.sortino:+.2f} | {extra} |"
    )


def render(
    report: research_report.Report,
    strategy: Metrics,
    buy_hold: Metrics,
    dca: tuple[Metrics, float],
    notes: Sequence[str],
) -> str:
    """Verdict et critères (``research/report.py``), puis la stratégie face au buy & hold et
    au DCA (TWR et MWR séparés), puis les limites de la simulation."""
    dca_twr, dca_mwr = dca
    lines = [
        research_report.render(report),
        "| | CAGR | Vol | Sharpe | MaxDD | Durée DD (j) | Calmar | Sortino | |",
        "|---|---|---|---|---|---|---|---|---|",
        _row("Stratégie", strategy, "TWR"),
        _row("Buy & hold", buy_hold, "TWR"),
        _row("DCA", dca_twr, f"TWR ; MWR {dca_mwr:+.1%}"),
        "",
        *(f"- {n}" for n in notes),
    ]
    return "\n".join(lines) + "\n"
