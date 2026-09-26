"""Tests de qlab.longterm.lt_report : mesures calculées à la main, MWR, DCA, rendu."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.longterm import lt_report as lr
from qlab.research import report as rr

D = MS_PER_DAY
T0 = date_to_ms("2024-12-30")


def test_metrics_by_hand() -> None:
    """r = +10 %, −50 %, +20 %, 0 sur 4 périodes = 1 an : valeur 1,1 → 0,55 → 0,66 → 0,66 ;
    CAGR −34 % ; MaxDD 0,55 / 1,1 − 1 = −50 %, 3 périodes sous le sommet ; Calmar −0,68 ;
    Sortino −0,05 / √(0,25 / 4) × 2 = −0,4."""
    r = [0.1, -0.5, 0.2, 0.0]
    m = lr.metrics(r, 4)
    assert m.cagr == pytest.approx(-0.34)
    assert m.max_drawdown == pytest.approx(-0.5) and m.drawdown_periods == 3
    assert m.calmar == pytest.approx(-0.68)
    assert m.sortino == pytest.approx(-0.4)
    assert m.vol == pytest.approx(float(np.std(r, ddof=1)) * 2)
    assert m.sharpe == pytest.approx(-0.05 / float(np.std(r, ddof=1)) * 2)


def test_no_drawdown() -> None:
    m = lr.metrics([0.01, 0.02, 0.01], 3)
    assert (m.max_drawdown, m.drawdown_periods, m.calmar) == (0.0, 0, math.inf)


def test_mwr_by_hand() -> None:
    """100 versés au départ, 110 reçus un an plus tard ⇒ 10 %. 100 au départ et 100 à mi-
    parcours, 210 à la fin : x = √(1 + r) vérifie 100 x² + 100 x = 210 ⇒ r = 6,703 %."""
    flows = [Decimal(100), Decimal(0), Decimal(0), Decimal(0), Decimal(0)]
    assert lr.mwr(flows, Decimal(110), 4) == pytest.approx(0.10)
    flows[2] = Decimal(100)
    x = (-100 + math.sqrt(100**2 + 4 * 100 * 210)) / 200
    assert lr.mwr(flows, Decimal(210), 4) == pytest.approx(x**2 - 1)


def test_dca_flows_and_labels() -> None:
    dates = [T0 + i * D for i in range(10)]
    flows = lr.dca_flows(dates, Decimal(10), 3)
    assert sorted((d - T0) // D for d in flows) == [0, 3, 6, 9]
    assert set(flows.values()) == {Decimal(10)}
    assert lr.year_labels(dates[:4]) == ["2024", "2025", "2025"]  # 31/12, 01/01, 02/01


def test_trial_result_matches_moments() -> None:
    r = np.random.default_rng(0).normal(0.001, 0.02, 500)
    t = lr.trial_result(r, 365)
    assert t.n_obs == 500 and t.periods_per_year == 365
    assert t.sharpe == pytest.approx(r.mean() / r.std(ddof=1))


def test_render_shows_verdict_references_and_notes() -> None:
    m = lr.metrics([0.01, -0.02, 0.03], 3)
    keys = ("sharpe_annualise", "sharpe_lo", "sharpe_buy_hold", "sr_star_annualise", "psr_0")
    headline = dict.fromkeys((*keys, "tstat_newey_west", "dsr", "min_trl"), 0.5)
    report = rr.Report("lt_x · a=1", rr.CANDIDATE, (rr.Check("DSR", True, "0.990"),), headline)
    text = lr.render(report, m, m, (m, 0.05), ["filtres actuels appliqués au passé"])
    assert "**Verdict : candidat au paper trading**" in text
    assert "| Stratégie |" in text and "| Buy & hold |" in text
    assert "| DCA |" in text and "MWR +5.0%" in text
    assert "- filtres actuels appliqués au passé" in text


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: lr.metrics([0.1], 4), "au moins 2"),
        (lambda: lr.metrics([0.1, -1.0], 4), "−100 %"),
        (lambda: lr.mwr([Decimal(0), Decimal(0)], Decimal(1), 4), "apport"),
        (lambda: lr.dca_flows([T0], Decimal(0), 7), "DCA"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
