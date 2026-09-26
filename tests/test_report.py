"""Tests de qlab.research.report : sous-périodes à la main, table du verdict, bout en bout."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.research.report import (
    ACCEPTED,
    CANDIDATE,
    INCONCLUSIVE,
    REJECTED,
    Check,
    Criteria,
    Evidence,
    SubResult,
    _stability,
    _verdict,
    evaluate,
    render,
)

CRIT = Criteria(
    dsr_min=0.95,
    confidence=0.95,
    min_subperiods=3,
    min_assets=2,
    require_bear=True,
    n_boot=300,
    mean_block=10,
    seed=0,
)
YEARS = ("2021", "2022", "2023", "2024")


def _market(seed: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """4 années de 365 jours, 2022 baissière ; stratégie = 30 % du marché + ``alpha`` + bruit."""
    rng = np.random.default_rng(seed)
    drift = np.repeat([0.004, -0.004, 0.004, 0.004], 365)
    bench = drift + rng.normal(0, 0.03, drift.size)
    return 0.3 * bench + alpha + rng.normal(0, 0.01, drift.size), bench


def _evidence(alpha: float, **kw: object) -> Evidence:
    strat, bench = _market(1, alpha)
    params: dict[str, object] = {
        "name": "lt_test · lookback=30",
        "strategy": strat,
        "benchmark": bench,
        "periods": [y for y in YEARS for _ in range(365)],
        "per_asset": {"BTC": _market(2, alpha), "ETH": _market(3, alpha)},
        "n_trials": 1,
        "sharpe_variance": 0.0,
        "periods_per_year": 365,
    }
    params.update(kw)
    return Evidence(**params)  # type: ignore[arg-type]


def test_subperiods_by_hand() -> None:
    """A : stratégie [0,01 ; 0,03] → SR 0,02 / 0,014142 = 1,4142 ; ×√4 = 2,8284 ; buy & hold
    [0,01 ; −0,03] → −0,01 / 0,028284 × 2 = −0,7071, composé 0,9797 < 1 ⇒ baissier.
    B : buy & hold [0,02 ; 0,01 ; 0,03] composé > 1."""
    ev = _evidence(
        0.0,
        strategy=np.array([0.01, 0.03, 0.0, 0.02, 0.01]),
        benchmark=np.array([0.01, -0.03, 0.02, 0.01, 0.03]),
        periods=["A", "A", "B", "B", "B"],
        periods_per_year=4,
    )
    loose = Criteria(0.0, 0.9, 1, 1, False, 20, 1, 0)
    a, b = evaluate(ev, loose).subperiods
    assert (a.label, a.n, b.label, b.n) == ("A", 2, "B", 3)
    assert a.sharpe == pytest.approx(2 * 0.02 / math.sqrt(2e-4))
    assert a.benchmark_sharpe == pytest.approx(-2 * 0.01 / math.sqrt(8e-4))
    assert a.bear and not b.bear and a.beats


def test_good_strategy_is_candidate_then_accepted_or_rejected_by_paper() -> None:
    report = evaluate(_evidence(0.0015), CRIT)
    assert report.verdict == CANDIDATE
    assert [c.passed for c in report.checks] == [True, True, True, True, True, None]
    assert [s.bear for s in report.subperiods] == [False, True, False, False]
    assert evaluate(_evidence(0.0015, paper_ok=True), CRIT).verdict == ACCEPTED
    assert evaluate(_evidence(0.0015, paper_ok=False), CRIT).verdict == REJECTED


def test_noise_is_rejected() -> None:
    """Sans alpha, 30 % du marché ne bat pas le buy & hold en Sharpe de façon fiable."""
    report = evaluate(_evidence(0.0), CRIT)
    assert report.verdict == REJECTED
    assert report.checks[1].passed is False


def test_many_trials_raise_the_bar() -> None:
    """Même stratégie, 1 000 essais très dispersés : SR* dépasse son Sharpe ⇒ rejetée."""
    report = evaluate(_evidence(0.0015, n_trials=1000, sharpe_variance=0.01), CRIT)
    assert report.headline["sr_star_annualise"] > report.headline["sharpe_annualise"]
    assert report.verdict == REJECTED and report.checks[0].passed is False


def test_missing_bear_market_fails_stability() -> None:
    ev = _evidence(0.0015)
    bull_only = _evidence(0.0015, benchmark=np.abs(ev.benchmark))
    stab = evaluate(bull_only, CRIT).checks[2]
    assert stab.passed is False and "dont 0/0 baissière" in stab.detail


def _sub(label: str, beats: bool, bear: bool) -> SubResult:
    return SubResult(label, 365, 1.0 if beats else 0.0, 0.5, bear)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("B+ b- B+ b+ B- B- B-", True),  # 3 battues dont 1 baissière (b = baissière)
        ("B+ B+ B+ b- b-", False),  # 3 battues, aucune baissière battue
        ("B+ b+ B- B- B-", False),  # 2 battues seulement
        ("B+ B+ B+ B+", False),  # aucune baissière du tout
    ],
)
def test_stability_rule_by_hand(pattern: str, expected: bool) -> None:
    """Au moins 3 sous-périodes battues, dont au moins une baissière (spec LT.6)."""
    subs = tuple(_sub(str(i), p[1] == "+", p[0] == "b") for i, p in enumerate(pattern.split()))
    assert _stability(subs, CRIT).passed is expected


def _checks(dsr: bool, robust: bool, history: bool, paper: bool | None) -> tuple[Check, ...]:
    return (
        Check("DSR", dsr, ""),
        Check("écart", robust, ""),
        Check("stabilité", True, ""),
        Check("actifs", True, ""),
        Check("MinTRL", history, ""),
        Check("paper", paper, ""),
    )


@pytest.mark.parametrize(
    "case",
    [
        (True, True, True, None, True, CANDIDATE),
        (True, True, True, True, True, ACCEPTED),
        (True, True, True, False, True, REJECTED),
        (True, False, True, None, True, REJECTED),
        (False, True, False, None, True, INCONCLUSIVE),  # au-dessus du hasard, trop court
        (False, True, False, None, False, REJECTED),  # Sharpe ≤ SR* : aucune donnée n'y changera
        (False, False, False, None, True, REJECTED),  # non robuste : rejeté même si court
        (False, True, True, None, True, REJECTED),  # DSR plus exigeant que MinTRL
    ],
)
def test_verdict_table(case: tuple[bool, bool, bool, bool | None, bool, str]) -> None:
    """(DSR, robustesse, historique, paper, Sharpe > SR*) → verdict."""
    dsr, robust, history, paper, beats_chance, expected = case
    assert _verdict(_checks(dsr, robust, history, paper), beats_chance=beats_chance) == expected


def test_render_lists_verdict_criteria_and_tables() -> None:
    text = render(evaluate(_evidence(0.0015), CRIT))
    assert "## lt_test · lookback=30" in text
    assert f"**Verdict : {CANDIDATE}**" in text
    assert "| paper trading | — | pas encore fait |" in text
    assert "| 2022 | 365 |" in text and "| BTC | 1460 |" in text
    assert text.count("| oui |") >= 5


@pytest.mark.parametrize(
    ("kw", "crit", "msg"),
    [
        ({"benchmark": np.zeros(10)}, CRIT, "mêmes dates"),
        ({"periods": ["2021"] * 1459}, CRIT, "mêmes dates"),
        ({"periods": ["A"] * 500 + ["B"] * 460 + ["A"] * 500}, CRIT, "contigu"),
        ({}, Criteria(0.95, 1.5, 3, 2, True, 10, 1, 0), "confidence"),
        ({}, Criteria(0.95, 0.95, 0, 2, True, 10, 1, 0), "minimums"),
    ],
)
def test_errors(kw: dict[str, object], crit: Criteria, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        evaluate(_evidence(0.0015, **kw), crit)
