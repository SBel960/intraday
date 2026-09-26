"""Tests de qlab.longterm.signals : poids calculés à la main, absence de regard vers l'avenir."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.longterm import signals as sg

D = MS_PER_DAY
T0 = date_to_ms("2024-01-01")


def _panel(**cols: list[float | None]) -> pl.DataFrame:
    n = len(next(iter(cols.values())))
    return pl.DataFrame(
        {sg.DATE: [T0 + i * D for i in range(n)], **cols},
        schema={sg.DATE: pl.Int64, **dict.fromkeys(cols, pl.Float64)},
    )


def _w(weights: pl.DataFrame, asset: str) -> list[float]:
    return [round(x, 6) for x in weights[asset].to_list()]


def test_wide_builds_a_complete_daily_grid() -> None:
    a = pl.DataFrame({"open_time_ms": [T0, T0 + 2 * D], "close": [1.0, 3.0]})
    b = pl.DataFrame({"open_time_ms": [T0 + D], "close": [5.0]})
    panel = sg.wide({"A": a, "B": b})
    assert panel[sg.DATE].to_list() == [T0, T0 + D, T0 + 2 * D]
    assert panel["A"].to_list() == [1.0, None, 3.0]  # trou gardé, jamais comblé
    assert panel["B"].to_list() == [None, 5.0, None]


def test_equal_weight() -> None:
    """Jour 0 : A et B présents ⇒ 1/2 chacun ; jour 1 : B absent ⇒ A reçoit tout."""
    w = sg.equal_weight(_panel(A=[1, 1], B=[1, None]))
    assert _w(w, "A") == [0.5, 1.0] and _w(w, "B") == [0.5, 0.0]


def test_log_return_with_skip_by_hand() -> None:
    """C = 1 2 4 8 16, L = 3, S = 1 : t = 3 → ln(C₂ / C₀) = ln 4 ; t = 4 → ln(C₃ / C₁) = ln 4."""
    r = sg.log_return(_panel(A=[1, 2, 4, 8, 16]), 3, 1)["A"].to_list()
    assert r[:3] == [None, None, None]
    assert r[3:] == pytest.approx([math.log(4), math.log(4)])


def test_ts_momentum_by_hand() -> None:
    """A monte, B baisse : A reçoit 1/2 dès que ln(C_t / C_{t−2}) existe ; B jamais. Jour 4 :
    B absent ⇒ N = 1 ⇒ A reçoit tout."""
    w = sg.ts_momentum(_panel(A=[1, 2, 3, 4, 5], B=[5, 4, 3, 2, None]), 2)
    assert _w(w, "A") == [0, 0, 0.5, 0.5, 1.0]
    assert _w(w, "B") == [0, 0, 0, 0, 0]


def test_market_breadth_by_hand() -> None:
    closes = _panel(A=[1, 1, 1], B=[1, 1, 1])
    breadth = pl.DataFrame({sg.DATE: [T0, T0 + D], "breadth": [0.2, 0.6]})
    w = sg.market_breadth(closes, breadth, 0.5)
    assert _w(w, "A") == [0, 0.5, 0]  # jour 2 : largeur inconnue ⇒ cash


def test_funding_leverage_by_hand() -> None:
    """Fenêtre 3 j, quantile 0,5 (médiane), financement 1 2 3 3 2 1 : jours 0-1 fenêtre
    incomplète ⇒ cash ; jour 2 : 3 > méd(1, 2, 3) = 2 ⇒ cash ; jour 3 : 3 ≤ méd(2, 3, 3) = 3,
    jour 4 : 2 ≤ méd(3, 3, 2) = 3, jour 5 : 1 ≤ méd(3, 2, 1) = 2 ⇒ investi."""
    closes = _panel(A=[1.0] * 6)
    funding = _panel(funding_1d=[1, 2, 3, 3, 2, 1])
    assert _w(sg.funding_leverage(closes, funding, 0.5, 3), "A") == [0, 0, 0, 1, 1, 1]


def test_low_volatility_by_hand() -> None:
    """Rendements ±1 % pour A et ±2 % pour B : σ_B = 2 σ_A ⇒ poids 2/3 et 1/3."""
    steps = [0.01, -0.01] * 4
    a = np.exp(np.concatenate([[0], np.cumsum(steps)]))
    b = np.exp(np.concatenate([[0], np.cumsum(np.array(steps) * 2)]))
    w = sg.low_volatility(_panel(A=list(a), B=list(b)), 4)
    assert _w(w, "A")[:4] == [0, 0, 0, 0]  # 4 rendements nécessaires : dès le jour 4
    assert w["A"][4:].to_list() == pytest.approx([2 / 3] * 5)
    assert w["B"][4:].to_list() == pytest.approx([1 / 3] * 5)


def test_top_k_by_hand() -> None:
    scores = _panel(A=[3.0], B=[1.0], C=[2.0], E=[None])
    w = sg.top_k(scores, 2)
    assert [w[a][0] for a in "ABCE"] == [0.5, 0, 0.5, 0]
    members = pl.DataFrame({sg.DATE: [T0], "A": [False], "B": [True], "C": [True], "E": [None]})
    w = sg.top_k(scores, 2, members)
    assert [w[a][0] for a in "ABCE"] == [0, 0.5, 0.5, 0]
    w = sg.top_k(scores, 5)  # 3 scores valides pour 5 places : 3/5 investis, reste en cash
    assert [w[a][0] for a in "ABCE"] == [0.2, 0.2, 0.2, 0]


def test_short_reversal_by_hand() -> None:
    """Sur 1 jour : A −10 %, B −30 %, C +20 % ⇒ tout sur B ; jour suivant tout monte ⇒ cash."""
    w = sg.short_reversal(_panel(A=[1, 0.9, 1], B=[1, 0.7, 0.8], C=[1, 1.2, 1.3]), 1)
    assert [w[a].to_list() for a in "ABC"] == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_turn_of_month_by_hand() -> None:
    """Du 27 janvier au 4 février, 3 derniers et 3 premiers jours : détenu les 29, 30, 31
    janvier et 1er, 2, 3 février ⇒ décidé la veille, lignes du 28 janvier au 2 février."""
    closes = pl.DataFrame(
        {sg.DATE: [date_to_ms("2024-01-27") + i * D for i in range(9)], "A": [1.0] * 9}
    )
    w = sg.turn_of_month(closes, 3, 3)
    assert _w(w, "A") == [0, 1, 1, 1, 1, 1, 1, 0, 0]


def test_relative_rotation_by_hand() -> None:
    """L = 2. Jour 2 : B ln 2, C 0 ⇒ moyenne 0,35 > BTC 0 ⇒ B et C à 1/2. Jour 3 : B ln 2,
    C ln 0,5 ⇒ moyenne 0, pas > 0 ⇒ tout sur BTC. Jours 0-1 : inconnu ⇒ cash."""
    w = sg.relative_rotation(_panel(BTC=[1, 1, 1, 1], B=[1, 1, 2, 2], C=[1, 1, 1, 0.5]), "BTC", 2)
    assert w.columns == [sg.DATE, "BTC", "B", "C"]
    assert _w(w, "BTC") == [0, 0, 0, 1] and _w(w, "B") == [0, 0, 0.5, 0] == _w(w, "C")


def test_breakout_by_hand() -> None:
    """Entrée 2 j, sortie 1 j. Clôtures 1 2 3 2 5 4 1 6 : entrée au jour 2 (3 > 2), sortie
    au jour 3 (2 < 3), entrée au jour 4 (5 > 3), sortie au jour 5 (4 < 5), entrée au jour 7
    (6 > 4)."""
    w = sg.breakout(_panel(A=[1, 2, 3, 2, 5, 4, 1, 6]), 2)
    assert _w(w, "A") == [0, 0, 1, 0, 1, 0, 0, 1]


def test_volume_shock_by_hand() -> None:
    """Base 3 j, k = 2, détention 2 j. Jour 3 : volume 30 ≥ 2 × 10 et hausse ⇒ investi les
    jours 3 et 4. Jour 5 : volume 30 mais baisse ⇒ rien."""
    closes = _panel(A=[1, 1, 1, 2, 2, 1.5, 1.5])
    volumes = _panel(A=[10, 10, 10, 30, 10, 30, 10])
    w = sg.volume_shock(closes, volumes, ratio=2, baseline_days=3, hold_days=2)
    assert _w(w, "A") == [0, 0, 0, 1, 1, 0, 0]


def test_near_high_by_hand() -> None:
    """Fenêtre 3 j, q = 0,8 : jour 2 → 9 ≥ 0,8 × 10 ; jour 3 → 5 < 0,8 × 9 ; jour 4 → 10."""
    w = sg.near_high(_panel(A=[10, 8, 9, 5, 10]), 0.8, 3)
    assert _w(w, "A") == [0, 0, 1, 0, 1]


Signal = Callable[[pl.DataFrame], pl.DataFrame]
SIGNALS: dict[str, Signal] = {
    "ts_momentum": lambda c: sg.ts_momentum(c, 5),
    "low_volatility": lambda c: sg.low_volatility(c, 5),
    "short_reversal": lambda c: sg.short_reversal(c, 3),
    "xs_momentum": lambda c: sg.xs_momentum(
        c, c.select(sg.DATE, *(pl.col(a).is_not_null() for a in "ABCD")), 6, 2, 2
    ),
    "turn_of_month": lambda c: sg.turn_of_month(c, 1, 3),
    "relative_rotation": lambda c: sg.relative_rotation(c, "A", 5),
    "breakout": lambda c: sg.breakout(c, 6),
    "volume_shock": lambda c: sg.volume_shock(c, c, ratio=1.02, baseline_days=5, hold_days=4),
    "near_high": lambda c: sg.near_high(c, 0.9, 10),
}


@pytest.mark.parametrize("name", sorted(SIGNALS))
def test_future_data_never_changes_past_weights(name: str) -> None:
    """Remplacer les 20 derniers jours par n'importe quoi ne change aucun poids antérieur."""
    rng = np.random.default_rng(0)
    prices = {a: list(np.exp(np.cumsum(rng.normal(0, 0.03, 60)))) for a in "ABCD"}
    panel = _panel(**prices)
    other = panel.with_columns(
        pl.when(pl.int_range(pl.len()) >= 40).then(pl.col(a) * 7).otherwise(pl.col(a))
        for a in "ABCD"
    )
    before, after = SIGNALS[name](panel)[:40], SIGNALS[name](other)[:40]
    assert before.equals(after)


def test_weights_are_long_only_and_at_most_fully_invested() -> None:
    rng = np.random.default_rng(1)
    panel = _panel(**{a: list(np.exp(np.cumsum(rng.normal(0, 0.03, 80)))) for a in "ABC"})
    for signal in SIGNALS.values():
        w = signal(panel.with_columns(D=pl.col("A") * 2)).drop(sg.DATE).to_numpy()
        assert (w >= 0).all() and (w.sum(axis=1) <= 1 + 1e-12).all()


def test_describe() -> None:
    w = sg.ts_momentum(_panel(A=[1, 2, 3, 4, 5], B=[5, 4, 3, 2, None]), 2)
    assert (
        sg.describe(w) == "2024-01-01 → 2024-01-05 : exposition moyenne 0.40, investi 60% des jours"
    )


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: sg.wide({}), "aucune série"),
        (lambda: sg.log_return(_panel(A=[1.0]), 0), "≥ 1"),
        (lambda: sg.log_return(_panel(A=[1.0]), 3, 3), "skip"),
        (lambda: sg.low_volatility(_panel(A=[1.0]), 1), "≥ 2"),
        (lambda: sg.top_k(_panel(A=[1.0]), 0), "top_k"),
        (lambda: sg.market_breadth(_panel(A=[1.0]), _panel(A=[1.0]), 1.5), "min_breadth"),
        (lambda: sg.funding_leverage(_panel(A=[1.0]), _panel(A=[1.0]), 1.0, 3), "quantile"),
        (lambda: sg.turn_of_month(_panel(A=[1.0]), 0, 3), "≥ 1"),
        (lambda: sg.relative_rotation(_panel(A=[1.0]), "A", 2), "rotation"),
        (lambda: sg.breakout(_panel(A=[1.0]), 1), "entry_days"),
        (
            lambda: sg.volume_shock(
                _panel(A=[1.0]), _panel(A=[1.0]), ratio=1, baseline_days=3, hold_days=2
            ),
            "ratio",
        ),
        (lambda: sg.near_high(_panel(A=[1.0]), 1.5, 3), "min_ratio"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]
