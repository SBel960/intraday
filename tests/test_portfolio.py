"""Tests de qlab.timing.portfolio : positions, coûts, financement, à plat — calculés à la main."""

from __future__ import annotations

import polars as pl
import pytest

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_HOUR, date_to_ms
from qlab.timing import portfolio as pf

H = MS_PER_HOUR
T0 = date_to_ms("2024-01-01")
OPENS = {"A": [100.0, 100, 110, 121], "B": [100.0, 100, 90, 90], "C": [100.0] * 4}
SIGNALS = {"A": [3.0, 1, None, None], "B": [1.0, 3, None, None], "C": [2.0, 2, None, None]}
COSTS = {"A": 0.001, "B": 0.002}


def _panel(funding_a: list[float] | None = None) -> tuple[pl.DataFrame, pl.Series]:
    rows = []
    for sym, opens in OPENS.items():
        for i, o in enumerate(opens):
            fund = (funding_a or [0.0] * 4)[i] if sym == "A" else 0.0
            rows.append(
                {
                    "symbol": sym,
                    "t_ms": T0 + i * H,
                    "open": o,
                    "close": o,
                    "funding": fund,
                    "eligible": True,
                    "s": SIGNALS[sym][i],
                }
            )
    df = pl.DataFrame(rows)
    return df.drop("s"), df["s"]


def test_weights_long_best_short_worst() -> None:
    panel, signal = _panel()
    w = pf.weights(panel, signal, pf.Book(1, 1, 1, COSTS, 0.0))
    assert w.select("t_ms", "symbol", "w").rows() == [
        (T0, "A", 0.5),
        (T0, "B", -0.5),
        (T0 + H, "A", -0.5),
        (T0 + H, "B", 0.5),
    ]


def test_period_returns_by_hand() -> None:
    """t0 : long A (+10 %), short B (−10 %) ⇒ brut 0,10 ; A paie 0,1 % de financement en heure 1
    (½ · 0,001) ; ouverture : ½ · 0,001 + ½ · 0,002. t1 : retournement (rotation 2) ; brut
    ½ · 0 − ½ · 10 % = −0,05. t2 : à plat, fermeture (rotation 1). t3 : rien."""
    panel, signal = _panel([0.0, 0.001, 0.0, 0.0])
    out = pf.period_returns(panel, signal, pf.Book(1, 1, 1, COSTS, 0.0))
    assert out["t_ms"].to_list() == [T0 + i * H for i in range(4)]
    assert out["gross"].to_list() == pytest.approx([0.10, -0.05, 0.0, 0.0])
    assert out["funding"].to_list() == pytest.approx([0.0005, 0.0, 0.0, 0.0])
    assert out["turnover"].to_list() == pytest.approx([1.0, 2.0, 1.0, 0.0])
    assert out["cost"].to_list() == pytest.approx([0.0015, 0.003, 0.0015, 0.0])
    assert out["net"].to_list() == pytest.approx([0.098, -0.053, -0.0015, 0.0])
    assert out["lines"].to_list() == [2, 2, 0, 0]


def test_too_few_candidates_means_flat_and_default_cost() -> None:
    """2 lignes de chaque côté avec 3 contrats : jamais assez ⇒ à plat partout."""
    panel, signal = _panel()
    out = pf.period_returns(panel, signal, pf.Book(1, 2, 2, {}, 0.01))
    assert out["net"].to_list() == [0.0] * 4
    ineligible = panel.with_columns(pl.lit(False).alias("eligible"))
    assert pf.weights(ineligible, signal, pf.Book(1, 1, 1, {}, 0.0)).is_empty()


def test_long_only_book() -> None:
    panel, signal = _panel()
    w = pf.weights(panel, signal, pf.Book(1, 1, 0, COSTS, 0.0))
    assert w.select("t_ms", "symbol", "w").rows() == [(T0, "A", 1.0), (T0 + H, "B", 1.0)]


def test_rebalance_every_h_hours_from_epoch() -> None:
    panel = pl.DataFrame({"t_ms": [T0 + 1 * H, T0 + 9 * H]})
    assert pf.rebalance_times(panel, 4).to_list() == [T0 + 4 * H, T0 + 8 * H]
    with pytest.raises(DataError, match="vide"):
        pf.rebalance_times(pl.DataFrame({"t_ms": []}, schema={"t_ms": pl.Int64}), 4)


def test_leg_costs_and_book_checks() -> None:
    """Jambe taker = frais + ½ spread + glissement : 0,0005 + 0,0002/2 + 0,0005 ; repli 0,002."""
    got = pf.leg_costs(
        ["A", "B"],
        {"A": 0.0002},
        taker_frac=0.0005,
        fallback_spread_frac=0.002,
        slippage_frac=0.0005,
    )
    assert got == pytest.approx({"A": 0.0011, "B": 0.002})
    with pytest.raises(DataError, match="au moins une ligne"):
        pf.Book(1, 0, 0, {}, 0.0)
    with pytest.raises(DataError, match="horizon"):
        pf.Book(0, 1, 1, {}, 0.0)


def test_buy_and_hold() -> None:
    panel, _ = _panel()
    bench = pf.buy_and_hold(panel, "A", 1)
    assert bench["bench"].to_list() == pytest.approx([0.10, 0.10, 0.0, 0.0])
    assert pf.buy_and_hold(panel, "ZZZ", 1)["bench"].to_list() == [0.0] * 4
