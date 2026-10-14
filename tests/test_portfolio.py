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
                    "bar": (T0 + i * H) // H,
                    "open": o,
                    "high": o,
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


def test_signed_book_trades_only_with_the_sign() -> None:
    """Signé : achat si signal > 0, vente si < 0, le reste en cash. Signaux tous positifs ⇒
    aucune vente ; 2 lignes de chaque côté pour 3 contrats ⇒ livre partiel (¼ par ligne)."""
    panel, signal = _panel()
    centred = signal - 2  # t0 : A +1, B −1, C 0 ; t1 : A −1, B +1, C 0
    w = pf.weights(panel, centred, pf.Book(1, 1, 1, COSTS, 0.0, signed=True))
    assert w.select("t_ms", "symbol", "w").rows() == [
        (T0, "A", 0.5),
        (T0, "B", -0.5),
        (T0 + H, "A", -0.5),
        (T0 + H, "B", 0.5),
    ]
    w = pf.weights(panel, signal, pf.Book(1, 1, 1, COSTS, 0.0, signed=True))
    assert w.select("t_ms", "symbol", "w").rows() == [(T0, "A", 0.5), (T0 + H, "B", 0.5)]
    w = pf.weights(panel, centred, pf.Book(1, 2, 2, COSTS, 0.0, signed=True))
    assert w.filter(pl.col("t_ms") == T0).select("symbol", "w").rows() == [
        ("A", 0.25),
        ("B", -0.25),
    ]


def test_rebalance_every_h_bars_from_epoch() -> None:
    """Barres multiples de h présentes dans le panel ; T0 (2024-01-01) = heure 473 352, ÷ 4."""
    panel = pl.DataFrame({"t_ms": [T0 + i * H for i in range(10)]}).with_columns(
        (pl.col("t_ms") // H).alias("bar")
    )
    got = pf.rebalance_times(panel, 4)
    assert got["t_ms"].to_list() == [T0, T0 + 4 * H, T0 + 8 * H]
    with pytest.raises(DataError, match="vide"):
        pf.rebalance_times(panel.clear(), 4)


def test_inverse_vol_weights_and_basket() -> None:
    """A oscille de ±2 %, B de ±1 % : σ_B = σ_A / 2 ⇒ B pèse 2 fois plus (⅔ contre ⅓).
    Le panier achète tout, avec les mêmes poids, sans coûts."""
    rows = []
    for sym, amp in (("A", 0.02), ("B", 0.01)):
        price = 100.0
        for i in range(8):
            price *= 1 + (amp if i % 2 else -amp)
            rows.append(
                {
                    "symbol": sym,
                    "t_ms": T0 + i * H,
                    "bar": (T0 + i * H) // H,
                    "open": price,
                    "high": price,
                    "close": price,
                    "funding": 0.0,
                    "eligible": True,
                }
            )
    df = pl.DataFrame(rows)
    ones = pl.Series([1.0] * df.height)
    book = pf.Book(1, 2, 0, {}, 0.0, signed=True, weighting="inverse_vol", vol_lookback=4)
    w = pf.weights(df, ones, book).filter(pl.col("t_ms") == T0 + 5 * H)
    assert w.select("symbol", "w").rows() == [
        ("A", pytest.approx(1 / 3, rel=0.02)),
        ("B", pytest.approx(2 / 3, rel=0.02)),
    ]
    bench = pf.basket(df, 1, 4)
    held = pf.period_returns(df, ones, book)
    assert bench["bench"].to_list() == pytest.approx(held["gross"].to_list())


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
    with pytest.raises(DataError, match="volatilité"):
        pf.Book(1, 1, 1, {}, 0.0, vol_lookback=1)


def test_measured_spread_is_charged_per_bar() -> None:
    """Spread mesuré : A 0,2 % à t0 puis 0,6 % à t1, B sans mesure à t1 (sa médiane 0,4 %).
    Coût d'une jambe = frais (A 0,001, B 0,002) + ½ spread de la barre.
    t0 : ouverture A et B (½ chacun) : ½ · (0,001 + 0,001) + ½ · (0,002 + 0,002) = 0,003.
    t1 : retournement (rotation 1 sur chacun) : 1 · (0,001 + 0,003) + 1 · (0,002 + 0,002)."""
    panel, signal = _panel()
    spreads = {("A", 0): 0.002, ("A", 1): 0.006, ("B", 0): 0.004}
    panel = panel.with_columns(
        pl.struct("symbol", "t_ms")
        .map_elements(
            lambda r: spreads.get((r["symbol"], (r["t_ms"] - T0) // H)), return_dtype=pl.Float64
        )
        .alias("spread_frac")
    )
    out = pf.period_returns(panel, signal, pf.Book(1, 1, 1, COSTS, 0.0))
    assert out["cost"].to_list()[:2] == pytest.approx([0.003, 0.008])


def test_buy_and_hold() -> None:
    panel, _ = _panel()
    bench = pf.buy_and_hold(panel, "A", 1)
    assert bench["bench"].to_list() == pytest.approx([0.10, 0.10, 0.0, 0.0])
    assert pf.buy_and_hold(panel, "ZZZ", 1)["bench"].to_list() == [0.0] * 4


def test_short_liquidated_when_peak_doubles_even_if_price_falls_back() -> None:
    """Vente de B (poids −½) : le plus haut de la détention atteint 2,1 × l'entrée puis le prix
    retombe sous l'entrée. À 1×, la ligne est liquidée : perte = la mise (½), pas un gain. Une
    hausse de 900 % sans le plafond coûterait 4,5 × la mise."""
    rows = []
    for sym, opens, highs, s in (
        ("A", [100.0, 100, 100], [100.0, 100, 100], 3.0),
        ("B", [100.0, 100, 90], [100.0, 210, 90], 1.0),
        ("C", [100.0, 100, 1000], [100.0, 100, 1000], 2.0),
    ):
        for i in range(3):
            rows.append(
                {
                    "symbol": sym,
                    "t_ms": T0 + i * H,
                    "bar": (T0 + i * H) // H,
                    "open": opens[i],
                    "high": highs[i],
                    "close": opens[i],
                    "funding": 0.0,
                    "eligible": True,
                    "s": s if i == 0 else None,
                }
            )
    df = pl.DataFrame(rows)
    w = pf.weights(df.drop("s"), df["s"], pf.Book(1, 1, 1, {}, 0.0))
    b = w.filter(pl.col("symbol") == "B").row(0, named=True)
    assert b["liquidated"] and b["pnl"] == pytest.approx(-0.5)
    out = pf.period_returns(df.drop("s"), df["s"], pf.Book(1, 1, 1, {}, 0.0))
    assert out["liquidations"].to_list()[0] == 1
    assert out["net"].to_list()[0] == pytest.approx(-0.5)  # A : 0 ; B : −½


def test_loss_capped_at_the_stake_without_peak_data() -> None:
    """Même sans plus haut connu (exit seul), une ligne ne perd jamais plus que sa mise."""
    rows = [
        {
            "symbol": "A",
            "t_ms": T0 + i * H,
            "bar": (T0 + i * H) // H,
            "open": o,
            "high": o,
            "close": o,
            "funding": 0.0,
            "eligible": True,
        }
        for i, o in enumerate([100.0, 100.0, 1100.0])
    ]
    df = pl.DataFrame(rows)
    sig = pl.Series([1.0, None, None])
    w = pf.weights(df, sig, pf.Book(1, 0, 1, {}, 0.0))
    assert w["pnl"].to_list() == pytest.approx([-1.0])
