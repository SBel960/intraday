"""Tests de qlab.timing.study : budget, nom de signal, portefeuille d'un essai, résumé."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fakes import timing_fiche

from qlab.core.errors import DataError
from qlab.research.hypothesis import load_hypothesis
from qlab.timing import study as st


def _study(unit: str = "h") -> st.Study:
    return st.Study(
        volet="timing",
        prefix="tm_",
        unit=unit,  # type: ignore[arg-type]
        panel=pl.DataFrame(),
        costs={},
        default_cost=0.001,
        n_long=3,
        n_short=3,
        weighting="equal",
        vol_lookback=20,
        holdout_bar=0,
        bars_per_year=8640,
        bars_per_day=24,
        benchmark=lambda h: pl.DataFrame(),
        holdout_path=Path("x"),
        note="",
    )


def test_budget_and_signal_name(tmp_path: Path) -> None:
    fiche = load_hypothesis(timing_fiche(tmp_path, "tm_momentum", [1, 4]))
    assert st.check_budget([fiche], 2) == 2
    with pytest.raises(DataError, match="budget"):
        st.check_budget([fiche], 1)
    assert _study().signal_name(fiche) == "momentum"
    other = load_hypothesis(timing_fiche(tmp_path, "autre", [1]))
    with pytest.raises(DataError, match="tm_<signal>"):
        _study().signal_name(other)


def test_book_from_card_parameters_else_study() -> None:
    study = _study()
    plain = study.book({"horizon_h": 4})
    assert (plain.horizon, plain.n_long, plain.n_short, plain.signed) == (4, 3, 3, False)
    custom = study.book({"horizon_h": 8, "lookback_h": 2, "n_short": 0, "signed": 1})
    assert (custom.horizon, custom.n_long, custom.n_short, custom.signed) == (8, 3, 0, True)
    assert study.periods_per_year(4) == 2160  # 8 640 barres / 4
    days = _study("d")
    assert days.horizon({"lookback_d": 63, "horizon_d": 21}) == 21
    with pytest.raises(KeyError):
        days.horizon({"horizon_h": 21})


def test_summary_row_by_hand(tmp_path: Path) -> None:
    """Rendements nets 0,01 / −0,01 / 0,03 sur 3 périodes, 100 périodes par an."""
    fiche = load_hypothesis(timing_fiche(tmp_path, "tm_momentum", [1]))
    out = pl.DataFrame(
        {
            "t_ms": [0, 1, 2],
            "net": [0.01, -0.01, 0.03],
            "gross": [0.02, 0.0, 0.04],
            "cost": [0.01, 0.01, 0.01],
            "funding": [0.0, 0.0, 0.0],
            "lines": [2, 0, 2],
            "liquidations": [0, 1, 0],
        }
    )
    row = st.summary_row(fiche, {"lookback_h": 2, "horizon_h": 1}, out, 100)
    assert row["sharpe"] == pytest.approx(0.01 / 0.02 * 10)  # moyenne 0,01, écart-type 0,02
    assert row["net_per_year"] == pytest.approx(1.0)
    assert row["cost_per_year"] == pytest.approx(1.0)
    assert row["invested"] == pytest.approx(2 / 3)
    assert (row["liquidations"], row["periods"], row["start"]) == (1, 3, "1970-01-01")


def test_band_cards_get_a_persistent_risk_budget_book() -> None:
    book = _study().book({"horizon_h": 1, "lookback_h": 1, "enter_z": 2, "exit_z": 0})
    assert (book.persistent, book.signed, book.weighting, book.vol_lookback) == (
        True,
        True,
        "risk_budget",
        20,
    )
    assert st.is_band({"enter_z": 2}) and not st.is_band({"horizon_h": 1})


def test_equal_weight_card_overrides_the_study_weighting() -> None:
    study = st.Study(
        **{**{f: getattr(_study(), f) for f in st.Study.__slots__}, "weighting": "inverse_vol"}
    )
    assert study.book({"horizon_h": 1}).weighting == "inverse_vol"
    assert study.book({"horizon_h": 1, "equal_weight": 1}).weighting == "equal"
