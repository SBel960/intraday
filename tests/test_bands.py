"""Tests de qlab.timing.bands : échelle causale, hystérésis, barres non tradables — à la main."""

from __future__ import annotations

import polars as pl
import pytest

from qlab.core.errors import DataError
from qlab.timing import bands


def test_zscore_uses_only_previous_values() -> None:
    """Fenêtre 4 : à la 5e barre, échelle = écart-type de 1, −1, 1, −1 (= 1,1547) ; modifier la
    barre courante ne change pas l'échelle."""
    panel = pl.DataFrame({"symbol": ["A"] * 5})
    z = bands.zscore(panel, pl.Series([1.0, -1.0, 1.0, -1.0, 3.0]), 4)
    assert z[4] == pytest.approx(3.0 / 1.1547005)
    z2 = bands.zscore(panel, pl.Series([1.0, -1.0, 1.0, -1.0, 30.0]), 4)
    assert z2[4] == pytest.approx(30.0 / 1.1547005)
    assert z[0] is None  # pas encore d'historique
    with pytest.raises(DataError, match="fenêtre"):
        bands.zscore(panel, pl.Series([1.0] * 5), 1)


def test_hysteresis_enters_holds_exits_and_reverses() -> None:
    """Entrée 2, sortie 0,5 : 2,1 achat ; 1,0 tenu ; 0,4 sortie ; −2,5 vente ; −0,6 tenue ;
    3,0 retournement direct à l'achat. Nouvel actif : repart à plat."""
    z = pl.Series([0.0, 2.1, 1.0, 0.4, -2.5, -0.6, 3.0, 1.0])
    symbols = pl.Series(["A"] * 7 + ["B"])
    ok = pl.Series([True] * 8)
    got = bands.hysteresis(symbols, z, ok, 2.0, 0.5).to_list()
    assert got == [0, 1, 1, 0, -1, -1, 1, 0]
    with pytest.raises(DataError, match="sortie < entrée"):
        bands.hysteresis(symbols, z, ok, 1.0, 1.0)


def test_state_only_changes_on_tradable_bars() -> None:
    """Barre 2 fermée : la sortie attendue (0,1) y est ignorée, la position reste ; sortie à la
    barre 3. Barre 4 fermée avec z = 5 : pas d'entrée."""
    z = pl.Series([2.5, 1.0, 0.1, 0.1, 5.0])
    ok = pl.Series([True, True, False, True, False])
    got = bands.hysteresis(pl.Series(["A"] * 5), z, ok, 2.0, 0.5).to_list()
    assert got == [1, 1, 1, 0, 0]


def test_tradable_needs_eligibility_and_next_open() -> None:
    panel = pl.DataFrame(
        {
            "symbol": ["A"] * 4,
            "open": [1.0, None, 1.0, 1.0],
            "eligible": [True, True, False, True],
        }
    )
    # barre 0 : ouverture suivante absente ; 2 : non éligible ; 3 : pas de barre suivante
    assert bands.tradable(panel).to_list() == [False, True, False, False]
    panel = panel.with_columns(pl.Series("eligible", [True, True, True, True]))
    assert bands.tradable(panel).to_list() == [False, True, True, False]
