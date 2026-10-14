"""Tests de qlab.trading.fx_panel : heures fermées, spread, éligibilité — à la main."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fakes import TRADING_T0, trading_series

from qlab.core.config import load_config
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_HOUR
from qlab.trading import fx_panel

H = MS_PER_HOUR


def test_closed_hours_and_eligibility(config_dir: Path) -> None:
    """Heures 0..49 depuis lundi 00:00 UTC, échangées sauf la 10 (fermée). Éligible après 24
    heures échangées (config de test) : heure 25 et suivantes, sauf 21 h et 22 h UTC du
    lendemain (heures 45 et 46, bascule exclue)."""
    cfg = load_config(config_dir).trading
    hours = [i for i in range(50) if i != 10]
    bars = pl.DataFrame(
        {
            "symbol": ["EURUSD"] * len(hours),
            "t_ms": [TRADING_T0 + i * H for i in hours],
            "open": [1.0 + i / 100 for i in hours],
            "high": [1.1 + i / 100 for i in hours],
            "low": [0.9 + i / 100 for i in hours],
            "close": [1.0 + i / 100 for i in hours],
            "spread_frac": [0.0001] * len(hours),
            "volume": [60.0] * len(hours),
        }
    )
    p = fx_panel.assemble(bars, cfg)
    closed = p.row(10, named=True)
    assert (closed["open"], closed["close"], closed["high"], closed["traded"]) == (
        None,
        pytest.approx(1.09),
        pytest.approx(1.09),
        False,
    )
    assert p["bar"].to_list() == [(TRADING_T0 + i * H) // H for i in range(50)]
    eligible = [i for i, ok in enumerate(p["eligible"].to_list()) if ok]
    assert eligible == [i for i in range(25, 50) if i not in (45, 46)]
    assert p["funding"].to_list() == [0.0] * 50


def test_load_needs_built_bars(config_dir: Path) -> None:
    cfg = load_config(config_dir).trading
    paths = DataPaths(config_dir.parent / "data")
    with pytest.raises(DataError, match="aucune barre"):
        fx_panel.load(paths, cfg)
    trading_series(paths)
    p = fx_panel.load(paths, cfg)
    assert sorted(p["symbol"].unique()) == ["EURUSD", "SPXUSD", "USDJPY"]
    assert p.filter(~pl.col("traded"))["open"].null_count() == p.filter(~pl.col("traded")).height
