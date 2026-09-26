"""Tests de qlab.longterm.lt_wave : vague complète sur marché synthétique, registre, commande."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.config import load_config
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import funding, lt_backtest, lt_wave, strategies
from qlab.longterm.signals import DATE
from qlab.research.hypothesis import load_all
from qlab.research.trials import TrialRegistry

D = MS_PER_DAY
T0 = date_to_ms("2021-01-01")
DAYS = 1100  # 3 ans : assez d'années civiles pour les sous-périodes
TRADE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HYPOTHESES = Path(__file__).resolve().parent.parent / "hypotheses"
FICHES = load_all(HYPOTHESES)


def _prices(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.03, DAYS))) for s in TRADE}


def _setup(config_dir: Path) -> lt_wave.Setup:
    config = load_config(config_dir)
    paths = DataPaths(config_dir.parent / "data")
    SnapshotStore(paths, "binance").save(
        exchange_info([binance_symbol(s, "USDT") for s in TRADE]), FALLBACK_FEES, T0, "https://x"
    )
    snapshot = SnapshotStore(paths, "binance").latest()
    assert snapshot is not None
    dates = [T0 + i * D for i in range(DAYS)]
    closes = pl.DataFrame({DATE: dates, **_prices(0)})
    rng = np.random.default_rng(1)
    fund = pl.DataFrame({DATE: dates, "funding_1d": rng.normal(0.0003, 0.0002, DAYS)})
    breadth = pl.DataFrame({DATE: dates, "breadth": rng.uniform(0, 1, DAYS)})
    market = strategies.Market(
        closes, fund, {50: breadth, 100: breadth}, 365, config.longterm.signals
    )
    rules = lt_backtest.pair_rules(config, snapshot, {})
    return lt_wave.Setup(config, snapshot, market, closes, rules, config.base.capital_tiers[1])


def test_wave_records_every_backtested_trial_before_judging(
    config_dir: Path, tmp_path: Path
) -> None:
    """Palier de 200 € : chaque essai passe d'abord le gate ; ceux qui le passent sont
    backtestés et enregistrés ; N du rapport = essais enregistrés ; chaque essai retenu a son
    verdict, ses références et son test anti-fuite."""
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    body = lt_wave.run_wave(_setup(config_dir), FICHES, registry)
    gate_lines = [line for line in body.splitlines() if line.startswith("- lt_")]
    assert len(gate_lines) == 17
    kept = [line for line in gate_lines if " : testée (" in line]
    assert kept and registry.n_trials("longterm") == len(kept)
    assert f"N = {len(kept)} essais dans le volet" in body
    assert body.count("**Verdict :") == len(kept)
    rows = body.splitlines()
    assert sum(r.startswith("| Buy & hold |") for r in rows) == len(kept)
    assert sum(r.startswith("| DCA |") for r in rows) == len(kept)
    assert "Contrôle anti-fuite du pipeline" in body and ") : ok" in body
    assert "Filtres d'ordre (pas, minimum) du snapshot actuel" in body


def test_rerun_does_not_inflate_trial_count(config_dir: Path, tmp_path: Path) -> None:
    """Relancer la vague ré-enregistre les mêmes essais : N (combinaisons distinctes) ne
    bouge pas."""
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    setup = _setup(config_dir)
    lt_wave.run_wave(setup, FICHES, registry)
    first = registry.n_trials("longterm")
    lt_wave.run_wave(setup, FICHES, registry)
    assert registry.n_trials("longterm") == first


def _write_klines(paths: DataPaths, symbol: str, closes: np.ndarray) -> None:
    out = paths.lt_klines("1d", symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "open_time_ms": [T0 + i * D for i in range(closes.size)],
            "open": closes,
            "close": closes,
            "volume_quote": [2e6] * closes.size,
        }
    ).write_parquet(out)


def test_cli(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = DataPaths(config_dir.parent / "data")
    args = ["--config", str(config_dir), "run", "--hypotheses", str(HYPOTHESES)]
    assert lt_wave.main(args) == 1  # pas de snapshot
    SnapshotStore(paths, "binance").save(
        exchange_info([binance_symbol(s, "USDT") for s in TRADE]), FALLBACK_FEES, T0, "https://x"
    )
    for s, closes in _prices(2).items():
        _write_klines(paths, s, closes)
        key = f"futures/um/monthly/fundingRate/{s}/{s}-fundingRate-2021.zip"
        path = paths.raw_archive("binance_vision", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"{T0 + k * 8 * 3_600_000 + 1},8,0.0001" for k in range(3 * DAYS))
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("f.csv", "calc_time,funding_interval_hours,last_funding_rate\n" + rows)
    assert funding.main(["--config", str(config_dir), "build"]) == 0
    assert lt_wave.main([*args, "--tier", "inconnu"]) == 1
    assert lt_wave.main([*args, "--tier", "t1"]) == 0
    report = next(paths.reports.glob("lt_wave_*.md")).read_text()
    assert report.startswith("# Vague long terme") and "Gate de coûts au palier t1" in report
    assert TrialRegistry(paths.trials).n_trials("longterm") > 0
    assert "Rapport :" in capsys.readouterr().out


def test_never_invested_trial_counts_with_zero_sharpe() -> None:
    """Rendements tous nuls (jamais investi) : Sharpe 0 par convention, l'essai compte dans N."""
    r = lt_wave._result(np.zeros(10), 365)
    assert (r.sharpe, r.n_obs, r.skew, r.kurtosis) == (0.0, 10, 0.0, 3.0)
    assert lt_wave._result(np.array([0.01, -0.02, 0.03]), 365).sharpe != 0
