"""Tests de qlab.research.trials : comptage de N, combinaisons déclarées, fiche figée."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from qlab.core.errors import DataError
from qlab.research import trials as tr
from qlab.research.hypothesis import Hypothesis, load_hypothesis
from qlab.research.trials import TrialRegistry, TrialResult

T = 1_704_067_200_000
FICHE: dict[str, Any] = {
    "id": "ts_momentum",
    "title": "Momentum",
    "volet": "longterm",
    "created": "2026-09-26",
    "mechanism": "Les investisseurs sous-réagissent aux tendances : elles persistent un temps.",
    "horizon_s": 604800,
    "universe": "trade",
    "features": ["close_1d"],
    "parameters": [
        {"name": "lookback_days", "values": [30, 90]},
        {"name": "top_k", "values": [1, 3]},
    ],
    "cost_round_trip_frac": 0.003,
    "min_edge_frac": 0.03,
    "edge_basis": "per_year",
    "abandon": "DSR < 0,95 ou Sharpe après coûts inférieur au buy & hold.",
}
RESULT = TrialResult(sharpe=0.05, n_obs=365, skew=-0.5, kurtosis=6.0, periods_per_year=365)


def _hyp(tmp_path: Path, **patch: Any) -> Hypothesis:
    path = tmp_path / "ts_momentum.yaml"
    path.write_text(yaml.safe_dump({**FICHE, **patch}, allow_unicode=True), encoding="utf-8")
    return load_hypothesis(path)


def test_n_counts_distinct_combinations(tmp_path: Path) -> None:
    reg, h = TrialRegistry(tmp_path / "trials.jsonl"), _hyp(tmp_path)
    reg.record(h, {"lookback_days": 30, "top_k": 1}, RESULT, ts_ms=T)
    reg.record(h, {"top_k": 3, "lookback_days": 30}, RESULT, ts_ms=T + 1)  # ordre indifférent
    rerun = TrialResult(0.07, 365, -0.5, 6.0, 365)
    reg.record(h, {"lookback_days": 30, "top_k": 1}, rerun, ts_ms=T + 2)  # relance
    assert reg.n_trials("longterm") == 2
    assert sorted(reg.sharpes("longterm")) == [0.05, 0.07]  # la dernière mesure fait foi
    assert reg.n_trials("intraday") == 0
    assert len(reg.trials()) == 3  # tout reste dans le registre (append-only)


def test_undeclared_combination_refused(tmp_path: Path) -> None:
    reg, h = TrialRegistry(tmp_path / "trials.jsonl"), _hyp(tmp_path)
    with pytest.raises(DataError, match="non déclarée"):
        reg.record(h, {"lookback_days": 45, "top_k": 1}, RESULT, ts_ms=T)
    with pytest.raises(DataError, match="non déclarée"):
        reg.record(h, {"lookback_days": 30}, RESULT, ts_ms=T)
    assert reg.n_trials("longterm") == 0


def test_fiche_modified_after_trials_refused(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.record(_hyp(tmp_path), {"lookback_days": 30, "top_k": 1}, RESULT, ts_ms=T)
    loosened = _hyp(tmp_path, min_edge_frac=0.01)  # on baisse l'exigence après coup…
    with pytest.raises(DataError, match="fiche modifiée après des essais"):
        reg.record(loosened, {"lookback_days": 30, "top_k": 1}, RESULT, ts_ms=T + 1)


def test_annual_sharpe() -> None:
    assert RESULT.annual_sharpe == pytest.approx(0.05 * math.sqrt(365))


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"sharpe": float("nan")}, "non fini"),
        ({"n_obs": 1}, "2 observations"),
        ({"kurtosis": 0.5}, "Pearson"),
        ({"periods_per_year": 0}, "periods_per_year"),
    ],
)
def test_invalid_results(kwargs: dict[str, Any], msg: str) -> None:
    base = {"sharpe": 0.05, "n_obs": 365, "skew": 0.0, "kurtosis": 3.0, "periods_per_year": 365}
    with pytest.raises(DataError, match=msg):
        TrialResult(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("field", "value", "msg"),
    [
        ("n_obs", 3.7, "entier attendu"),
        ("ts_ms", True, "entier attendu"),
        ("sharpe", "0.05", "nombre attendu"),
        ("note", 3, "texte attendu"),
    ],
)
def test_strict_reading(tmp_path: Path, field: str, value: Any, msg: str) -> None:
    reg, h = TrialRegistry(tmp_path / "trials.jsonl"), _hyp(tmp_path)
    reg.record(h, {"lookback_days": 30, "top_k": 1}, RESULT, ts_ms=T)
    obj = json.loads(reg.path.read_text())
    obj[field] = value
    reg.path.write_text(json.dumps(obj) + "\n")
    with pytest.raises(DataError, match=f"essai, ligne 1 : .*{msg}"):
        reg.trials()


def test_cli(config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reg = TrialRegistry(tmp_path / "data" / "meta" / "trials.jsonl")
    reg.record(_hyp(tmp_path), {"lookback_days": 90, "top_k": 3}, RESULT, ts_ms=T)
    assert tr.main(["--config", str(config_dir), "show"]) == 0
    out = capsys.readouterr().out
    assert "Volet longterm : N = 1 essai(s) distinct(s)" in out
    assert "ts_momentum" in out
