"""Tests de qlab.research.hypothesis : fiche valide, essais déclarés, empreinte, refus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from qlab.research import hypothesis as hy
from qlab.research.hypothesis import HypothesisError, load_all, load_hypothesis

REPO = Path(__file__).resolve().parent.parent
FICHE: dict[str, Any] = {
    "id": "ts_momentum",
    "title": "Momentum temporel",
    "volet": "longterm",
    "created": "2026-09-26",
    "mechanism": "Les investisseurs sous-réagissent aux tendances : elles persistent un temps.",
    "horizon_s": 2_592_000,
    "universe": "trade",
    "features": ["close_1d"],
    "parameters": [
        {"name": "lookback_days", "values": [30, 90, 180]},
        {"name": "rebalance_days", "values": [7, 30]},
    ],
    "cost_round_trip_frac": 0.002,
    "min_edge_frac": 0.05,
    "edge_basis": "per_year",
    "abandon": "DSR < 0,95 ou edge net négatif après coûts sur la période de test.",
}


def _write(tmp_path: Path, fiche: dict[str, Any], name: str | None = None) -> Path:
    path = tmp_path / f"{name or fiche['id']}.yaml"
    path.write_text(yaml.safe_dump(fiche, allow_unicode=True), encoding="utf-8")
    return path


def test_valid_fiche_trials_and_grid(tmp_path: Path) -> None:
    h = load_hypothesis(_write(tmp_path, FICHE))
    assert h.n_trials == 6  # 3 × 2 combinaisons déclarées d'avance
    grid = list(h.grid())
    assert grid[0] == {"lookback_days": 30.0, "rebalance_days": 7.0}
    assert grid[-1] == {"lookback_days": 180.0, "rebalance_days": 30.0}
    assert len(grid) == 6


def test_no_parameters_is_one_trial(tmp_path: Path) -> None:
    h = load_hypothesis(_write(tmp_path, {**FICHE, "parameters": []}))
    assert h.n_trials == 1 and list(h.grid()) == [{}]


def test_fingerprint_ignores_formatting_not_content(tmp_path: Path) -> None:
    a = load_hypothesis(_write(tmp_path, FICHE))
    path = tmp_path / "ts_momentum.yaml"
    path.write_text("# commentaire ajouté\n" + path.read_text(), encoding="utf-8")
    assert load_hypothesis(path).fingerprint == a.fingerprint
    changed = load_hypothesis(_write(tmp_path, {**FICHE, "min_edge_frac": 0.04}))
    assert changed.fingerprint != a.fingerprint  # fiche modifiée après coup : ça se voit


def test_template_and_repo_fiches_are_valid() -> None:
    assert load_hypothesis(REPO / "hypotheses" / "_template.yaml").id == "_template"
    for h in load_all(REPO / "hypotheses"):  # le modèle (_…) est ignoré par load_all
        assert not h.id.startswith("_")


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"volet": "swing"}, "volet"),
        ({"created": "26/09/2026"}, "created"),
        ({"mechanism": "momentum"}, "expliquer le mécanisme"),
        ({"horizon_s": 0}, "horizon_s"),
        ({"universe": "all"}, "universe"),
        ({"features": []}, "features"),
        ({"parameters": [{"name": "a", "values": []}]}, "au moins une valeur"),
        ({"parameters": [{"name": "a", "values": [1, 1]}]}, "en double"),
        (
            {"parameters": [{"name": "a", "values": [1]}, {"name": "a", "values": [2]}]},
            "noms en double",
        ),
        ({"min_edge_frac": 0}, "min_edge_frac"),
        ({"edge_basis": "per_trade", "min_edge_frac": 0.001}, "doit dépasser le coût"),
        ({"id": "Mauvais-Id"}, "id invalide"),
    ],
)
def test_invalid_fiches(tmp_path: Path, patch: dict[str, Any], msg: str) -> None:
    with pytest.raises(HypothesisError, match=msg):
        load_hypothesis(_write(tmp_path, {**FICHE, **patch}, name="ts_momentum"))


def test_id_must_match_filename(tmp_path: Path) -> None:
    with pytest.raises(HypothesisError, match="≠ nom du fichier"):
        load_hypothesis(_write(tmp_path, FICHE, name="autre_nom"))


def test_missing_and_unknown_keys(tmp_path: Path) -> None:
    fiche = {k: v for k, v in FICHE.items() if k != "abandon"}
    with pytest.raises(HypothesisError, match=r"manquante\(s\) \['abandon'\]"):
        load_hypothesis(_write(tmp_path, fiche))
    with pytest.raises(HypothesisError, match="inconnue"):
        load_hypothesis(_write(tmp_path, {**FICHE, "optimiser_jusqu_a_ce_que_ca_marche": True}))


def test_cli(config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, FICHE)
    assert hy.main(["--config", str(config_dir), "check", str(path)]) == 0
    assert "ts_momentum" in capsys.readouterr().out
    bad = _write(tmp_path, {**FICHE, "volet": "x"}, name="ts_momentum")
    assert hy.main(["--config", str(config_dir), "check", str(bad)]) == 1
    assert "ERREUR (HypothesisError)" in capsys.readouterr().err
