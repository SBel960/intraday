"""Tests de qlab.core.paths : l'arbre des données correspond à docs/ARBORESCENCE.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from qlab.core.paths import DataPaths


def test_tree_matches_documentation() -> None:
    p = DataPaths(Path("/data"))
    assert p.meta == Path("/data/meta")
    assert p.ledger == Path("/data/meta/ledger.jsonl")
    assert p.exchange_info_dir("binance") == Path("/data/meta/exchange_info/binance")
    assert p.logs == Path("/data/logs")
    assert p.reports == Path("/data/reports")


def test_documented_in_arborescence() -> None:
    """Chaque dossier de DataPaths figure dans l'arbre « Données » de ARBORESCENCE.md."""
    doc = (Path(__file__).resolve().parent.parent / "docs" / "ARBORESCENCE.md").read_text()
    for fragment in ("ledger.jsonl", "exchange_info/{source}/", "logs/{component}/", "reports/"):
        assert fragment in doc


def test_no_directory_created(tmp_path: Path) -> None:
    DataPaths(tmp_path / "d").exchange_info_dir("binance")
    assert not (tmp_path / "d").exists()


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "", "Binance", "bin ance"])
def test_unsafe_names_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="invalide pour un chemin"):
        DataPaths(Path("/data")).exchange_info_dir(bad)
