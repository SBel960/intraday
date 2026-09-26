"""Tests de qlab.core.paths : l'arbre des données correspond à docs/ARBORESCENCE.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from qlab.core.paths import DataPaths


def test_tree_matches_documentation() -> None:
    p = DataPaths(Path("/data"))
    assert p.meta == Path("/data/meta")
    assert p.ledger == Path("/data/meta/ledger.jsonl")
    assert p.trials == Path("/data/meta/trials.jsonl")
    assert p.exchange_info_dir("binance") == Path("/data/meta/exchange_info/binance")
    assert p.logs == Path("/data/logs")
    assert p.reports == Path("/data/reports")
    assert p.lt_klines("1d", "BTCEUR") == Path("/data/lt/klines_1d/BTCEUR.parquet")
    assert p.lt_klines("1h", "币安人生USDT").name == "币安人生USDT.parquet"
    assert p.raw_archive("binance_vision", "spot/monthly/klines/BTCEUR/1d/X.zip") == Path(
        "/data/raw/binance_vision/spot/monthly/klines/BTCEUR/1d/X.zip"
    )


def test_documented_in_arborescence() -> None:
    """Chaque dossier de DataPaths figure dans l'arbre « Données » de ARBORESCENCE.md."""
    doc = (Path(__file__).resolve().parent.parent / "docs" / "ARBORESCENCE.md").read_text()
    for fragment in (
        "ledger.jsonl",
        "trials.jsonl",
        "exchange_info/{source}/",
        "logs/{component}/",
        "reports/",
        "raw/{source}/",
        "lt/{klines_1d|klines_1h}/{symbol}.parquet",
    ):
        assert fragment in doc


def test_no_directory_created(tmp_path: Path) -> None:
    DataPaths(tmp_path / "d").exchange_info_dir("binance")
    assert not (tmp_path / "d").exists()


@pytest.mark.parametrize("bad", ["..", "../x", "a/b", "", "Binance", "bin ance"])
def test_unsafe_names_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="invalide pour un chemin"):
        DataPaths(Path("/data")).exchange_info_dir(bad)


@pytest.mark.parametrize("key", ["", "/etc/passwd", "a/../../x", "a//b", "./a", "a/.."])
def test_unsafe_archive_keys_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="clé d'archive invalide"):
        DataPaths(Path("/data")).raw_archive("binance_vision", key)


@pytest.mark.parametrize("symbol", ["", ".", "..", "A/B", "A\\B", "A\0B"])
def test_unsafe_symbols_rejected(symbol: str) -> None:
    with pytest.raises(ValueError, match="symbole invalide"):
        DataPaths(Path("/data")).lt_klines("1d", symbol)
