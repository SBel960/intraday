"""Tests de qlab.longterm.klines_build : écriture, reprise, paires non latines, erreurs, CLI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fakes import kline_archive, kline_csv, kline_row

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms
from qlab.longterm import klines as kl
from qlab.longterm import klines_build as kb

D = MS_PER_DAY
T0 = date_to_ms("2024-12-30")


def test_build_symbol_writes_skips_and_rebuilds(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    kline_archive(
        paths, "monthly", "BTCEUR-1d-2024-12", kline_csv(kline_row(T0), kline_row(T0 + D))
    )
    daily = kline_archive(
        paths,
        "daily",
        "BTCEUR-1d-2025-01-02",
        kline_csv(kline_row(T0 + 3 * D, us=True), kline_row(T0, high=1.0, us=True)),
    )
    res = kb.build_symbol(paths, "BTCEUR", "1d")
    assert (res.status, res.bars, res.gaps, res.missing, res.anomalies) == ("construit", 3, 1, 1, 1)
    assert kl.load(paths, "1d", "BTCEUR")["open_time_ms"].to_list() == [T0, T0 + D, T0 + 3 * D]
    assert kb.build_symbol(paths, "BTCEUR", "1d").status == "à jour"
    assert kb.build_symbol(paths, "BTCEUR", "1d", force=True).status == "construit"
    later = paths.lt_klines("1d", "BTCEUR").stat().st_mtime + 10
    os.utime(daily, (later, later))
    assert kb.build_symbol(paths, "BTCEUR", "1d").status == "construit"
    assert kb.build_symbol(paths, "ETHEUR", "1d").status == "vide"


def test_symbols_on_disk_include_non_latin_names(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    assert kb.symbols_on_disk(paths) == []
    kline_archive(paths, "monthly", "x", kline_csv(kline_row(T0)))
    kline_archive(paths, "daily", "y", kline_csv(kline_row(T0)), symbol="币安人生USDT")
    assert kb.symbols_on_disk(paths) == ["BTCEUR", "币安人生USDT"]


def test_cli_builds_all_pairs(
    config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = DataPaths(tmp_path / "data")
    kline_archive(
        paths, "monthly", "BTCEUR-1d-2024-12", kline_csv(kline_row(T0), kline_row(T0 + 2 * D))
    )
    kline_archive(paths, "monthly", "ETHEUR-1d-2024-12", kline_csv(kline_row(T0)), symbol="ETHEUR")
    assert kb.main(["--config", str(config_dir), "build", "--intervals", "1d"]) == 0
    out = capsys.readouterr().out
    assert "1d : paires 2, construites 2, a_jour 0, vides 0, erreurs 0, bougies 3, trous 1" in out
    assert paths.lt_klines("1d", "ETHEUR").exists()


def test_broken_archive_stops_its_pair_only(
    config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = DataPaths(tmp_path / "data")
    kline_archive(paths, "monthly", "BTCEUR-1d-2024-12", kline_csv(kline_row(T0)))
    bad = kline_row(T0).replace(",7,", ",sept,")
    kline_archive(paths, "monthly", "ETHEUR-1d-2024-12", kline_csv(bad), symbol="ETHEUR")
    assert kb.main(["--config", str(config_dir), "build", "--intervals", "1d"]) == 1
    out = capsys.readouterr().out
    assert "construites 1" in out and "erreurs 1" in out
    assert "ERREUR ETHEUR 1d : ETHEUR-1d-2024-12.zip : CSV illisible" in out
    assert paths.lt_klines("1d", "BTCEUR").exists()
    with pytest.raises(DataError, match="non géré"):
        kb.build_symbol(paths, "BTCEUR", "4h")
