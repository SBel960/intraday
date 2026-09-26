"""Tests de qlab.data.archives : sélection, vérification, reprise, échecs — sans réseau."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest
from fakes import FakeVision

from qlab.core.errors import DataError, ExchangeError
from qlab.core.jsonlog import read_log
from qlab.core.paths import DataPaths
from qlab.data import archives
from qlab.data.archives import SyncReport, archive_prefixes, download_one, list_all, sync
from qlab.data.binance_vision import ArchiveFile

M = "data/spot/monthly/klines"
CONTENT = {
    f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip": b"btc-1d-janvier",
    f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-02.zip": b"btc-1d-fevrier",
    f"{M}/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip": b"btc-1h-janvier" * 10,
    f"{M}/OLDUSDT/1d/OLDUSDT-1d-2019-05.zip": b"paire-retiree",
    "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-03-01.zip": b"btc-1d-1er-mars",
    "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-02-29.zip": b"deja-dans-le-mensuel",
    "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip": b"funding",
    "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip": b"metrics-btc",
    "data/futures/um/daily/metrics/ETHUSDT/ETHUSDT-metrics-2024-01-01.zip": b"metrics-eth",
}


def _remote(vision: FakeVision, **kwargs: object) -> list[ArchiveFile]:
    options: dict[str, object] = {
        "datasets": ("klines", "funding", "metrics"),
        "intervals": ("1d", "1h"),
        "symbols": None,
        "metrics_symbols": ("BTCUSDT",),
        "today": "2024-03-02",
        "trading": None,
        "active": None,
    }
    options.update(kwargs)
    prefixes = archive_prefixes(vision, vision.list_url, **options)  # type: ignore[arg-type]
    return list_all(vision, vision.list_url, prefixes, workers=3)


def _sync(
    vision: FakeVision, paths: DataPaths, files: list[ArchiveFile], *, dry_run: bool = False
) -> SyncReport:
    return sync(
        files,
        base_url=vision.base_url,
        paths=paths,
        fetch=vision,
        workers=3,
        dry_run=dry_run,
        progress=lambda _: None,
    )


# --- sélection ---------------------------------------------------------------------------


def test_selection_all_datasets() -> None:
    """Toutes les archives attendues, triées par clé (listages en parallèle ⇒ tri explicite)."""
    keys = [f.key for f in _remote(FakeVision(CONTENT))]
    assert keys == sorted(keys)
    assert keys == [
        "data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip",  # pas ETHUSDT
        "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip",
        "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-03-01.zip",  # mois en cours seulement
        f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip",
        f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-02.zip",
        f"{M}/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip",
        f"{M}/OLDUSDT/1d/OLDUSDT-1d-2019-05.zip",  # paire retirée gardée
    ]


def test_daily_listed_only_for_active_pairs() -> None:
    """Paire retirée (absente du snapshot) : pas de listage quotidien du mois en cours."""
    prefixes = archive_prefixes(
        FakeVision(CONTENT),
        "https://s3.example/bucket",
        datasets=("klines",),
        intervals=("1d",),
        symbols=None,
        metrics_symbols=(),
        today="2024-03-02",
        trading=None,
        active={"BTCUSDT"},
    )
    assert prefixes == [
        f"{M}/BTCUSDT/1d/",
        "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-03",
        f"{M}/OLDUSDT/1d/",  # mensuel toujours listé (historique)
    ]


def test_selection_filters() -> None:
    vision = FakeVision(CONTENT)
    only_old = _remote(vision, datasets=("klines",), intervals=("1d",), symbols=["OLDUSDT"])
    assert [f.size for f in only_old] == [len(b"paire-retiree")]
    trading_only = _remote(vision, datasets=("klines",), intervals=("1d",), trading={"BTCUSDT"})
    assert not any("OLDUSDT" in f.key for f in trading_only)  # include_delisted faux


# --- téléchargement vérifié --------------------------------------------------------------


def test_download_verified_and_placed(tmp_path: Path) -> None:
    vision, paths = FakeVision(CONTENT), DataPaths(tmp_path)
    key = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    assert download_one(ArchiveFile(key, 14), vision.base_url, paths, vision) == 14
    local = tmp_path / "raw" / "binance_vision" / key.removeprefix("data/")
    assert local.read_bytes() == b"btc-1d-janvier"


def test_bad_checksum_writes_nothing(tmp_path: Path) -> None:
    key = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    vision, paths = FakeVision(CONTENT, bad_checksum={key}), DataPaths(tmp_path)
    with pytest.raises(DataError, match="SHA-256 incorrect"):
        download_one(ArchiveFile(key, 14), vision.base_url, paths, vision)
    assert not (tmp_path / "raw").exists()


def test_size_mismatch_writes_nothing(tmp_path: Path) -> None:
    vision, paths = FakeVision(CONTENT), DataPaths(tmp_path)
    with pytest.raises(DataError, match="taille"):
        download_one(
            ArchiveFile(f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip", 999),
            vision.base_url,
            paths,
            vision,
        )
    assert not (tmp_path / "raw").exists()


# --- synchronisation ---------------------------------------------------------------------


def test_sync_then_resume_is_idempotent(tmp_path: Path) -> None:
    vision, paths = FakeVision(CONTENT), DataPaths(tmp_path)
    files = _remote(vision)
    first = _sync(vision, paths, files)
    assert (first.listed, first.present, first.downloaded) == (7, 0, 7)
    assert first.downloaded_bytes == sum(f.size for f in files)
    downloads = len(vision.downloads)
    second = _sync(vision, paths, files)
    assert (second.present, second.downloaded) == (7, 0)
    assert len(vision.downloads) == downloads  # rien retéléchargé


def test_dry_run_downloads_nothing(tmp_path: Path) -> None:
    vision, paths = FakeVision(CONTENT), DataPaths(tmp_path)
    report = _sync(vision, paths, _remote(vision), dry_run=True)
    assert report.downloaded == 0 and vision.downloads == []


def test_one_failure_does_not_stop_others(tmp_path: Path) -> None:
    bad = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-02.zip"
    vision = FakeVision(CONTENT, bad_checksum={bad})
    report = _sync(vision, DataPaths(tmp_path), _remote(vision))
    assert report.downloaded == 6
    assert [k for k, _ in report.failed] == [bad]
    retry = _sync(FakeVision(CONTENT), DataPaths(tmp_path), _remote(vision))
    assert retry.downloaded == 1  # repris au passage suivant


def test_republished_archive_is_never_overwritten(tmp_path: Path) -> None:
    vision, paths = FakeVision(CONTENT), DataPaths(tmp_path)
    key = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    _sync(vision, paths, _remote(vision))
    changed = FakeVision({**CONTENT, key: b"version-corrigee-plus-longue"})
    report = _sync(changed, paths, _remote(changed))
    assert report.republished == [key]
    local = paths.raw_archive("binance_vision", key.removeprefix("data/"))
    assert local.read_bytes() == b"btc-1d-janvier"  # RAW jamais modifié


def test_exchange_error_is_collected(tmp_path: Path) -> None:
    key = f"{M}/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    vision = FakeVision(CONTENT, failing={key: ExchangeError("HTTP 404")})
    report = _sync(vision, DataPaths(tmp_path), _remote(vision))
    assert report.failed == [(key, "HTTP 404")]


# --- commande ----------------------------------------------------------------------------


def _patch(monkeypatch: pytest.MonkeyPatch, vision: FakeVision) -> None:
    class _Result:
        def __init__(self, body: bytes) -> None:
            self.body = body

    monkeypatch.setattr("qlab.core.http.get", lambda url, **kw: _Result(vision(url)))
    monkeypatch.setattr(archives, "now_ms", lambda: 1_709_337_600_000)  # 2024-03-02


def test_cli(
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vision = FakeVision(
        CONTENT,
        list_url="https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
        base_url="https://data.binance.vision",
    )
    _patch(monkeypatch, vision)
    cfg = str(config_dir)
    assert archives.main(["--config", cfg, "sync", "--dry-run"]) == 0
    assert "à télécharger : 7 (simulation)" in capsys.readouterr().out
    assert (
        archives.main(["--config", cfg, "sync", "--dataset", "klines", "--symbols", "BTCUSDT"]) == 0
    )
    assert "téléchargées : 4" in capsys.readouterr().out
    assert archives.main(["--config", cfg, "sync"]) == 0
    out = capsys.readouterr().out
    assert "Déjà présentes : 4 ; à télécharger : 3 ; téléchargées : 3" in out
    logs = sorted((tmp_path / "data" / "logs" / "archives").glob("*.jsonl"))
    kinds = [r["kind"] for f in logs for r in read_log(f).records]
    assert kinds.count("archives.sync") == 3


def test_cli_failure_exit_code(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    vision = FakeVision(
        CONTENT,
        list_url="https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
        base_url="https://data.binance.vision",
        failing={key: ExchangeError("HTTP 403")},
    )
    _patch(monkeypatch, vision)
    assert archives.main(["--config", str(config_dir), "sync"]) == 1
    assert f"ÉCHEC : {key} : HTTP 403" in capsys.readouterr().err


def test_network_errors_propagate_from_http(tmp_path: Path) -> None:
    """Une coupure réseau est gérée par core/http (attente) ; ici la doublure la lève tout de
    suite, et l'archive est simplement retentée au passage suivant."""
    key = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    vision = FakeVision(CONTENT, failing={key: urllib.error.URLError("coupé")})
    with pytest.raises(urllib.error.URLError):
        _sync(vision, DataPaths(tmp_path), [ArchiveFile(key, 14)])


# --- non-régression de l'audit (2026-09-26) -----------------------------------------------


def test_malicious_key_is_skipped_not_fatal(tmp_path: Path) -> None:
    """Clé d'archive qui sortirait du dossier : écartée et signalée, les autres continuent."""
    evil = ArchiveFile("data/../../../tmp/pwned.zip", 5)
    good_key = f"{M}/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    vision = FakeVision(CONTENT)
    report = _sync(vision, DataPaths(tmp_path), [evil, ArchiveFile(good_key, 14)])
    assert [k for k, _ in report.failed] == [evil.key]
    assert report.downloaded == 1
    assert not (tmp_path.parent / "tmp" / "pwned.zip").exists()


def test_non_ascii_symbol_downloads_end_to_end(tmp_path: Path) -> None:
    """Paire au nom chinois : listée, téléchargée, vérifiée, rangée sous son vrai nom."""
    key = f"{M}/币安人生USDT/1d/币安人生USDT-1d-2026-01.zip"
    vision = FakeVision({key: b"archive-chinoise"})
    report = _sync(
        vision, DataPaths(tmp_path), _remote(vision, datasets=("klines",), intervals=("1d",))
    )
    assert (report.downloaded, report.failed) == (1, [])
    local = DataPaths(tmp_path).raw_archive("binance_vision", key.removeprefix("data/"))
    assert local.read_bytes() == b"archive-chinoise"
