"""Tests de qlab.core.downloads : présence, changement, échecs isolés, parallélisme, bugs."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from qlab.core.downloads import Job, download_all
from qlab.core.errors import DataError, ExchangeError


def _job(tmp_path: Path, name: str, data: bytes | Exception, size: int | None = None) -> Job:
    def fetch() -> bytes:
        if isinstance(data, Exception):
            raise data
        return data

    return Job(name, tmp_path / "raw" / name, fetch, size)


def test_downloads_and_resumes(tmp_path: Path) -> None:
    jobs = [_job(tmp_path, f"f{i}", bytes([i]) * (i + 1), i + 1) for i in range(5)]
    first = download_all(jobs, workers=3, progress=lambda _: None)
    assert (first.listed, first.present, first.downloaded, first.downloaded_bytes) == (5, 0, 5, 15)
    assert (tmp_path / "raw" / "f4").read_bytes() == b"\x04" * 5
    second = download_all(jobs, workers=3, progress=lambda _: None)
    assert (second.present, second.downloaded, second.missing) == (5, 0, 0)


def test_unknown_size_counts_as_present(tmp_path: Path) -> None:
    job = _job(tmp_path, "f", b"abc")  # taille inconnue (ex. Tardis)
    download_all([job], workers=1, progress=lambda _: None)
    assert download_all([job], workers=1, progress=lambda _: None).present == 1


def test_changed_file_is_never_rewritten(tmp_path: Path) -> None:
    download_all([_job(tmp_path, "f", b"v1", 2)], workers=1, progress=lambda _: None)
    report = download_all(
        [_job(tmp_path, "f", b"version-2", 9)], workers=1, progress=lambda _: None
    )
    assert report.changed == ["f"] and report.downloaded == 0
    assert (tmp_path / "raw" / "f").read_bytes() == b"v1"


def test_failures_are_isolated(tmp_path: Path) -> None:
    jobs = [
        _job(tmp_path, "ok", b"x", 1),
        _job(tmp_path, "sha", DataError("SHA-256 incorrect")),
        _job(tmp_path, "http", ExchangeError("HTTP 404")),
        _job(tmp_path, "size", b"xx", 1),
    ]
    report = download_all(jobs, workers=2, progress=lambda _: None)
    assert report.downloaded == 1
    assert sorted(report.failed) == [
        ("http", "HTTP 404"),
        ("sha", "SHA-256 incorrect"),
        ("size", "taille 2 ≠ 1 annoncée pour size"),
    ]
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["ok"]


def test_bug_propagates(tmp_path: Path) -> None:
    """Une exception inattendue est un bug : elle remonte (code 2), elle n'est pas avalée."""
    with pytest.raises(ZeroDivisionError):
        download_all(
            [_job(tmp_path, "f", ZeroDivisionError("bug"))], workers=1, progress=lambda _: None
        )


def test_dry_run_and_empty(tmp_path: Path) -> None:
    report = download_all(
        [_job(tmp_path, "f", b"x")], workers=1, dry_run=True, progress=lambda _: None
    )
    assert (report.listed, report.downloaded, report.missing) == (1, 0, 1)
    assert download_all([], workers=1, progress=lambda _: None).listed == 0


def test_parallelism_is_real(tmp_path: Path) -> None:
    """3 téléchargements simultanés : les 3 attendent ensemble (barrière) sans se bloquer."""
    barrier = threading.Barrier(3, timeout=5)

    def fetch() -> bytes:
        barrier.wait()
        return b"x"

    jobs = [Job(f"f{i}", tmp_path / f"f{i}", fetch, 1) for i in range(3)]
    assert download_all(jobs, workers=3, progress=lambda _: None).downloaded == 3
