"""Tests de qlab.core.files : écriture atomique, pas d'écrasement, pas de reste après crash."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qlab.core.errors import DataError
from qlab.core.files import fsync_dir, write_atomic


def test_write_creates_parents_and_content(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "f.bin"
    write_atomic(target, b"\x00abc")
    assert target.read_bytes() == b"\x00abc"
    assert [p.name for p in target.parent.iterdir()] == ["f.bin"]  # pas de .tmp-* restant


def test_empty_data(tmp_path: Path) -> None:
    write_atomic(tmp_path / "empty", b"")
    assert (tmp_path / "empty").read_bytes() == b""


def test_no_overwrite_by_default(tmp_path: Path) -> None:
    target = tmp_path / "f"
    write_atomic(target, b"v1")
    with pytest.raises(DataError, match="pas d'écrasement"):
        write_atomic(target, b"v2")
    assert target.read_bytes() == b"v1"
    write_atomic(target, b"v2", overwrite=True)
    assert target.read_bytes() == b"v2"


def test_failure_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Crash pendant le renommage : ni fichier cible, ni fichier temporaire."""

    def boom(src: str, dst: str) -> None:
        raise OSError("disque plein")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disque plein"):
        write_atomic(tmp_path / "f", b"data")
    assert list(tmp_path.iterdir()) == []


def test_fsync_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fsync_dir(tmp_path / "absent")
