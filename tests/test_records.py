"""Tests de qlab.core.records : écriture canonique, lecture stricte, verrou, refus."""

from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

import pytest

from qlab.core.errors import DataError
from qlab.core.records import append_record, encode_record, read_records


def test_canonical_encoding() -> None:
    assert encode_record({"b": 1, "a": "é"}) == '{"a":"é","b":1}\n'.encode()


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"x": object()}, {"x": "\ud800"}])
def test_unserializable_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(DataError, match="non sérialisable"):
        encode_record(bad)


def test_append_and_read(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "r.jsonl"
    assert read_records(path) == []
    append_record(path, {"n": 1})
    append_record(path, {"n": 2})
    assert read_records(path) == [{"n": 1}, {"n": 2}]


def test_precheck_can_refuse_and_sees_existing(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    append_record(path, {"n": 1})
    seen: list[object] = []

    def refuse(existing: list[dict[str, object]]) -> None:
        seen.append(existing)
        raise DataError("refusé")

    with pytest.raises(DataError, match="refusé"):
        append_record(path, {"n": 2}, precheck=refuse)
    assert seen == [[{"n": 1}]] and read_records(path) == [{"n": 1}]  # rien d'écrit


@pytest.mark.parametrize(
    ("content", "msg"),
    [
        (b'{"n":1}', "ligne 1 : ligne tronquée"),
        (b"pas du json\n", "JSON illisible"),
        (b"[1, 2]\n", "objet JSON attendu"),
        (b'{"n":1}\n\xff\n', "ligne 2 : JSON illisible"),
    ],
)
def test_strict_reading(tmp_path: Path, content: bytes, msg: str) -> None:
    path = tmp_path / "r.jsonl"
    path.write_bytes(content)
    with pytest.raises(DataError, match=msg):
        read_records(path)


def test_writer_waits_for_lock(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    append_record(path, {"n": 1})
    holder = os.open(path, os.O_RDONLY)
    fcntl.flock(holder, fcntl.LOCK_EX)
    worker = threading.Thread(target=append_record, args=(path, {"n": 2}))
    worker.start()
    time.sleep(0.3)
    try:
        assert worker.is_alive() and len(read_records(path)) == 1
    finally:
        os.close(holder)
    worker.join(timeout=5)
    assert read_records(path) == [{"n": 1}, {"n": 2}]
