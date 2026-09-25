"""Tests de qlab.core.jsonlog : lignes exactes attendues, cas limites, cas d'erreur."""

from __future__ import annotations

import itertools
import multiprocessing
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from qlab.core.jsonlog import JsonLog, read_log

T = 1_704_116_832_345  # 2024-01-01T13:47:12.345Z
T_LEAP_END = 1_709_251_199_999  # 2024-02-29T23:59:59.999Z


def clock(*values: int) -> Callable[[], int]:
    """Horloge déterministe qui renvoie ``values`` dans l'ordre."""
    it: Iterator[int] = iter(values)
    return lambda: next(it)


# --- cas nominal -------------------------------------------------------------------------


def test_exact_line_written(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal:
        journal.info("order.sent", {"qty": 1, "price": "42000.10"})
    path = tmp_path / "paper" / "2024-01-01.jsonl"
    expected = (
        '{"component":"paper","data":{"price":"42000.10","qty":1},'
        '"kind":"order.sent","level":"info","ts_ms":1704116832345}\n'
    )
    assert path.read_text(encoding="utf-8") == expected


def test_levels_and_read_back(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "gate", clock_ms=clock(T, T + 1, T + 2)) as journal:
        journal.info("run.start")
        journal.warning("data.gap", {"missing_ms": 60_000})
        journal.error("fetch.failed", {"url": "https://x", "retries": [1, 2], "ok": False})
    read = read_log(tmp_path / "gate" / "2024-01-01.jsonl")
    assert read.corrupt_lines == ()
    assert [r["level"] for r in read.records] == ["info", "warning", "error"]
    assert [r["ts_ms"] for r in read.records] == [T, T + 1, T + 2]
    assert read.records[0]["data"] == {}
    assert read.records[2]["data"] == {"url": "https://x", "retries": [1, 2], "ok": False}


def test_daily_file_rotation_utc(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "paper", clock_ms=clock(T_LEAP_END, T_LEAP_END + 1)) as journal:
        journal.info("a")
        journal.info("b")
    assert sorted(p.name for p in (tmp_path / "paper").iterdir()) == [
        "2024-02-29.jsonl",
        "2024-03-01.jsonl",
    ]
    assert read_log(tmp_path / "paper" / "2024-03-01.jsonl").records[0]["kind"] == "b"


def test_append_only_across_instances(tmp_path: Path) -> None:
    for kind in ("first", "second"):
        with JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal:
            journal.info(kind)
    read = read_log(tmp_path / "paper" / "2024-01-01.jsonl")
    assert [r["kind"] for r in read.records] == ["first", "second"]


def test_deterministic_bytes(tmp_path: Path) -> None:
    for sub, data in (("a", {"x": 1, "y": 2}), ("b", {"y": 2, "x": 1})):
        with JsonLog(tmp_path / sub, "paper", clock_ms=clock(T)) as journal:
            journal.info("k", data)
    a = (tmp_path / "a" / "paper" / "2024-01-01.jsonl").read_bytes()
    b = (tmp_path / "b" / "paper" / "2024-01-01.jsonl").read_bytes()
    assert a == b


def test_fsync_mode(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "paper", clock_ms=clock(T), fsync=True) as journal:
        journal.info("decision")
    assert len(read_log(journal.path_for(T)).records) == 1


def test_default_clock_is_real_utc(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "paper") as journal:
        journal.info("now")
    files = list((tmp_path / "paper").iterdir())
    assert len(files) == 1
    ts = read_log(files[0]).records[0]["ts_ms"]
    assert isinstance(ts, int) and ts > 1_700_000_000_000


# --- cas limites -------------------------------------------------------------------------


def test_unicode_preserved(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal:
        journal.info("note", {"texte": "écart d'horloge ≥ 1 s"})
    raw = (tmp_path / "paper" / "2024-01-01.jsonl").read_text(encoding="utf-8")
    assert "écart d'horloge ≥ 1 s" in raw
    assert read_log(tmp_path / "paper" / "2024-01-01.jsonl").records[0]["data"] == {
        "texte": "écart d'horloge ≥ 1 s"
    }


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    read = read_log(path)
    assert read.records == () and read.corrupt_lines == ()


def test_no_file_created_without_event(tmp_path: Path) -> None:
    JsonLog(tmp_path, "paper").close()
    assert not (tmp_path / "paper").exists()


def test_truncated_line_after_crash_is_terminated_and_reported(tmp_path: Path) -> None:
    path = tmp_path / "paper" / "2024-01-01.jsonl"
    with JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal:
        journal.info("before")
    with path.open("ab") as f:
        f.write(b'{"ts_ms":17041')  # crash au milieu d'une écriture
    with JsonLog(tmp_path, "paper", clock_ms=clock(T + 5)) as journal:
        journal.info("after")
    read = read_log(path)
    assert [r["kind"] for r in read.records] == ["before", "after"]
    assert read.corrupt_lines == (2,)
    assert path.read_bytes().count(b"\n") == 3


@pytest.mark.parametrize(
    "line",
    [
        b"",  # ligne vide
        b"not json",
        b'{"ts_ms":1,"level":"info","component":"c","kind":"k"}',  # data manquant
        b'{"ts_ms":1,"level":"debug","component":"c","kind":"k","data":{}}',  # niveau inconnu
        b'{"ts_ms":true,"level":"info","component":"c","kind":"k","data":{}}',  # bool
        b'{"ts_ms":1.5,"level":"info","component":"c","kind":"k","data":{}}',  # float
        b'{"ts_ms":1,"level":"info","component":"c","kind":"k","data":[]}',
        b'{"ts_ms":1,"level":"info","component":"c","kind":"k","data":{},"x":1}',  # en trop
        b'{"ts_ms":1,"level":"info","component":"c","kind":"k","data":{"v":NaN}}',
        b"\xff\xfe",  # UTF-8 invalide
    ],
)
def test_reader_reports_bad_lines(tmp_path: Path, line: bytes) -> None:
    good = b'{"component":"c","data":{},"kind":"k","level":"info","ts_ms":1}\n'
    path = tmp_path / "log.jsonl"
    path.write_bytes(good + line + b"\n" + good)
    read = read_log(path)
    assert read.corrupt_lines == (2,)
    assert len(read.records) == 2


# --- cas d'erreur ------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_rejected(tmp_path: Path, value: float) -> None:
    with (
        JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal,
        pytest.raises(ValueError, match="non finie"),
    ):
        journal.info("k", {"v": value})


def test_non_json_object_rejected(tmp_path: Path) -> None:
    with (
        JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal,
        pytest.raises(TypeError, match="not JSON serializable"),
    ):
        journal.info("k", {"v": object()})  # type: ignore[dict-item]


def test_non_str_key_rejected(tmp_path: Path) -> None:
    with (
        JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal,
        pytest.raises(TypeError, match="clés de data"),
    ):
        journal.info("k", {1: "x"})  # type: ignore[dict-item]


@pytest.mark.parametrize("component", ["", "../evil", "Paper", "a/b", "a b"])
def test_bad_component(tmp_path: Path, component: str) -> None:
    with pytest.raises(ValueError, match="component invalide"):
        JsonLog(tmp_path, component)


@pytest.mark.parametrize("kind", ["", "Order", "order.", ".x", "a..b", "a-b"])
def test_bad_kind(tmp_path: Path, kind: str) -> None:
    with (
        JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal,
        pytest.raises(ValueError, match="kind invalide"),
    ):
        journal.info(kind)


def test_bad_level(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal, pytest.raises(ValueError):
        journal.log("debug", "k")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_ts", [1.5, True, "1704116832345"])
def test_bad_clock(tmp_path: Path, bad_ts: object) -> None:
    with (
        JsonLog(tmp_path, "paper", clock_ms=lambda: bad_ts) as journal,  # type: ignore[arg-type,return-value]
        pytest.raises(TypeError, match="horloge"),
    ):
        journal.info("k")


def test_write_after_close(tmp_path: Path) -> None:
    journal = JsonLog(tmp_path, "paper", clock_ms=clock(*itertools.repeat(T, 2)))
    journal.close()
    with pytest.raises(RuntimeError, match="fermé"):
        journal.info("k")


def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_log(tmp_path / "absent.jsonl")


# --- non-régression du break test (2026-09-25) --------------------------------------------


def _concurrent_writer(log_dir: str, worker: int) -> None:
    with JsonLog(Path(log_dir), "c", clock_ms=lambda: T) as journal:
        for i in range(200):
            journal.info("k", {"w": worker, "i": i, "pad": "x" * 3000})


def test_concurrent_processes_no_corruption(tmp_path: Path) -> None:
    """4 processus × 200 lignes de 3 Ko dans le même fichier : aucune ligne parasite."""
    procs = [
        multiprocessing.get_context("spawn").Process(
            target=_concurrent_writer, args=(str(tmp_path), w)
        )
        for w in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    read = read_log(tmp_path / "c" / "2024-01-01.jsonl")
    assert read.corrupt_lines == ()
    assert len(read.records) == 800


def test_lone_surrogate_rejected_clearly(tmp_path: Path) -> None:
    with (
        JsonLog(tmp_path, "paper", clock_ms=clock(T)) as journal,
        pytest.raises(ValueError, match="non encodable en UTF-8"),
    ):
        journal.info("k", {"v": "\ud800"})
    assert not (tmp_path / "paper" / "2024-01-01.jsonl").exists()  # rien d'écrit
