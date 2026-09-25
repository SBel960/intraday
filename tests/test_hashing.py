"""Tests de qlab.core.hashing : vecteurs de référence, idempotence, cas limites, erreurs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from qlab.core import hashing
from qlab.core.errors import DataError
from qlab.core.hashing import hash_bytes, hash_file, hash_frame, hash_parquet, hash_tree

# Vecteurs officiels SHA-256 (FIPS 180-2).
SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {"ts_ms": [3, 1, 2], "px": [42000.1, 41999.9, None], "side": ["b", "s", None]},
        schema={"ts_ms": pl.Int64, "px": pl.Float64, "side": pl.String},
    )


# --- fichiers et arborescences -----------------------------------------------------------


def test_reference_vectors(tmp_path: Path) -> None:
    assert hash_bytes(b"abc") == SHA_ABC
    assert hash_bytes(b"") == SHA_EMPTY
    (tmp_path / "abc").write_bytes(b"abc")
    (tmp_path / "empty").write_bytes(b"")
    assert hash_file(tmp_path / "abc") == SHA_ABC
    assert hash_file(tmp_path / "empty") == SHA_EMPTY


def test_file_read_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = bytes(range(256)) * 50
    (tmp_path / "f").write_bytes(data)
    monkeypatch.setattr(hashing, "CHUNK_BYTES", 7)  # beaucoup de petits blocs
    assert hash_file(tmp_path / "f") == _sha(data)


def test_tree_expected_encoding(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "empty_dir").mkdir()  # ignoré
    (tmp_path / "a.txt").write_bytes(b"abc")
    (tmp_path / "sub" / "b.txt").write_bytes(b"")
    expected = _sha(
        b"a.txt\x1f" + SHA_ABC.encode() + b"\n" + b"sub/b.txt\x1f" + SHA_EMPTY.encode() + b"\n"
    )
    assert hash_tree(tmp_path) == expected


def test_tree_detects_changes(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"1")
    h0 = hash_tree(tmp_path)
    assert hash_tree(tmp_path) == h0  # déterministe
    (tmp_path / "a").write_bytes(b"2")
    h1 = hash_tree(tmp_path)
    (tmp_path / "a").rename(tmp_path / "b")
    assert len({h0, h1, hash_tree(tmp_path)}) == 3  # contenu puis nom changés


def test_tree_empty_dir(tmp_path: Path) -> None:
    assert hash_tree(tmp_path) == SHA_EMPTY


# --- tables ------------------------------------------------------------------------------


def test_frame_expected_encoding() -> None:
    df = pl.DataFrame({"k": [2, 1], "s": ["x", None]}, schema={"k": pl.Int64, "s": pl.String})
    expected = _sha(b'k\x1fInt64\ns\x1fString\nrows=2\n2,"x"\n1,\x00N\n')
    assert hash_frame(df) == expected


def test_frame_empty() -> None:
    empty = pl.DataFrame(schema={"k": pl.Int64})
    assert hash_frame(empty) == _sha(b"k\x1fInt64\nrows=0\n")
    assert hash_frame(pl.DataFrame()) == _sha(b"rows=0\n")


def test_frame_single_row() -> None:
    df = pl.DataFrame({"k": [7]}, schema={"k": pl.Int64})
    assert hash_frame(df) == _sha(b"k\x1fInt64\nrows=1\n7\n")


def test_row_order_matters_unless_order_by() -> None:
    df = _df()
    shuffled = df.reverse()
    assert hash_frame(df) != hash_frame(shuffled)
    assert hash_frame(df, order_by=["ts_ms"]) == hash_frame(shuffled, order_by=["ts_ms"])


def test_slicing_does_not_change_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pl.DataFrame({"i": list(range(1_000)), "s": [str(i) for i in range(1_000)]})
    h = hash_frame(df)
    monkeypatch.setattr(hashing, "SLICE_ROWS", 7)
    assert hash_frame(df) == h


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (pl.DataFrame({"s": [""]}), pl.DataFrame({"s": [None]}, schema={"s": pl.String})),
        (pl.DataFrame({"s": ["\x00N"]}), pl.DataFrame({"s": [None]}, schema={"s": pl.String})),
        (
            pl.DataFrame({"x": [1]}, schema={"x": pl.Int64}),
            pl.DataFrame({"x": [1]}, schema={"x": pl.Int32}),
        ),
        (pl.DataFrame({"x": [1]}), pl.DataFrame({"y": [1]})),
        (pl.DataFrame({"x": [0.0]}), pl.DataFrame({"x": [-0.0]})),
        (pl.DataFrame({"x": [0.1]}), pl.DataFrame({"x": [0.1 + 1e-16]})),
        (pl.DataFrame({"s": ["1"]}), pl.DataFrame({"s": [1]})),
        (pl.DataFrame({"a": ["x,y"]}), pl.DataFrame({"a": ["x"], "b": ["y"]})),
    ],
    ids=["vide≠null", "marqueur≠null", "Int64≠Int32", "nom", "-0.0", "1ulp", "str≠int", "virgule"],
)
def test_distinct_contents_distinct_hashes(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert hash_frame(a) != hash_frame(b)


def test_nan_is_deterministic() -> None:
    df = pl.DataFrame({"x": [float("nan"), 1.0]})
    assert hash_frame(df) == hash_frame(df.clone())


# --- Parquet : idempotence indépendante de l'encodage ------------------------------------


def test_parquet_hash_ignores_file_encoding(tmp_path: Path) -> None:
    df = _df()
    df.write_parquet(tmp_path / "zstd.parquet", compression="zstd")
    df.write_parquet(tmp_path / "snappy.parquet", compression="snappy", row_group_size=1)
    assert (tmp_path / "zstd.parquet").read_bytes() != (tmp_path / "snappy.parquet").read_bytes()
    assert hash_parquet(tmp_path / "zstd.parquet") == hash_frame(df)
    assert hash_parquet(tmp_path / "snappy.parquet") == hash_frame(df)


def test_parquet_directory_split_in_files(tmp_path: Path) -> None:
    """Même contenu réparti en deux fichiers (partition hive) : même hash, avec order_by."""
    df = _df()
    part = tmp_path / "date=2024-01-01"
    part.mkdir()
    df.slice(0, 2).write_parquet(part / "b.parquet")
    df.slice(2).write_parquet(part / "a.parquet")
    (tmp_path / "notes.txt").write_text("ignoré")
    # Sans tri : ordre des fichiers (a puis b) ≠ ordre d'origine.
    assert hash_parquet(tmp_path) != hash_frame(df)
    assert hash_parquet(tmp_path, order_by=["ts_ms"]) == hash_frame(df, order_by=["ts_ms"])
    # Pas de colonne « date » ajoutée depuis le chemin.
    files = [part / "a.parquet", part / "b.parquet"]
    assert hash_parquet(files) == hash_parquet(tmp_path)


# --- cas d'erreur ------------------------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "absent")


def test_tree_not_a_directory(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        hash_tree(tmp_path / "f")
    with pytest.raises(FileNotFoundError):
        hash_tree(tmp_path / "absent")


def test_tree_symlink_rejected(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to("/etc")
    with pytest.raises(DataError, match="lien symbolique"):
        hash_tree(tmp_path)


def test_nested_columns_rejected() -> None:
    with pytest.raises(DataError, match="imbriquées"):
        hash_frame(pl.DataFrame({"l": [[1, 2]], "ok": [1]}))


def test_order_by_errors() -> None:
    with pytest.raises(DataError, match="absentes"):
        hash_frame(_df(), order_by=["nope"])
    with pytest.raises(ValueError, match="ne doit pas être vide"):
        hash_frame(_df(), order_by=[])


def test_parquet_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="aucun fichier"):
        hash_parquet(tmp_path)
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"x": ["1"]}).write_parquet(tmp_path / "b.parquet")
    with pytest.raises(DataError, match="schéma différent"):
        hash_parquet(tmp_path)
