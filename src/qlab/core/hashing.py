"""Hash déterministe (SHA-256, hex) de fichiers, d'arborescences et de tables.

Sert aux tests d'idempotence : ingérer deux fois le même lot doit donner le même hash.

- ``hash_file`` : octets bruts d'un fichier (RAW, archives téléchargées).
- ``hash_tree`` : dossier entier = chemins relatifs + hash de chaque fichier, dans un ordre fixe.
- ``hash_frame`` / ``hash_parquet`` : **contenu logique** d'une table (schéma + valeurs), pas les
  octets Parquet. Deux fichiers Parquet au même contenu mais écrits différemment (métadonnées de
  l'outil, découpage en row groups, répartition en plusieurs fichiers) ont le même hash.

Encodage canonique d'une table : noms et types des colonnes, nombre de lignes, puis les lignes en
CSV (chaînes toujours entre guillemets, null = marqueur non quoté, donc distinct de toute chaîne).
L'ordre des lignes compte, sauf si ``order_by`` est fourni : la table est alors triée d'abord.
Flottants : représentation la plus courte qui se relit à l'identique ; ``-0.0`` ≠ ``0.0``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from qlab.core.errors import DataError

CHUNK_BYTES = 1 << 20  # lecture des fichiers par blocs de 1 Mio
SLICE_ROWS = 100_000  # encodage des tables par tranches, mémoire bornée
_NULL = "\x00N"
_SEP = b"\x1f"  # séparateur d'unités ASCII, absent des noms de colonnes et de chemins usuels


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 des octets de ``path``, lu par blocs. ``FileNotFoundError`` s'il n'existe pas."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> str:
    """Hash d'un dossier : pour chaque fichier (ordre des chemins relatifs POSIX), chemin + hash.

    Les dossiers vides ne comptent pas. ``NotADirectoryError`` / ``FileNotFoundError`` si
    ``root`` n'est pas un dossier existant. Les liens symboliques sont refusés (``DataError``) :
    leur cible pourrait sortir de l'arborescence hachée.
    """
    if not root.is_dir():
        raise (NotADirectoryError if root.exists() else FileNotFoundError)(str(root))
    h = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise DataError(f"lien symbolique refusé dans {root} : {path}")
        if path.is_file():
            rel = path.relative_to(root).as_posix().encode("utf-8")
            h.update(rel + _SEP + hash_file(path).encode("ascii") + b"\n")
    return h.hexdigest()


def _check_columns(df: pl.DataFrame) -> None:
    nested = [name for name, dtype in df.schema.items() if dtype.is_nested()]
    if nested:
        raise DataError(f"colonnes imbriquées non hachables : {nested}")


def hash_frame(df: pl.DataFrame, order_by: Sequence[str] | None = None) -> str:
    """Hash du contenu logique de ``df`` (schéma + valeurs).

    ``order_by`` : colonnes de tri appliquées avant le hash, pour un hash indépendant de l'ordre
    des lignes ; les nulls sont placés en dernier. Sans ``order_by``, l'ordre compte.
    """
    _check_columns(df)
    if order_by is not None:
        if not order_by:
            raise ValueError("order_by ne doit pas être vide (None pour garder l'ordre)")
        missing = [c for c in order_by if c not in df.columns]
        if missing:
            raise DataError(f"colonnes de tri absentes : {missing}")
        df = df.sort(list(order_by), nulls_last=True, maintain_order=True)
    h = hashlib.sha256()
    for name, dtype in df.schema.items():
        h.update(name.encode("utf-8") + _SEP + str(dtype).encode("utf-8") + b"\n")
    h.update(f"rows={df.height}\n".encode("ascii"))
    for part in df.iter_slices(SLICE_ROWS):
        csv = part.write_csv(include_header=False, null_value=_NULL, quote_style="non_numeric")
        h.update(csv.encode("utf-8"))
    return h.hexdigest()


def hash_parquet(paths: Path | Sequence[Path], order_by: Sequence[str] | None = None) -> str:
    """Hash du contenu d'un fichier Parquet, d'une liste de fichiers ou d'un dossier.

    Un dossier est lu récursivement (``*.parquet``, ordre des chemins relatifs) ; les fichiers
    sont concaténés dans cet ordre, sans colonnes de partition déduites des chemins. Schémas
    différents entre fichiers ou aucun fichier : ``DataError``.
    """
    if isinstance(paths, Path) and paths.is_dir():
        files = sorted(paths.rglob("*.parquet"), key=lambda p: p.relative_to(paths).as_posix())
    elif isinstance(paths, Path):
        files = [paths]
    else:
        files = list(paths)
    if not files:
        raise DataError(f"aucun fichier Parquet dans {paths}")
    frames = [pl.read_parquet(f, hive_partitioning=False) for f in files]
    first = frames[0].schema
    for f, frame in zip(files[1:], frames[1:], strict=True):
        if frame.schema != first:
            raise DataError(f"schéma différent dans {f} : {dict(frame.schema)} ≠ {dict(first)}")
    return hash_frame(pl.concat(frames, how="vertical"), order_by=order_by)
