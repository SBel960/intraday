"""Registres « append-only » en JSONL : un enregistrement par ligne, jamais modifié ni effacé.

Utilisé par les registres dont chaque ligne compte (apports d'argent, essais statistiques) :

- **écriture** (``append_record``) sous verrou exclusif (``flock``) : relecture et contrôle de
  cohérence (``precheck``) puis écriture et ``fsync`` ; un second écrivain attend son tour. À la
  création du fichier, le dossier est aussi ``fsync`` (l'entrée survit à une coupure) ;
- **lecture stricte** (``read_records``) : une ligne tronquée, illisible ou qui n'est pas un
  objet JSON bloque la lecture (``DataError`` avec son numéro). On ne saute jamais une ligne :
  la réparation est manuelle.

Différent de ``core/jsonlog.py`` (journal d'événements tolérant, un fichier par jour).
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from qlab.core.errors import DataError
from qlab.core.files import fsync_dir

Record = dict[str, Any]


def encode_record(record: Mapping[str, Any]) -> bytes:
    """Une ligne JSON canonique (clés triées), stricte : pas de NaN, UTF-8 valide."""
    try:
        text = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        )
        return (text + "\n").encode("utf-8")
    except (ValueError, TypeError) as exc:  # NaN / inf, objet non JSON, demi-caractère UTF-16
        raise DataError(f"enregistrement non sérialisable : {exc}") from exc


def read_records(path: Path) -> list[Record]:
    """Tous les enregistrements de ``path`` (``[]`` s'il n'existe pas), lecture stricte."""
    if not path.exists():
        return []
    records: list[Record] = []
    with path.open("rb") as f:
        for n, raw in enumerate(f, start=1):
            where = f"{path}, ligne {n}"
            if not raw.endswith(b"\n"):
                raise DataError(f"{where} : ligne tronquée (pas de fin de ligne)")
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise DataError(f"{where} : JSON illisible") from exc
            if not isinstance(obj, dict):
                raise DataError(f"{where} : objet JSON attendu")
            records.append(obj)
    return records


def append_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    precheck: Callable[[list[Record]], None] | None = None,
) -> None:
    """Ajoute ``record`` sous verrou ; ``precheck(existants)`` peut refuser (``DataError``)."""
    line = encode_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # libéré par os.close
        existing = read_records(path)
        if precheck is not None:
            precheck(existing)
        if os.write(fd, line) != len(line):
            raise OSError(f"écriture partielle dans {path}")
        os.fsync(fd)
    finally:
        os.close(fd)
    if created:
        fsync_dir(path.parent)
