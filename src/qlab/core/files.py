"""Écriture de fichiers sûre : un fichier écrit est complet ou absent, jamais à moitié.

``write_atomic`` : fichier temporaire caché dans le même dossier, ``fsync``, renommage atomique
(``os.replace``), puis ``fsync`` du dossier pour que l'entrée survive à une coupure de courant.
Un crash ne laisse au pire qu'un ``.tmp-*`` ignoré par les lecteurs. Refuse d'écraser un
fichier existant, sauf demande explicite.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from qlab.core.errors import DataError


def fsync_dir(directory: Path) -> None:
    """Rend durable la création / le renommage d'une entrée dans ``directory``."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    """Écrit ``data`` dans ``path`` de façon atomique (dossiers parents créés au besoin).

    ``overwrite=False`` (défaut) : ``DataError`` si ``path`` existe déjà (données brutes et
    snapshots ne sont jamais réécrits).
    """
    if not overwrite and path.exists():
        raise DataError(f"fichier déjà présent, pas d'écrasement : {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)
