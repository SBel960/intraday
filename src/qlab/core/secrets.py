"""Secrets (clés d'API) : lus depuis un fichier ``.env`` privé, jamais affichés ni journalisés.

Le fichier (chemin : ``base.secrets_file``) est hors dépôt (``.gitignore``) et doit être
lisible par son seul propriétaire (``chmod 600``) : sinon ``DataError``. Format strict :
``CLE=valeur`` par ligne, lignes vides et ``# commentaires`` permises, pas de guillemets.

``Secret`` enveloppe une valeur sensible : ``repr`` / ``str`` affichent ``***`` ; la valeur
n'est accessible que par ``reveal()``, à l'endroit précis où elle est envoyée.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

from qlab.core.errors import DataError

_LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


class Secret:
    """Valeur sensible qui ne s'affiche jamais par accident (journal, trace, message)."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not value:
            raise DataError("secret vide")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__


def check_private(path: Path) -> None:
    """``DataError`` si ``path`` est lisible ou modifiable par le groupe ou les autres."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise DataError(
            f"{path} doit être privé : chmod 600 {path} (droits actuels {stat.filemode(mode)})"
        )


def load_env_file(path: Path) -> dict[str, Secret]:
    """Lit ``path`` (``CLE=valeur``) ; fichier absent : ``FileNotFoundError``."""
    check_private(path)
    values: dict[str, Secret] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m is None:
            raise DataError(f"{path}, ligne {n} : format attendu CLE=valeur (clé en MAJUSCULES)")
        key, value = m.group(1), m.group(2).strip()
        if key in values:
            raise DataError(f"{path}, ligne {n} : clé {key} en double")
        if value and value[0] in "\"'":
            raise DataError(f"{path}, ligne {n} : pas de guillemets autour de la valeur de {key}")
        values[key] = Secret(value)
    return values


def require(values: dict[str, Secret], key: str, path: Path) -> Secret:
    """Valeur obligatoire ; absente : ``DataError`` qui dit quoi ajouter (sans valeur)."""
    if key not in values:
        raise DataError(f"{key} absent de {path} : ajouter une ligne {key}=…")
    return values[key]
