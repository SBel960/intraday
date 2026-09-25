"""Arbre des données : le **seul** endroit où les chemins sous ``$DATA_ROOT`` sont définis.

Reflète la section « Données » de ``docs/ARBORESCENCE.md``. Un module ne construit jamais un
chemin de données lui-même : il passe par ``DataPaths``, ce qui garde l'arbre réel identique à
l'arbre documenté. Aucune méthode ne crée de dossier (c'est à l'écrivain de le faire).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME_RE = re.compile(r"^[a-z0-9_]+$")  # composant de chemin choisi par le code (source, …)


def _name(kind: str, value: str) -> str:
    """Refuse tout nom qui pourrait sortir de l'arbre (``..``, ``/``, vide, majuscules…)."""
    if not _NAME_RE.match(value):
        raise ValueError(f"{kind} invalide pour un chemin (attendu [a-z0-9_]+) : {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Chemins sous la racine des données (``base.data.root`` de la config)."""

    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta"

    @property
    def ledger(self) -> Path:
        """Registre des apports / retraits (``core/ledger.py``)."""
        return self.meta / "ledger.jsonl"

    def exchange_info_dir(self, source: str) -> Path:
        """Snapshots versionnés d'``exchangeInfo`` (``exchange/snapshots.py``)."""
        return self.meta / "exchange_info" / _name("source", source)

    @property
    def logs(self) -> Path:
        """Racine des journaux JSONL ; ``JsonLog`` y ajoute ``{component}/``."""
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"
