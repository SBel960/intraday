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

    @property
    def trials(self) -> Path:
        """Registre des essais statistiques (``research/trials.py``)."""
        return self.meta / "trials.jsonl"

    def exchange_info_dir(self, source: str) -> Path:
        """Snapshots versionnés d'``exchangeInfo`` (``exchange/snapshots.py``)."""
        return self.meta / "exchange_info" / _name("source", source)

    def raw_archive(self, source: str, key: str) -> Path:
        """Archive brute téléchargée telle quelle : ``raw/{source}/{key}`` (clé relative, ex.
        ``spot/monthly/klines/BTCEUR/1d/BTCEUR-1d-2024-01.zip``). Jamais modifiée."""
        parts = key.split("/")
        if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"clé d'archive invalide pour un chemin : {key!r}")
        return self.root / "raw" / _name("source", source) / key

    def lt_klines(self, interval: str, symbol: str) -> Path:
        """Bougies long terme d'une paire : ``lt/klines_{interval}/{symbol}.parquet``. Le symbole
        vient de Binance (peut contenir des caractères non latins) : seul un nom qui sortirait
        du dossier est refusé."""
        if not symbol or symbol in (".", "..") or any(c in symbol for c in "/\\\0"):
            raise ValueError(f"symbole invalide pour un chemin : {symbol!r}")
        return self.root / "lt" / f"klines_{_name('intervalle', interval)}" / f"{symbol}.parquet"

    @property
    def logs(self) -> Path:
        """Racine des journaux JSONL ; ``JsonLog`` y ajoute ``{component}/``."""
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"
