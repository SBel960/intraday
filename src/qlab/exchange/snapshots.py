"""Stockage versionné des snapshots ``exchangeInfo`` et différences entre versions.

Fichiers : ``DataPaths.exchange_info_dir(source)/{YYYYMMDDTHHMMSSZ}.json.zst`` (la réponse brute
fait ~17 Mo, ~110 Ko compressée). Un snapshot contient ::

    {"source", "fetched_ms", "url", "content_hash", "fees": {...}, "exchange_info": {...}}

- **Versionnement** : un fichier n'est écrit que si le contenu change (SHA-256 de la réponse
  sans ``serverTime``, qui change à chaque appel, et des frais). Écriture atomique (fichier
  temporaire, ``fsync``, ``os.replace``), jamais d'écrasement. Le hash est revérifié à la
  lecture : un fichier modifié après coup est détecté.
- **Point-in-time** : ``snapshot_at(ts_ms)`` rend la version en vigueur à ``ts_ms``. Avant la
  première, on applique la première et on le signale (``exchangeInfo`` n'a pas d'historique).
- **Différences** : ``diff_snapshots`` liste paires ajoutées ou disparues, changements de statut
  (retrait = ``TRADING`` → ``BREAK``, dont la date sert à l'univers point-in-time) et de filtres.

Ce module ne fait aucun accès réseau (voir ``exchange_info.py``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

from qlab.core.errors import DataError, ExchangeError
from qlab.core.files import write_atomic
from qlab.core.paths import DataPaths
from qlab.exchange.lot import SymbolFilters

MAX_SNAPSHOT_BYTES = 1 << 30  # 1 Gio décompressé : borne contre un fichier piégé
_FILE_RE = re.compile(r"^\d{8}T\d{6}Z\.json\.zst$")
_REQUIRED_SYMBOL_KEYS = ("symbol", "status", "baseAsset", "quoteAsset", "filters")


def validate_exchange_info(info: object) -> None:
    """Vérifie la structure minimale utilisée par le projet ; sinon ``ExchangeError``."""
    if not isinstance(info, dict) or not isinstance(info.get("symbols"), list):
        raise ExchangeError("exchangeInfo : liste « symbols » absente")
    if not info["symbols"]:
        raise ExchangeError("exchangeInfo : aucune paire")
    seen: set[str] = set()
    for i, s in enumerate(info["symbols"]):
        if not isinstance(s, dict) or any(k not in s for k in _REQUIRED_SYMBOL_KEYS):
            raise ExchangeError(f"exchangeInfo : paire n° {i} incomplète ({_REQUIRED_SYMBOL_KEYS})")
        if not isinstance(s["filters"], list) or s["symbol"] in seen:
            raise ExchangeError(f"exchangeInfo : paire {s['symbol']!r} invalide ou en double")
        seen.add(s["symbol"])


def content_hash(info: dict[str, Any], fees: dict[str, Any]) -> str:
    """SHA-256 du contenu utile : réponse sans ``serverTime`` + frais, JSON à clés triées."""
    stable = {k: v for k, v in info.items() if k != "serverTime"}
    text = json.dumps(
        {"exchange_info": stable, "fees": fees},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Un snapshot chargé. ``info`` est la réponse ``exchangeInfo`` telle que reçue."""

    path: Path
    source: str
    fetched_ms: int
    content_hash: str
    fees: dict[str, Any]
    info: dict[str, Any]

    def symbols(self, status: str | None = "TRADING") -> list[dict[str, Any]]:
        """Paires (dicts bruts), filtrées sur ``status`` ; ``None`` = toutes."""
        return [s for s in self.info["symbols"] if status is None or s["status"] == status]

    def by_symbol(self) -> dict[str, dict[str, Any]]:
        return {s["symbol"]: s for s in self.info["symbols"]}

    def filters(self, symbol: str) -> SymbolFilters:
        s = self.by_symbol().get(symbol)
        if s is None:
            raise DataError(f"{symbol} absent du snapshot {self.path.name}")
        return SymbolFilters.from_binance(symbol, s["filters"])


def _ms_to_name(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms // 1000, tz=UTC).strftime("%Y%m%dT%H%M%SZ") + ".json.zst"


def _name_to_ms(name: str) -> int:
    d = datetime.strptime(name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return int(d.timestamp()) * 1000


class SnapshotStore:
    """Snapshots d'une source (``binance``…) sous ``DataPaths.exchange_info_dir(source)``."""

    def __init__(self, paths: DataPaths, source: str) -> None:
        self.source = source
        self.dir = paths.exchange_info_dir(source)

    def paths(self) -> list[Path]:
        """Snapshots triés du plus ancien au plus récent (le nom encode l'heure UTC)."""
        if not self.dir.exists():
            return []
        return sorted(p for p in self.dir.iterdir() if _FILE_RE.match(p.name))

    def load(self, path: Path) -> Snapshot:
        try:
            raw = zstandard.ZstdDecompressor().decompress(
                path.read_bytes(), max_output_size=MAX_SNAPSHOT_BYTES
            )
            doc = json.loads(raw)
            snap = Snapshot(
                path,
                doc["source"],
                doc["fetched_ms"],
                doc["content_hash"],
                doc["fees"],
                doc["exchange_info"],
            )
        except (zstandard.ZstdError, ValueError, KeyError, TypeError) as exc:
            raise DataError(f"snapshot illisible : {path}") from exc
        if content_hash(snap.info, snap.fees) != snap.content_hash:
            raise DataError(f"snapshot altéré (hash incohérent) : {path}")
        if snap.fetched_ms != _name_to_ms(path.name):
            raise DataError(f"nom de fichier et heure du snapshot incohérents : {path}")
        return snap

    def latest(self) -> Snapshot | None:
        paths = self.paths()
        return self.load(paths[-1]) if paths else None

    def snapshot_at(self, ts_ms: int) -> tuple[Snapshot, bool]:
        """Version en vigueur à ``ts_ms`` (prise à ``ts_ms`` ou avant) : ``(snapshot, extrapolé)``.

        ``extrapolé`` : ``ts_ms`` précède le premier snapshot, qu'on applique alors au passé.
        Aucun snapshot : ``DataError``.
        """
        paths = self.paths()
        if not paths:
            raise DataError(f"aucun snapshot exchangeInfo dans {self.dir}")
        eligible = [p for p in paths if _name_to_ms(p.name) <= ts_ms]
        return self.load(eligible[-1] if eligible else paths[0]), not eligible

    def save(
        self, info: dict[str, Any], fees: dict[str, Any], fetched_ms: int, url: str
    ) -> SaveResult:
        """Écrit un snapshot si le contenu a changé (sinon rien n'est écrit).

        ``fetched_ms`` est arrondi à la seconde inférieure pour correspondre au nom du fichier.
        """
        validate_exchange_info(info)
        fetched_ms -= fetched_ms % 1000
        digest = content_hash(info, fees)
        last = self.latest()
        if last is not None:
            if last.content_hash == digest:
                return SaveResult(last, written=False, previous=None)
            if fetched_ms <= last.fetched_ms:
                raise DataError("snapshot pas plus récent que le dernier enregistré")
        doc = {
            "source": self.source,
            "fetched_ms": fetched_ms,
            "url": url,
            "content_hash": digest,
            "fees": fees,
            "exchange_info": info,
        }
        data = zstandard.ZstdCompressor(level=19).compress(
            json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        write_atomic(self.dir / _ms_to_name(fetched_ms), data)
        path = self.dir / _ms_to_name(fetched_ms)
        return SaveResult(
            Snapshot(path, self.source, fetched_ms, digest, fees, info),
            written=True,
            previous=last,
        )


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Issue de ``save`` : ``snapshot`` courant, ``written`` (nouvelle version écrite ?) et
    ``previous`` (version précédente si une nouvelle a été écrite ; ``None`` sinon ou si c'est
    la toute première)."""

    snapshot: Snapshot
    written: bool
    previous: Snapshot | None


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    """Différences entre deux versions. ``status_changed`` : ``(paire, avant, après)``."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    status_changed: tuple[tuple[str, str, str], ...]
    filters_changed: tuple[str, ...]
    fees_changed: bool

    @property
    def is_empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.status_changed
            or self.filters_changed
            or self.fees_changed
        )

    def touching(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        """Paires de ``symbols`` touchées (ajout, disparition, statut, filtres), dans l'ordre."""
        hit = {
            *self.added,
            *self.removed,
            *(s for s, _, _ in self.status_changed),
            *self.filters_changed,
        }
        return tuple(s for s in symbols if s in hit)


def diff_snapshots(old: Snapshot, new: Snapshot) -> SnapshotDiff:
    """Compare deux versions (paires et filtres bruts, frais)."""
    before, after = old.by_symbol(), new.by_symbol()
    common = sorted(before.keys() & after.keys())
    return SnapshotDiff(
        added=tuple(sorted(after.keys() - before.keys())),
        removed=tuple(sorted(before.keys() - after.keys())),
        status_changed=tuple(
            (s, before[s]["status"], after[s]["status"])
            for s in common
            if before[s]["status"] != after[s]["status"]
        ),
        filters_changed=tuple(s for s in common if before[s]["filters"] != after[s]["filters"]),
        fees_changed=old.fees != new.fees,
    )
