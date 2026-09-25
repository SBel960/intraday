"""Journal structuré JSONL, append-only : événements, décisions, erreurs.

Un fichier par composant et par jour UTC : ``{log_dir}/{component}/{YYYY-MM-DD}.jsonl``.
Chaque ligne est un objet JSON complet, sérialisé de façon déterministe (clés triées) ::

    {"component": "paper", "data": {...}, "kind": "order.sent", "level": "info", "ts_ms": 1704…}

``ts_ms`` est l'epoch UTC en ms de l'horloge injectée (``now_ms`` par défaut), ce qui rend
les tests déterministes. Les valeurs doivent être du JSON strict : pas de NaN/inf, pas de
conversion silencieuse d'objets (``TypeError``). Chaque ligne est écrite d'un seul appel
système puis vidée ; ``fsync=True`` force l'écriture disque (décisions de trading).

Crash pendant une écriture : la dernière ligne peut être tronquée. À la réouverture, l'écrivain
termine la ligne partielle par ``\\n`` ; la lecture la signale dans ``corrupt_lines`` au lieu
de l'ignorer en silence.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from qlab.core.timeutils import date_str, now_ms

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
Level = Literal["info", "warning", "error"]

LEVELS: tuple[Level, ...] = ("info", "warning", "error")
_NAME_RE = re.compile(r"^[a-z0-9_]+$")  # composant : sert de nom de dossier
_KIND_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")  # ex. "order.sent"
_REQUIRED = frozenset({"ts_ms", "level", "component", "kind", "data"})


def _encode(record: Mapping[str, object]) -> bytes:
    """Sérialisation stricte et déterministe d'un enregistrement en une ligne UTF-8."""
    try:
        text = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        )
    except ValueError as exc:  # NaN ou inf
        raise ValueError(f"valeur non finie dans le journal : {exc}") from exc
    return (text + "\n").encode("utf-8")


class JsonLog:
    """Écrivain de journal pour un composant ; fichier choisi par la date UTC de l'événement."""

    def __init__(
        self,
        log_dir: Path,
        component: str,
        *,
        clock_ms: Callable[[], int] = now_ms,
        fsync: bool = False,
    ) -> None:
        if not _NAME_RE.match(component):
            raise ValueError(f"component invalide (attendu [a-z0-9_]+) : {component!r}")
        self._dir = log_dir / component
        self._component = component
        self._clock_ms = clock_ms
        self._fsync = fsync
        self._fd: int | None = None
        self._fd_date: str | None = None
        self._closed = False

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, ts_ms: int) -> Path:
        """Fichier du jour UTC contenant ``ts_ms``."""
        return self._dir / f"{date_str(ts_ms)}.jsonl"

    def _fd_for(self, ts_ms: int) -> int:
        day = date_str(ts_ms)
        if self._fd is not None and self._fd_date == day:
            return self._fd
        self._close_fd()
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(ts_ms)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        size = os.fstat(fd).st_size
        if size > 0:
            with path.open("rb") as f:
                f.seek(size - 1)
                if f.read(1) != b"\n":  # ligne tronquée par un crash : on la termine
                    os.write(fd, b"\n")
        self._fd, self._fd_date = fd, day
        return fd

    def log(self, level: Level, kind: str, data: Mapping[str, JsonValue] | None = None) -> None:
        """Écrit un enregistrement ``{ts_ms, level, component, kind, data}``."""
        if self._closed:
            raise RuntimeError("journal fermé")
        if level not in LEVELS:
            raise ValueError(f"level invalide : {level!r}")
        if not _KIND_RE.match(kind):
            raise ValueError(f"kind invalide (attendu a.b_c) : {kind!r}")
        payload = dict(data or {})
        if not all(isinstance(k, str) for k in payload):
            raise TypeError("les clés de data doivent être des str")
        ts_ms = self._clock_ms()
        if not isinstance(ts_ms, int) or isinstance(ts_ms, bool):
            raise TypeError(f"l'horloge doit renvoyer un int (epoch ms), reçu {ts_ms!r}")
        line = _encode(
            {
                "ts_ms": ts_ms,
                "level": level,
                "component": self._component,
                "kind": kind,
                "data": payload,
            }
        )
        fd = self._fd_for(ts_ms)
        written = os.write(fd, line)
        if written != len(line):
            raise OSError(f"écriture partielle du journal ({written}/{len(line)} octets)")
        if self._fsync:
            os.fsync(fd)

    def info(self, kind: str, data: Mapping[str, JsonValue] | None = None) -> None:
        self.log("info", kind, data)

    def warning(self, kind: str, data: Mapping[str, JsonValue] | None = None) -> None:
        self.log("warning", kind, data)

    def error(self, kind: str, data: Mapping[str, JsonValue] | None = None) -> None:
        self.log("error", kind, data)

    def _close_fd(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd, self._fd_date = None, None

    def close(self) -> None:
        self._close_fd()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class LogRead:
    """Résultat de lecture : enregistrements valides et numéros (1-based) des lignes corrompues."""

    records: tuple[dict[str, JsonValue], ...]
    corrupt_lines: tuple[int, ...]


def _valid(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and set(obj) == _REQUIRED
        and isinstance(obj["ts_ms"], int)
        and not isinstance(obj["ts_ms"], bool)
        and obj["level"] in LEVELS
        and isinstance(obj["kind"], str)
        and isinstance(obj["component"], str)
        and isinstance(obj["data"], dict)
    )


def _reject_constant(name: str) -> object:
    raise ValueError(f"constante JSON non standard : {name}")


def read_log(path: Path) -> LogRead:
    """Lit un fichier de journal. Les lignes illisibles ou hors schéma sont listées, jamais tues.

    ``FileNotFoundError`` si le fichier n'existe pas.
    """
    records: list[dict[str, JsonValue]] = []
    corrupt: list[int] = []
    with path.open("rb") as f:
        for n, raw in enumerate(f, start=1):
            try:
                obj = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError):
                corrupt.append(n)
                continue
            if _valid(obj):
                records.append(obj)
            else:
                corrupt.append(n)
    return LogRead(tuple(records), tuple(corrupt))
