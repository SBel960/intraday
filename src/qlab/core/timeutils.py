"""Temps UTC en entiers : epoch en millisecondes (``_ms``) ou microsecondes (``_us``).

Aucun flottant : toutes les conversions sont des multiplications ou des divisions entières
par défaut (``//``, arrondi vers −∞, y compris avant 1970). Les fonctions scalaires servent au
code de contrôle ; les expressions Polars (suffixe ``_expr``) font le même calcul sur colonnes.

Convention jour de semaine : 0 = lundi … 6 = dimanche (comme ``datetime.weekday``).
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Literal

import polars as pl

US_PER_MS = 1_000
MS_PER_S = 1_000
US_PER_S = US_PER_MS * MS_PER_S
MS_PER_MIN = 60 * MS_PER_S
MS_PER_HOUR = 60 * MS_PER_MIN
MS_PER_DAY = 24 * MS_PER_HOUR
MINUTES_PER_DAY = 1_440
EPOCH_WEEKDAY = 3  # 1970-01-01 était un jeudi

# Bornes de plausibilité pour détecter l'unité : 2000-01-01 ≤ t < 2100-01-01 (UTC).
_MIN_PLAUSIBLE_MS = 946_684_800_000
_MAX_PLAUSIBLE_MS = 4_102_444_800_000

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)

EpochUnit = Literal["ms", "us"]


def _int(name: str, x: object) -> int:
    """Refuse tout ce qui n'est pas un entier (bool et float compris)."""
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{name} : entier attendu, reçu {type(x).__name__} ({x!r})")
    return x


def _slot_ms(slot_minutes: int) -> int:
    _int("slot_minutes", slot_minutes)
    if slot_minutes <= 0 or MINUTES_PER_DAY % slot_minutes != 0:
        raise ValueError(f"slot_minutes doit diviser 1440 (reçu {slot_minutes})")
    return slot_minutes * MS_PER_MIN


# --- horloge -----------------------------------------------------------------------------


def now_ms() -> int:
    """Heure courante UTC, epoch en ms (horloge système, sans flottant)."""
    return time.time_ns() // 1_000_000


def now_us() -> int:
    """Heure courante UTC, epoch en µs (horloge système, sans flottant)."""
    return time.time_ns() // 1_000


# --- conversions d'unité -----------------------------------------------------------------


def ms_to_us(ts_ms: int) -> int:
    """Epoch ms → epoch µs (exact)."""
    return _int("ts_ms", ts_ms) * US_PER_MS


def us_to_ms(ts_us: int) -> int:
    """Epoch µs → epoch ms, arrondi vers −∞ (la ms qui contient l'instant)."""
    return _int("ts_us", ts_us) // US_PER_MS


def infer_epoch_unit(ts: int) -> EpochUnit:
    """Devine l'unité d'un epoch entier par sa magnitude : ``"ms"`` ou ``"us"``.

    Utile pour les archives Binance spot, passées de ms à µs le 2025-01-01. Seules les dates
    entre 2000-01-01 et 2100-01-01 sont acceptées ; sinon ``ValueError`` (jamais de devinette
    silencieuse). Les deux plages ne se chevauchent pas.
    """
    _int("ts", ts)
    if _MIN_PLAUSIBLE_MS <= ts < _MAX_PLAUSIBLE_MS:
        return "ms"
    if _MIN_PLAUSIBLE_MS * US_PER_MS <= ts < _MAX_PLAUSIBLE_MS * US_PER_MS:
        return "us"
    raise ValueError(f"epoch hors plage 2000–2100 en ms comme en µs : {ts}")


def to_us(ts: int, unit: EpochUnit) -> int:
    """Epoch dans l'unité ``unit`` → epoch µs."""
    if unit == "us":
        return _int("ts", ts)
    if unit == "ms":
        return ms_to_us(ts)
    raise ValueError(f"unité inconnue : {unit!r}")


# --- découpage du temps ------------------------------------------------------------------


def floor_ms(ts_ms: int, step_ms: int) -> int:
    """Début de l'intervalle de ``step_ms`` ms (aligné sur l'epoch) contenant ``ts_ms``."""
    _int("ts_ms", ts_ms)
    if _int("step_ms", step_ms) <= 0:
        raise ValueError(f"step_ms doit être > 0 (reçu {step_ms})")
    return ts_ms - ts_ms % step_ms


def day_start_ms(ts_ms: int) -> int:
    """Minuit UTC du jour contenant ``ts_ms`` (epoch ms)."""
    return floor_ms(ts_ms, MS_PER_DAY)


def slot_of_day(ts_ms: int, slot_minutes: int) -> int:
    """Indice de la tranche UTC de ``slot_minutes`` dans la journée : 0 … 1440/slot − 1."""
    slot_ms = _slot_ms(slot_minutes)
    return (_int("ts_ms", ts_ms) % MS_PER_DAY) // slot_ms


def weekday(ts_ms: int) -> int:
    """Jour de semaine UTC de ``ts_ms`` : 0 = lundi … 6 = dimanche."""
    return (_int("ts_ms", ts_ms) // MS_PER_DAY + EPOCH_WEEKDAY) % 7


def hour_of_day(ts_ms: int) -> int:
    """Heure UTC de ``ts_ms`` : 0 … 23 (sert aux fichiers RAW horaires)."""
    return (_int("ts_ms", ts_ms) % MS_PER_DAY) // MS_PER_HOUR


# --- dates et texte ----------------------------------------------------------------------


def date_str(ts_ms: int) -> str:
    """Date UTC ``YYYY-MM-DD`` du jour contenant ``ts_ms`` (partitions ``date=``)."""
    days = _int("ts_ms", ts_ms) // MS_PER_DAY
    return (_EPOCH + dt.timedelta(days=days)).date().isoformat()


def date_to_ms(date_iso: str) -> int:
    """``YYYY-MM-DD`` → epoch ms de minuit UTC. Format strict, sinon ``ValueError``."""
    try:
        d = dt.datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=dt.UTC)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"date YYYY-MM-DD attendue, reçu {date_iso!r}") from exc
    if d.strftime("%Y-%m-%d") != date_iso:  # refuse « 2024-1-5 »
        raise ValueError(f"date YYYY-MM-DD attendue, reçu {date_iso!r}")
    delta = d - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * MS_PER_S


def ms_to_iso(ts_ms: int) -> str:
    """Epoch ms → ``YYYY-MM-DDTHH:MM:SS.mmmZ`` (journaux et rapports ; arithmétique entière)."""
    d = _EPOCH + dt.timedelta(milliseconds=_int("ts_ms", ts_ms))
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // US_PER_MS:03d}Z"


# --- expressions Polars (mêmes calculs, sur colonnes Int64 d'epoch ms) --------------------


def slot_of_day_expr(ts_ms: pl.Expr, slot_minutes: int) -> pl.Expr:
    """Version colonne de :func:`slot_of_day`."""
    return (ts_ms % MS_PER_DAY) // _slot_ms(slot_minutes)


def weekday_expr(ts_ms: pl.Expr) -> pl.Expr:
    """Version colonne de :func:`weekday` (0 = lundi)."""
    return (ts_ms // MS_PER_DAY + EPOCH_WEEKDAY) % 7


def floor_ms_expr(ts_ms: pl.Expr, step_ms: int) -> pl.Expr:
    """Version colonne de :func:`floor_ms`."""
    if _int("step_ms", step_ms) <= 0:
        raise ValueError(f"step_ms doit être > 0 (reçu {step_ms})")
    return ts_ms - ts_ms % step_ms
