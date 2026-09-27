"""Dernières bougies par l'API REST : le complément des archives pour le paper trading.

Les archives (``klines.py``) s'arrêtent la veille ou l'avant-veille ; le paper décide à la
clôture du jour et exécute à l'ouverture suivante (LT.5). Ce module :

1. lit ``/api/v3/klines`` (format des archives : mêmes 12 champs, mêmes types) par
   ``core/http.get`` ;
2. applique le **même contrôle qualité** que les archives (``klines.valid_mask``) : une bougie
   fautive est écartée et comptée, jamais corrigée ;
3. sépare les bougies **closes** (``close_time_ms`` < heure du serveur) de la bougie **en
   cours**, dont seule l'ouverture sert (prix de référence de l'exécution) ;
4. ``extend`` : archives + bougies closes plus récentes ; en cas de chevauchement, l'archive
   fait foi. Un trou entre les deux reste un trou (jamais interpolé).

L'heure du serveur est fournie par l'appelant (``now_ms`` + écart mesuré) : l'horloge du PC
peut dériver.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from qlab.core import http
from qlab.core.errors import DataError, ExchangeError
from qlab.longterm import klines

ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000  # plafond Binance par requête


@dataclass(frozen=True, slots=True)
class Recent:
    closed: pl.DataFrame  # colonnes ``klines.OUTPUT``, bougies closes et valides
    current_open_ms: int | None  # bougie en cours : début et ouverture (None si absente)
    current_open: float | None
    anomalies: int  # bougies écartées par le contrôle qualité


def parse(rows: Any, interval_ms: int, server_ms: int) -> Recent:
    """Réponse JSON de ``/api/v3/klines`` → bougies closes valides + bougie en cours."""
    if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != 12 for r in rows):
        raise ExchangeError("klines REST : liste de bougies à 12 champs attendue")
    names = list(klines.SCHEMA)
    frame = pl.DataFrame(
        [dict(zip(names, (str(v) for v in r), strict=True)) for r in rows],
        schema=dict.fromkeys(names, pl.String),
    ).cast({n: t for n, t in klines.SCHEMA.items() if n != "ignore"})
    bars = frame.select(
        pl.col("open_time").alias("open_time_ms"),
        *klines.COLUMNS[1:],
        close_time_ms=pl.col("close_time"),
    )
    done = pl.col("close_time_ms") < server_ms
    live = bars.filter(~done)
    closed = bars.filter(done)
    valid = closed.filter(klines.valid_mask(interval_ms))
    end = pl.col("open_time_ms") + interval_ms - 1
    out = valid.select(*klines.COLUMNS, partial=pl.col("close_time_ms") < end)
    first = live.row(0, named=True) if live.height else None
    return Recent(
        out.sort("open_time_ms"),
        None if first is None else first["open_time_ms"],
        None if first is None else first["open"],
        closed.height - valid.height,
    )


def fetch(
    rest_url: str,
    symbol: str,
    interval: str,
    since_ms: int,
    server_ms: int,
    *,
    notify: Callable[[str], None],
    get: Callable[..., http.HttpResult] = http.get,
) -> Recent:
    """Bougies de ``symbol`` ouvertes depuis ``since_ms`` (au plus ``MAX_LIMIT``)."""
    interval_ms = klines.interval_to_ms(interval)
    if (server_ms - since_ms) // interval_ms + 1 > MAX_LIMIT:
        raise DataError(
            f"{symbol} : plus de {MAX_LIMIT} bougies à rattraper, reconstruire les archives"
        )
    query = urllib.parse.urlencode(
        {"symbol": symbol, "interval": interval, "startTime": since_ms, "limit": MAX_LIMIT}
    )
    result = get(f"{rest_url.rstrip('/')}{ENDPOINT}?{query}", notify=notify)
    try:
        rows = json.loads(result.body)
    except ValueError as exc:
        raise ExchangeError(f"{symbol} : réponse klines non JSON") from exc
    return parse(rows, interval_ms, server_ms)


def extend(archive: pl.DataFrame, recent: pl.DataFrame) -> pl.DataFrame:
    """Archives puis bougies REST strictement plus récentes (l'archive fait foi)."""
    if archive.columns != list(klines.OUTPUT) or recent.columns != list(klines.OUTPUT):
        raise DataError(f"colonnes {klines.OUTPUT} attendues des deux côtés")
    last = archive["open_time_ms"].max()
    newer = recent if last is None else recent.filter(pl.col("open_time_ms") > last)
    return pl.concat([archive, newer.cast(archive.schema)])
