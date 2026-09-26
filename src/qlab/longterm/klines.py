"""Bougies spot long terme : lecture des archives Binance, contrôle qualité, trous.

Format et contrôle d'une série de bougies d'une paire, lue dans les archives mensuelles puis
quotidiennes (vérifiées au téléchargement, jamais modifiées) :

1. une archive zip = un CSV Binance (en-tête facultatif) ;
2. horodatages ramenés en ms : Binance est passé des ms aux µs le 2025-01-01 ; l'unité est
   reconnue **ligne par ligne** à sa magnitude (une archive KLAYBTC d'octobre 2024 mélange les
   deux), puis chaque date doit tomber entre 2000 et 2100 (``timeutils.infer_epoch_unit``) ;
3. **contrôle qualité** : bougie alignée sur l'intervalle, clôture dans l'intervalle,
   ``low ≤ open, close ≤ high``, prix > 0, volumes ≥ 0. Une bougie fautive est **écartée** et
   comptée : elle devient un trou, jamais une valeur corrigée (ex. : bougies horaires décalées
   de 28 min après l'arrêt de Binance du 2018-02-08) ;
   une bougie **tronquée** (clôture avant la fin de l'intervalle : dernière bougie avant un
   retrait, arrêt de cotation) est gardée avec ``partial`` vrai : écarter le dernier prix avant
   un retrait favoriserait les survivants ;
4. doublons : identiques ⇒ un seul ; différents (mensuelle contre quotidienne) ⇒ la mensuelle,
   publiée après coup, fait foi, et le conflit est compté ;
5. trous : jamais interpolés ; ``gaps`` les liste à la lecture (LT.1 : période exclue).

Une bougie d'archive est toujours close. Les derniers jours viendront du REST (paper trading).
Construction sur disque : ``longterm/klines_build.py``.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, MS_PER_HOUR, US_PER_MS, date_to_ms, infer_epoch_unit

INTERVAL_MS = {"1d": MS_PER_DAY, "1h": MS_PER_HOUR}
_RAW = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "close_time",
    "volume_quote",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)
_INTS = {"open_time", "close_time", "n_trades"}
# « ignore » : champ inutilisé, entier en général mais décimal dans des archives de 2017.
SCHEMA = {n: pl.String if n == "ignore" else pl.Int64 if n in _INTS else pl.Float64 for n in _RAW}
COLUMNS: tuple[str, ...] = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "volume_quote",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
)

OUTPUT = (*COLUMNS, "partial")
_PARSED = {c: SCHEMA.get(c, pl.Int64) for c in (*COLUMNS, "close_time_ms")}
_US_FROM = date_to_ms("2000-01-01") * US_PER_MS  # au-delà : µs (les ms s'arrêtent en 2100)


def interval_to_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise DataError(f"intervalle non géré : {interval!r} (connus : {sorted(INTERVAL_MS)})")
    return INTERVAL_MS[interval]


def parse_csv(data: bytes) -> pl.DataFrame:
    """CSV d'archive Binance (en-tête facultatif) → colonnes de base + ``close_time_ms``."""
    if not data.strip():
        return pl.DataFrame(schema=_PARSED)
    has_header = not data[:1].isdigit()
    df = pl.read_csv(
        io.BytesIO(data), has_header=has_header, new_columns=list(_RAW), schema_overrides=SCHEMA
    )
    out = df.select(
        _to_ms("open_time").alias("open_time_ms"), *COLUMNS[1:], close_time_ms=_to_ms("close_time")
    )
    for ts in (out["open_time_ms"].min(), out["open_time_ms"].max()):
        if not _plausible_ms(ts):
            raise DataError(f"archive de bougies : horodatage hors 2000–2100 ({ts!r})")
    return out


def _plausible_ms(ts: object) -> bool:
    try:
        return isinstance(ts, int) and infer_epoch_unit(ts) == "ms"
    except ValueError:
        return False


def _to_ms(column: str) -> pl.Expr:
    c = pl.col(column)
    return pl.when(c >= _US_FROM).then(c // US_PER_MS).otherwise(c)


def read_archive(path: Path) -> pl.DataFrame:
    """Une archive zip contenant un seul CSV."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise DataError(f"{path.name} : un seul fichier attendu dans l'archive, {len(names)}")
        data = z.read(names[0])
    try:
        return parse_csv(data)
    except pl.exceptions.ComputeError as e:
        raise DataError(f"{path.name} : CSV illisible ({str(e).splitlines()[0]})") from e


def valid_mask(interval_ms: int) -> pl.Expr:
    """Vrai pour une bougie cohérente (voir en-tête du module, point 3)."""
    t = pl.col("open_time_ms")
    return (
        (t % interval_ms == 0)
        & pl.col("close_time_ms").is_between(t, t + interval_ms - 1)
        & (pl.col("low") > 0)
        & (pl.col("low") <= pl.min_horizontal("open", "close"))
        & (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.min_horizontal("volume_base", "volume_quote", "n_trades") >= 0)
    )


@dataclass(frozen=True, slots=True)
class Merged:
    bars: pl.DataFrame  # OUTPUT, triées par open_time_ms, une ligne par bougie
    anomalies: int
    conflicts: int


def merge(frames: Sequence[pl.DataFrame], interval_ms: int) -> Merged:
    """Assemble les archives dans l'ordre de priorité donné (mensuelles d'abord)."""
    if not frames:
        return Merged(_output(pl.DataFrame(schema=_PARSED), interval_ms), 0, 0)
    ranked = pl.concat([f.with_columns(_rank=pl.lit(i)) for i, f in enumerate(frames)])
    ok = ranked.filter(valid_mask(interval_ms))
    distinct = ok.unique(subset=list(COLUMNS), keep="first").sort("open_time_ms", "_rank")
    bars = distinct.unique(subset=["open_time_ms"], keep="first", maintain_order=True)
    return Merged(
        _output(bars, interval_ms),
        anomalies=ranked.height - ok.height,
        conflicts=distinct.height - bars.height,
    )


def _output(bars: pl.DataFrame, interval_ms: int) -> pl.DataFrame:
    end = pl.col("open_time_ms") + interval_ms - 1
    return bars.select(*COLUMNS, partial=pl.col("close_time_ms") < end)


def gaps(open_times: pl.Series, interval_ms: int) -> pl.DataFrame:
    """Trous entre la première et la dernière bougie : début du trou, bougies manquantes."""
    t = open_times.sort()
    step = t.diff()
    missing = (step // interval_ms - 1).alias("missing")
    frame = pl.DataFrame({"start_ms": t.shift(1) + interval_ms, "missing": missing})
    return frame.filter(pl.col("missing") > 0)


def load(paths: DataPaths, interval: str, symbol: str) -> pl.DataFrame:
    """Bougies construites d'une paire (``DataError`` si absentes : lancer ``build``)."""
    path = paths.lt_klines(interval, symbol)
    if not path.exists():
        raise DataError(f"{symbol} {interval} : bougies absentes, lancer « klines_build build »")
    return pl.read_parquet(path)
