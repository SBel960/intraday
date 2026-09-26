"""Tardis.dev : cotations top-of-book historiques (``book_ticker``) de Binance spot.

Seul le **1er jour de chaque mois** est gratuit (sans clé) ; il sert d'échantillon mensuel de
spreads réels pour le cost gate. Fichier : ``{tardis_url}/v1/binance/book_ticker/YYYY/MM/DD/
{SYMBOLE}.csv.gz`` ; colonnes ``exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,
bid_price,bid_amount`` ; horodatages en **microsecondes**.

- Disponibilité : dates officielles de l'API Tardis (``availableSince`` par paire), jamais
  devinées à partir de messages d'erreur.
- Stockage : ``raw/tardis/binance/book_ticker/{SYMBOLE}/YYYY-MM-DD.csv.gz`` tel que reçu.
- Vérification (Tardis ne publie pas d'empreinte) : gzip intègre (CRC), en-tête exact, paire et
  échange attendus. Téléchargement par ``core/downloads`` (parallèle, atomique, reprise).
- ``read_quotes`` : ``ts_ms`` (Int64, horodatage d'échange µs → ms), ``bid``, ``ask``.

Commande : ``python -m qlab.data.tardis --config config sync [--symbols A,B] [--dry-run]``
(par défaut : paires tradées).
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Callable, Sequence
from datetime import date
from functools import partial
from pathlib import Path

import polars as pl

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.downloads import Job, download_all
from qlab.core.errors import DataError, ExchangeError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import date_str, now_ms

SOURCE = "tardis"
EXCHANGE = "binance"
DATA_TYPE = "book_ticker"
HEADER = "exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,bid_amount"
MAX_CSV_BYTES = 4 << 30  # 4 Gio décompressés : borne contre un fichier piégé
Fetch = Callable[[str], bytes]


def availability(fetch: Fetch, api_url: str) -> dict[str, str]:
    """Paire spot → première date disponible (``YYYY-MM-DD``) d'après l'API Tardis."""
    try:
        doc = json.loads(fetch(f"{api_url.rstrip('/')}/v1/exchanges/{EXCHANGE}"))
        return {
            s["id"].upper(): str(s["availableSince"])[:10]
            for s in doc["availableSymbols"]
            if s.get("type") == "spot"
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ExchangeError("métadonnées Tardis illisibles (availableSymbols)") from exc


def free_days(since: str, today: str) -> list[str]:
    """1ers du mois gratuits, de la 1re date disponible (``since`` inclus) à hier inclus."""
    start, end = date.fromisoformat(since), date.fromisoformat(today)
    year, month = start.year, start.month
    if start.day > 1:  # le 1er de ce mois-là précède la disponibilité
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    days = []
    while date(year, month, 1) < end:
        days.append(date(year, month, 1).isoformat())
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return days


def file_url(base_url: str, symbol: str, day: str) -> str:
    y, m, d = day.split("-")
    return f"{base_url.rstrip('/')}/v1/{EXCHANGE}/{DATA_TYPE}/{y}/{m}/{d}/{symbol}.csv.gz"


def file_key(symbol: str, day: str) -> str:
    """Chemin relatif sous ``raw/tardis/``."""
    return f"{EXCHANGE}/{DATA_TYPE}/{symbol}/{day}.csv.gz"


def local_files(paths: DataPaths, symbol: str) -> list[Path]:
    """Journées Tardis déjà téléchargées (et donc vérifiées) pour ``symbol``, par date."""
    folder = paths.raw_archive(SOURCE, f"{EXCHANGE}/{DATA_TYPE}/{symbol}")
    return sorted(folder.glob("????-??-??.csv.gz")) if folder.is_dir() else []


def verify(data: bytes, symbol: str) -> bytes:
    """Contrôle un ``.csv.gz`` Tardis (gzip intègre, en-tête, paire) ; renvoie ``data``."""
    inflater = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        text = inflater.decompress(data, MAX_CSV_BYTES)
    except zlib.error as exc:
        raise DataError(f"{symbol} : gzip corrompu") from exc
    if not inflater.eof or inflater.unconsumed_tail:
        raise DataError(f"{symbol} : gzip tronqué ou trop volumineux")
    lines = text.split(b"\n", 2)
    if lines[0].decode("utf-8", errors="replace").strip() != HEADER:
        raise DataError(f"{symbol} : en-tête Tardis inattendu")
    if len(lines) < 2 or not lines[1].startswith(f"{EXCHANGE},{symbol},".encode()):
        raise DataError(f"{symbol} : fichier vide ou d'une autre paire / d'un autre échange")
    return data


def read_quotes(csv_gz: bytes) -> pl.DataFrame:
    """Cotations d'un fichier Tardis : ``ts_ms`` (Int64), ``bid``, ``ask`` (Float64), triées."""
    frame = pl.read_csv(
        zlib.decompress(csv_gz, wbits=16 + zlib.MAX_WBITS),
        schema_overrides={"timestamp": pl.Int64, "bid_price": pl.Float64, "ask_price": pl.Float64},
        columns=["timestamp", "bid_price", "ask_price"],
    )
    quotes = frame.select(
        (pl.col("timestamp") // 1000).alias("ts_ms"),
        pl.col("bid_price").alias("bid"),
        pl.col("ask_price").alias("ask"),
    )
    return quotes.sort("ts_ms", maintain_order=True)


def _fetch_verified(fetch: Fetch, url: str, symbol: str) -> bytes:
    return verify(fetch(url), symbol)


def jobs(
    paths: DataPaths,
    fetch: Fetch,
    *,
    base_url: str,
    symbols: Sequence[str],
    since: dict[str, str],
    today: str,
) -> list[Job]:
    """Un ``Job`` par (paire, 1er du mois disponible)."""
    out = []
    for symbol in symbols:
        if symbol not in since:
            raise DataError(f"{symbol} inconnu de Tardis (spot {EXCHANGE})")
        for day in free_days(since[symbol], today):
            url = file_url(base_url, symbol, day)
            key = file_key(symbol, day)
            out.append(
                Job(
                    key,
                    paths.raw_archive(SOURCE, key),
                    partial(_fetch_verified, fetch, url, symbol),
                )
            )
    return out


# --- commande ----------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["sync"])
    parser.add_argument("--symbols", help="paires, ex. BTCEUR,ETHEUR ; défaut : symbols.trade")
    parser.add_argument("--dry-run", action="store_true", help="compter sans télécharger")


def _action(ctx: Context) -> int:
    cfg = ctx.config.base

    def fetch(url: str) -> bytes:
        return http.get(url, notify=ctx.notify).body

    symbols = (
        list(cfg.symbols.trade)
        if ctx.args.symbols is None
        else [s.strip() for s in ctx.args.symbols.split(",")]
    )
    since = availability(fetch, cfg.archives.tardis_api_url)
    todo = jobs(
        ctx.paths,
        fetch,
        base_url=cfg.archives.tardis_url,
        symbols=symbols,
        since=since,
        today=date_str(now_ms()),
    )
    print(f"{len(todo)} journées gratuites pour {', '.join(symbols)}")
    report = download_all(todo, workers=cfg.archives.download_workers, dry_run=ctx.args.dry_run)
    print(
        f"Déjà présentes : {report.present} ; à télécharger : {report.missing}"
        + (
            " (simulation)"
            if ctx.args.dry_run
            else f" ; téléchargées : {report.downloaded} ({report.downloaded_bytes / 1e6:.1f} Mo)"
        )
    )
    for key, msg in report.failed:
        print(f"ÉCHEC : {key} : {msg}")
    ctx.journal.info(
        "tardis.sync",
        {
            "symbols": list(symbols),
            "listed": report.listed,
            "present": report.present,
            "downloaded": report.downloaded,
            "bytes": report.downloaded_bytes,
            "failed": list(k for k, _ in report.failed),
            "dry_run": ctx.args.dry_run,
        },
    )
    return 1 if report.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.data.tardis",
        description="Cotations book_ticker gratuites de Tardis.dev",
        component="tardis",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
