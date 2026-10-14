"""Barres d'une minute de HistData (gratuites) → barres horaires UTC du volet trading.

La source publie un zip par (instrument, année) — par mois pour l'année en cours — contenant un
CSV ``AAAAMMJJ HHMMSS;ouverture;plus haut;plus bas;clôture;volume`` à l'heure de New York
**sans heure d'été** (UTC − 5 toute l'année : ``trading.source_utc_offset_h``). Le zip se
télécharge en deux temps : page de la période (jeton ``tk`` du formulaire), puis formulaire
envoyé avec cette page comme ``Referer``. Pas de bid / ask : le spread vient de la config.

- ``download`` : zip brut dans ``raw/histdata/{instrument}/{période}.zip`` (jamais modifié ; un
  fichier présent n'est pas retéléchargé ; réponse vide = période absente chez la source,
  gardée vide pour ne pas la redemander) ; périodes complètes seulement ;
- ``build`` : minutes → heures UTC (ouverture de la 1re minute, plus haut, plus bas, clôture
  de la dernière, ``volume`` = minutes cotées) + ``spread_frac`` de l'instrument →
  ``trading/hourly/{instrument}.parquet``.

Commandes : ``python -m qlab.data.histdata --config config download [--only EURUSD]`` ;
``… build``
"""

from __future__ import annotations

import argparse
import io
import re
import time
import urllib.parse
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime

import polars as pl

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.config import Instrument, TradingConfig
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_HOUR, now_ms

SOURCE = "histdata"
HEADERS = {"User-Agent": "Mozilla/5.0 (qlab research)"}
TOKEN = re.compile(rb'id="tk" value="([0-9a-f]+)"')
MINUTE_COLUMNS = ("stamp", "open", "high", "low", "close", "volume")


def periods(start_year: int, end_ms: int) -> list[tuple[str, str]]:
    """(chemin de la page, ``datemonth``) des périodes complètes : années passées, puis mois
    achevés de l'année en cours."""
    now = datetime.fromtimestamp(end_ms / 1000, tz=UTC)
    out = [(str(y), str(y)) for y in range(start_year, now.year)]
    out += [(f"{now.year}/{m}", f"{now.year}{m:02d}") for m in range(1, now.month)]
    return out


def raw_key(name: str, datemonth: str) -> str:
    return f"{name}/{datemonth}.zip"


def form(token: str, name: str, datemonth: str) -> bytes:
    return urllib.parse.urlencode(
        {
            "tk": token,
            "date": datemonth[:4],
            "datemonth": datemonth,
            "platform": "ASCII",
            "timeframe": "M1",
            "fxpair": name,
        }
    ).encode()


def fetch(cfg: TradingConfig, name: str, page: str, datemonth: str) -> bytes:
    """Zip d'une période ; ``b""`` si la source n'a rien pour elle."""
    page_url = f"{cfg.page_url}{name.lower()}/{page}"
    found = TOKEN.search(http.get(page_url, headers=HEADERS).body)
    if found is None:
        return b""
    headers = {**HEADERS, "Referer": page_url}
    body = http.get(
        cfg.download_url, headers=headers, data=form(found[1].decode(), name, datemonth)
    ).body
    if body and not zipfile.is_zipfile(io.BytesIO(body)):
        raise DataError(f"{name} {datemonth} : la source n'a pas renvoyé un zip")
    return body


def hourly(blob: bytes, offset_h: int) -> pl.DataFrame:
    """Zip de la source → barres horaires UTC (``t_ms``, prix, ``volume`` = minutes)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise DataError(f"zip de la source : un CSV attendu, trouvé {names}")
        raw = z.read(names[0])
    minutes = pl.read_csv(
        raw,
        separator=";",
        has_header=False,
        new_columns=list(MINUTE_COLUMNS),
        schema_overrides={"stamp": pl.String},
    )
    stamp = pl.col("stamp").str.to_datetime("%Y%m%d %H%M%S", time_zone="UTC")
    t_ms = stamp.dt.epoch("ms") + offset_h * MS_PER_HOUR
    return (
        minutes.with_columns((t_ms // MS_PER_HOUR * MS_PER_HOUR).alias("t_ms"))
        .group_by("t_ms", maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.len().cast(pl.Float64).alias("volume"),
        )
    )


def build(
    paths: DataPaths, cfg: TradingConfig, instrument: Instrument, end_ms: int
) -> pl.DataFrame:
    """Barres horaires d'un instrument à partir des zips présents (non vides)."""
    parts = [
        hourly(f.read_bytes(), cfg.source_utc_offset_h)
        for _, datemonth in periods(cfg.start_year, end_ms)
        if (f := paths.raw_archive(SOURCE, raw_key(instrument.name, datemonth))).exists()
        and f.stat().st_size
    ]
    if not parts:
        raise DataError(f"{instrument.name} : aucun fichier de la source (lancer download)")
    return (
        pl.concat(parts)
        .unique("t_ms", keep="last")
        .sort("t_ms")
        .with_columns(pl.lit(instrument.spread_frac).alias("spread_frac"))
        .select("t_ms", "open", "high", "low", "close", "spread_frac", "volume")
    )


def _chosen(ctx: Context) -> list[Instrument]:
    wanted = ctx.args.only.split(",") if ctx.args.only else None
    return [i for i in ctx.config.trading.instruments if wanted is None or i.name in wanted]


def _download(ctx: Context) -> int:
    cfg = ctx.config.trading
    todo = [
        (i.name, page, datemonth)
        for i in _chosen(ctx)
        for page, datemonth in periods(cfg.start_year, now_ms())
        if not ctx.paths.raw_archive(SOURCE, raw_key(i.name, datemonth)).exists()
    ]
    print(f"{len(todo)} fichiers à télécharger", flush=True)
    for n, (name, page, datemonth) in enumerate(todo, 1):
        body = fetch(cfg, name, page, datemonth)
        write_atomic(ctx.paths.raw_archive(SOURCE, raw_key(name, datemonth)), body)
        size = f"{len(body) / 1e6:.1f} Mo" if body else "absent chez la source"
        print(f"{n}/{len(todo)} {name} {datemonth} : {size}", flush=True)
        time.sleep(cfg.request_interval_s)
    ctx.journal.info("histdata.download", {"files": len(todo)})
    return 0


def _build(ctx: Context) -> int:
    cfg = ctx.config.trading
    for inst in _chosen(ctx):
        bars = build(ctx.paths, cfg, inst, now_ms())
        buffer = io.BytesIO()
        bars.write_parquet(buffer, compression="zstd")
        write_atomic(ctx.paths.trading_hourly(inst.name), buffer.getvalue(), overwrite=True)
        first = datetime.fromtimestamp(int(bars["t_ms"][0]) / 1000, tz=UTC).date()
        print(f"{inst.name:8} depuis {first} : {bars.height} heures", flush=True)
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, text in (("download", "copier les zips de la source"), ("build", "barres horaires")):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--only", help="instruments, ex. EURUSD,SPXUSD")


def _action(ctx: Context) -> int:
    return _download(ctx) if ctx.args.cmd == "download" else _build(ctx)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.data.histdata",
        description="Barres d'une minute HistData → barres horaires du volet trading",
        component="histdata",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
