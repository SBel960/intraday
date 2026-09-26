"""Taux de financement des contrats perpétuels USDⓈ-M : archives → ``lt/futures/funding``.

Un taux de financement positif = les acheteurs à levier paient les vendeurs, à chaque
échéance (toutes les 8 h en général ; 4 h ou 1 h pour certains contrats, et l'intervalle d'un
contrat peut changer : il est lu dans chaque ligne, jamais supposé). Fiche concernée :
``lt_funding_leverage``.

- lecture des archives mensuelles ``fundingRate`` (colonnes ``calc_time``,
  ``funding_interval_hours``, ``last_funding_rate``) ; l'heure de calcul (quelques ms après
  l'échéance) est gardée telle quelle : certains contrats (actions tokenisées) ont deux
  règlements distincts à quelques ms d'écart, qu'un arrondi fusionnerait ;
- contrôle : horodatage plausible (ms, 2000–2100), intervalle > 0, taux fini ; doublons
  identiques retirés, deux taux différents pour la même échéance ⇒ erreur ;
- ``daily`` : somme des taux de chaque jour UTC (coût d'une journée de position acheteuse à
  levier 1) et nombre d'échéances.

Les archives s'arrêtent au dernier mois complet ; le mois en cours viendra du REST (paper).

Commande : ``python -m qlab.longterm.funding --config config build [--symbols A,B]``
"""

from __future__ import annotations

import argparse
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, floor_ms_expr, infer_epoch_unit
from qlab.data import binance_vision as bv
from qlab.data.archives import SOURCE

DATASET = "funding"
SCHEMA = {
    "calc_time": pl.Int64,
    "funding_interval_hours": pl.Int64,
    "last_funding_rate": pl.Float64,
}
COLUMNS = ("funding_time_ms", "interval_hours", "rate")


def parse_csv(data: bytes) -> pl.DataFrame:
    """CSV d'archive (en-tête présent) → ``funding_time_ms``, ``interval_hours``, ``rate``."""
    if not data.strip():
        return pl.DataFrame(schema=dict(zip(COLUMNS, SCHEMA.values(), strict=True)))
    df = pl.read_csv(io.BytesIO(data), schema=SCHEMA)
    for ts in (df["calc_time"].min(), df["calc_time"].max()):
        if not isinstance(ts, int) or _unit(ts) != "ms":
            raise DataError(f"financement : horodatage invalide ({ts!r})")
    if (df["funding_interval_hours"] <= 0).any() or not df["last_funding_rate"].is_finite().all():
        raise DataError("financement : intervalle ≤ 0 ou taux non fini")
    return df.select(
        pl.col("calc_time").alias("funding_time_ms"),
        pl.col("funding_interval_hours").alias("interval_hours"),
        pl.col("last_funding_rate").alias("rate"),
    )


def _unit(ts: int) -> str:
    try:
        return infer_epoch_unit(ts)
    except ValueError:
        return "invalide"


def read_archive(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise DataError(f"{path.name} : un seul fichier attendu dans l'archive, {len(names)}")
        data = z.read(names[0])
    try:
        return parse_csv(data)
    except pl.exceptions.ComputeError as e:
        raise DataError(f"{path.name} : CSV illisible ({str(e).splitlines()[0]})") from e


def merge(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Une ligne par échéance ; deux taux différents pour une même échéance ⇒ ``DataError``."""
    if not frames:
        return parse_csv(b"")
    rows = pl.concat(frames).unique().sort("funding_time_ms")
    clash = rows.filter(pl.col("funding_time_ms").is_duplicated())
    if not clash.is_empty():
        raise DataError(f"financement : {clash.height} lignes contradictoires pour une échéance")
    return rows


def daily(funding: pl.DataFrame) -> pl.DataFrame:
    """``date_ms`` (jour UTC), ``funding_1d`` (somme des taux du jour), ``n_events``."""
    day = floor_ms_expr(pl.col("funding_time_ms"), MS_PER_DAY).alias("date_ms")
    return (
        funding.group_by(day)
        .agg(pl.col("rate").sum().alias("funding_1d"), pl.len().alias("n_events"))
        .sort("date_ms")
    )


def load(paths: DataPaths, symbol: str) -> pl.DataFrame:
    path = paths.lt_futures(DATASET, symbol)
    if not path.exists():
        raise DataError(f"{symbol} : financement absent, lancer « funding build »")
    return pl.read_parquet(path)


def _archive_dir(paths: DataPaths, symbol: str | None = None) -> Path:
    key = bv.UM_FUNDING_MONTHLY.removeprefix("data/").rstrip("/")
    return paths.raw_archive(SOURCE, key if symbol is None else f"{key}/{symbol}")


def build_symbol(paths: DataPaths, symbol: str) -> int:
    """Construit la série d'une paire ; renvoie le nombre d'échéances."""
    archives = sorted(_archive_dir(paths, symbol).glob("*.zip"))
    rows = merge([read_archive(a) for a in archives])
    buffer = io.BytesIO()
    rows.write_parquet(buffer, compression="zstd")
    write_atomic(paths.lt_futures(DATASET, symbol), buffer.getvalue(), overwrite=True)
    return rows.height


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="construire lt/futures/funding depuis les archives")
    build.add_argument("--symbols", help="contrats, ex. BTCUSDT ; défaut : tous")


def _action(ctx: Context) -> int:
    root = _archive_dir(ctx.paths)
    if ctx.args.symbols is not None:
        symbols = ctx.args.symbols.split(",")
    else:
        symbols = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    events, failed = 0, []
    for symbol in symbols:
        try:
            events += build_symbol(ctx.paths, symbol)
        except DataError as e:  # une archive fautive arrête son contrat, pas les autres
            failed.append(f"{symbol} : {e}")
    print(f"Financement : {len(symbols) - len(failed)} contrats, {events} échéances")
    for line in failed:
        print(f"ERREUR {line}")
    ctx.journal.info(
        "funding.build", {"symbols": len(symbols), "events": events, "errors": "; ".join(failed)}
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.funding",
        description="Taux de financement USDⓈ-M depuis les archives Binance",
        component="funding",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
