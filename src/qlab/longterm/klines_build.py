"""Construction de ``lt/klines_{intervalle}/{paire}.parquet`` depuis ``raw/binance_vision``.

Toutes les paires présentes sur disque (retirées comprises), chaque intervalle de
``observe.kline_intervals`` ; format et contrôle qualité : ``longterm/klines.py``. Une paire déjà
construite n'est refaite que si une archive est plus récente que sa sortie (ou ``--force``,
après un changement du format). Une archive
illisible arrête sa paire, pas les autres : l'erreur est rapportée et le code de sortie vaut 1.

Commande : ``python -m qlab.longterm.klines_build --config config build [--symbols A,B]
[--intervals 1d,1h]``
"""

from __future__ import annotations

import argparse
import io
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.jsonlog import JsonValue
from qlab.core.paths import DataPaths
from qlab.data import binance_vision as bv
from qlab.data.archives import SOURCE
from qlab.longterm.klines import gaps, interval_to_ms, merge, read_archive


def _archives(paths: DataPaths, symbol: str, interval: str) -> list[Path]:
    """Mensuelles triées, puis quotidiennes triées (ordre de priorité)."""
    found = []
    for prefix in (bv.SPOT_KLINES_MONTHLY, bv.SPOT_KLINES_DAILY):
        key = f"{prefix.removeprefix('data/')}{symbol}/{interval}"
        folder = paths.raw_archive(SOURCE, key)
        found += sorted(folder.glob("*.zip")) if folder.is_dir() else []
    return found


def symbols_on_disk(paths: DataPaths) -> list[str]:
    names: set[str] = set()
    for prefix in (bv.SPOT_KLINES_MONTHLY, bv.SPOT_KLINES_DAILY):
        folder = paths.raw_archive(SOURCE, prefix.removeprefix("data/").rstrip("/"))
        names |= {p.name for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()
    return sorted(names)


@dataclass(frozen=True, slots=True)
class BuildResult:
    symbol: str
    interval: str
    status: str  # "construit", "à jour", "vide", "erreur"
    bars: int = 0
    gaps: int = 0
    missing: int = 0
    anomalies: int = 0
    conflicts: int = 0
    error: str = ""


def build_symbol(
    paths: DataPaths, symbol: str, interval: str, *, force: bool = False
) -> BuildResult:
    step = interval_to_ms(interval)
    out = paths.lt_klines(interval, symbol)
    archives = _archives(paths, symbol, interval)
    if not archives:
        return BuildResult(symbol, interval, "vide")
    newest = max(a.stat().st_mtime for a in archives)
    if not force and out.exists() and out.stat().st_mtime >= newest:
        return BuildResult(symbol, interval, "à jour")
    merged = merge([read_archive(a) for a in archives], step)
    holes = gaps(merged.bars["open_time_ms"], step)
    buffer = io.BytesIO()
    merged.bars.write_parquet(buffer, compression="zstd")
    write_atomic(out, buffer.getvalue(), overwrite=True)
    return BuildResult(
        symbol,
        interval,
        "construit",
        merged.bars.height,
        holes.height,
        int(holes["missing"].sum()),
        merged.anomalies,
        merged.conflicts,
    )


def _build_or_report(paths: DataPaths, symbol: str, interval: str, force: bool) -> BuildResult:
    """Une archive illisible arrête sa paire, pas les milliers d'autres : erreur rapportée."""
    try:
        return build_symbol(paths, symbol, interval, force=force)
    except DataError as e:
        return BuildResult(symbol, interval, "erreur", error=str(e))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="construire lt/klines_* depuis les archives")
    build.add_argument("--symbols", help="paires, ex. BTCEUR,ETHEUR ; défaut : toutes")
    build.add_argument("--intervals", help="ex. 1d ; défaut : observe.kline_intervals")
    build.add_argument("--force", action="store_true", help="reconstruire même les paires à jour")


def _summary(done: list[BuildResult]) -> dict[str, JsonValue]:
    built = [r for r in done if r.status == "construit"]
    return {
        "paires": len(done),
        "construites": len(built),
        "a_jour": sum(r.status == "à jour" for r in done),
        "vides": sum(r.status == "vide" for r in done),
        "erreurs": sum(r.status == "erreur" for r in done),
        "bougies": sum(r.bars for r in built),
        "trous": sum(r.gaps for r in built),
        "bougies_manquantes": sum(r.missing for r in built),
        "anomalies": sum(r.anomalies for r in built),
        "conflits": sum(r.conflicts for r in built),
        "suspects": {
            r.symbol: [r.anomalies, r.conflicts] for r in built if r.anomalies or r.conflicts
        },
    }


def _action(ctx: Context) -> int:
    args = ctx.args
    symbols = symbols_on_disk(ctx.paths) if args.symbols is None else args.symbols.split(",")
    intervals = (
        list(ctx.config.base.observe.kline_intervals)
        if args.intervals is None
        else args.intervals.split(",")
    )
    tasks = [(s, i) for i in intervals for s in symbols]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = list(
            pool.map(
                _build_or_report, repeat(ctx.paths), *zip(*tasks, strict=True), repeat(args.force)
            )
        )
    for interval in intervals:
        summary = _summary([r for r in results if r.interval == interval])
        shown = {k: v for k, v in summary.items() if k != "suspects"}
        print(f"{interval} : " + ", ".join(f"{k} {v}" for k, v in shown.items()))
        ctx.journal.info("klines.build", {"interval": interval, **summary})
    failed = [r for r in results if r.status == "erreur"]
    for r in failed:
        print(f"ERREUR {r.symbol} {r.interval} : {r.error}")
        ctx.journal.error(
            "klines.error", {"symbol": r.symbol, "interval": r.interval, "error": r.error}
        )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.klines_build",
        description="Bougies long terme depuis les archives Binance",
        component="klines_build",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
