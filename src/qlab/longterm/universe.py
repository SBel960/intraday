"""Univers point-in-time : quelles paires existent, et sont éligibles, à chaque date.

Une date ``d`` = ouverture (``open_time_ms``) d'une bougie 1d ; la décision a lieu après sa
clôture et n'utilise que les bougies ≤ ``d``. Une paire est présente les jours où elle a une
bougie : une paire retirée sort d'elle-même après sa dernière bougie (pas de biais du
survivant), et son historique reste.

- **Univers tradé** (``symbols.trade``) : paire présente et chauffe écoulée (``warmup_days``
  après sa première bougie). Pas de filtre de volume : ce qu'on trade est choisi, son coût est
  mesuré par le cost gate.
- **Univers observé** (tout le marché, pour le momentum transversal et la largeur) : un actif =
  une paire cotée en dollar ; chaque jour, parmi les paires de l'actif éligibles ce jour-là,
  celle de la devise la plus prioritaire de ``reference_quotes`` (ex. BUSD retiré en 2023 ⇒
  USDT). Exclus : stablecoins et devises (``excluded_bases``), tokens à levier (base = autre
  base + suffixe, ex. BTCUP ; JUP gardé). Éligible : chauffe écoulée **et** volume médian
  des ``volume_lookback_days`` derniers jours (bougie du jour comprise, jamais après)
  ≥ ``min_volume_quote``.

Base et devise d'une paire viennent du dernier snapshot exchangeInfo (elles ne changent jamais,
aucune fuite d'information) ; une paire absente du snapshot est reconnue par le suffixe de sa
devise, sinon ignorée et comptée.

Commande : ``python -m qlab.longterm.universe --config config show [--date AAAA-MM-JJ]``
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.config import UniverseConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_str, date_to_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import klines

INTERVAL = "1d"


@dataclass(frozen=True, slots=True)
class Pair:
    symbol: str
    base: str
    quote: str


def pair_of(
    symbol: str, known: Mapping[str, Mapping[str, Any]], quotes: Sequence[str]
) -> Pair | None:
    """Base et devise d'une paire : exchangeInfo, sinon suffixe d'une devise de ``quotes``."""
    if symbol in known:
        return Pair(symbol, str(known[symbol]["baseAsset"]), str(known[symbol]["quoteAsset"]))
    for quote in sorted(quotes, key=len, reverse=True):
        if symbol.endswith(quote) and symbol != quote:
            return Pair(symbol, symbol.removesuffix(quote), quote)
    return None


def is_asset(base: str, all_bases: Collection[str], cfg: UniverseConfig) -> bool:
    """Faux pour un stablecoin, une devise ou un token à levier (BTCUP = BTC + UP)."""
    if base in cfg.excluded_bases:
        return False
    return not any(
        base.endswith(suffix) and base.removesuffix(suffix) in all_bases
        for suffix in cfg.leveraged_suffixes
    )


def eligible_dates(
    bars: pl.DataFrame,
    *,
    warmup_days: int,
    lookback_days: int | None = None,
    min_volume: float | None = None,
) -> pl.Series:
    """Dates (``open_time_ms``) où la paire est éligible : chauffe écoulée et, si demandé,
    volume médian des ``lookback_days`` derniers jours (jour compris) ≥ ``min_volume``."""
    if bars.is_empty():
        return pl.Series("date_ms", [], dtype=pl.Int64)
    df = bars.sort("open_time_ms")
    ok = pl.col("open_time_ms") >= df["open_time_ms"][0] + warmup_days * MS_PER_DAY
    if lookback_days is not None and min_volume is not None:
        at = pl.from_epoch("open_time_ms", time_unit="ms")
        df = df.with_columns(
            median=pl.col("volume_quote").rolling_median_by(at, window_size=f"{lookback_days}d")
        )
        ok &= pl.col("median") >= min_volume
    return df.filter(ok)["open_time_ms"].rename("date_ms")


def traded(paths: DataPaths, symbols: Sequence[str], warmup_days: int) -> pl.DataFrame:
    """Univers tradé : ``date_ms``, ``symbol`` (paire présente, chauffe écoulée)."""
    frames = [
        pl.DataFrame({"date_ms": eligible_dates(bars, warmup_days=warmup_days)}).with_columns(
            symbol=pl.lit(s)
        )
        for s in symbols
        for bars in [klines.load(paths, INTERVAL, s)]
    ]
    return pl.concat(frames).sort("date_ms", "symbol")


@dataclass(frozen=True, slots=True)
class Observed:
    """Univers observé : ``members`` (``date_ms``, ``base``, ``symbol``), paires considérées,
    paires ignorées faute de base/devise connues."""

    members: pl.DataFrame
    candidates: int
    unknown: int


def observed(
    paths: DataPaths, known: Mapping[str, Mapping[str, Any]], cfg: UniverseConfig
) -> Observed:
    symbols = [p.stem for p in sorted(paths.lt_klines_dir(INTERVAL).glob("*.parquet"))]
    pairs = [pair_of(s, known, cfg.reference_quotes) for s in symbols]
    bases = {p.base for p in pairs if p is not None}
    chosen = [
        p for p in pairs if p and p.quote in cfg.reference_quotes and is_asset(p.base, bases, cfg)
    ]
    priority = {q: i for i, q in enumerate(cfg.reference_quotes)}
    frames = []
    for p in chosen:
        bars = pl.read_parquet(
            paths.lt_klines(INTERVAL, p.symbol), columns=["open_time_ms", "volume_quote"]
        )
        dates = eligible_dates(
            bars,
            warmup_days=cfg.warmup_days,
            lookback_days=cfg.volume_lookback_days,
            min_volume=cfg.min_volume_quote,
        )
        frames.append(
            pl.DataFrame({"date_ms": dates}).with_columns(
                base=pl.lit(p.base), symbol=pl.lit(p.symbol), prio=pl.lit(priority[p.quote])
            )
        )
    schema = {"date_ms": pl.Int64, "base": pl.String, "symbol": pl.String, "prio": pl.Int64}
    everything = pl.concat(frames) if frames else pl.DataFrame(schema=schema)
    members = (
        everything.sort("date_ms", "base", "prio")
        .unique(subset=["date_ms", "base"], keep="first", maintain_order=True)
        .drop("prio")
    )
    return Observed(members, len(chosen), sum(p is None for p in pairs))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="taille de l'univers par année, membres à une date")
    show.add_argument("--date", help="AAAA-MM-JJ : liste des membres ce jour-là")
    show.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _sizes_by_year(members: pl.DataFrame) -> pl.DataFrame:
    """Nombre d'actifs par jour, résumé par année (min, médiane, max)."""
    daily = members.group_by("date_ms").len()
    year = pl.from_epoch("date_ms", time_unit="ms").dt.year().alias("year")
    return (
        daily.group_by(year)
        .agg(low=pl.col("len").min(), mid=pl.col("len").median(), high=pl.col("len").max())
        .sort("year")
    )


def _action(ctx: Context) -> int:
    cfg = ctx.config.longterm.universe
    snapshot = SnapshotStore(ctx.paths, ctx.config.base.exchange(ctx.args.exchange).name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    obs = observed(ctx.paths, snapshot.by_symbol(), cfg)
    print(f"Observé : {obs.candidates} paires en dollar candidates, {obs.unknown} inconnues")
    for y, low, mid, high in _sizes_by_year(obs.members).iter_rows():
        print(f"  {y} : actifs par jour min {low}, médiane {mid:.0f}, max {high}")
    trade = traded(ctx.paths, ctx.config.base.symbols.trade, cfg.warmup_days)
    for sym, first in trade.group_by("symbol").agg(pl.col("date_ms").min()).sort("symbol").rows():
        print(f"Tradé {sym} : éligible dès le {date_str(first)}")
    if ctx.args.date:
        day = obs.members.filter(pl.col("date_ms") == date_to_ms(ctx.args.date))
        print(f"{ctx.args.date} : {day.height} actifs : {' '.join(day['symbol'].to_list())}")
    ctx.journal.info("universe.show", {"candidates": obs.candidates, "unknown": obs.unknown})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.universe",
        description="Univers point-in-time (tradé et observé)",
        component="universe",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
