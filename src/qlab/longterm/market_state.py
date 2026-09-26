"""État du marché, jour par jour, calculé sur l'univers observé point-in-time.

Indicateurs par paire, sur **tout** son historique et avec les seules bougies passées (jour
compris), puis agrégés sur les membres de l'univers de chaque date (``universe.observed``) :

- **largeur** ``breadth_{L}`` : part des actifs dont la clôture dépasse leur moyenne mobile sur
  ``L`` jours (fiche ``lt_market_breadth``). Moyenne calculée seulement si les ``L`` jours sont
  tous présents, sinon l'actif ne compte pas ce jour-là (ni au numérateur ni au dénominateur) ;
- **dispersion** : écart-type transversal des rendements journaliers ln(C_t / C_{t−1}) (le jour
  précédent doit exister, sinon rendement absent) ;
- **part de BTC** dans le volume en dollar des membres ;
- ``n_assets`` : nombre de membres.

Les longueurs ``L`` sont des paramètres de fiche (essais journalisés), pas de la config.
Corrélations moyennes et financement : à écrire quand une fiche les utilisera (le financement
aura son propre module de données).

Commande : ``python -m qlab.longterm.market_state --config config show --ma-days 50,100``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY
from qlab.exchange.snapshots import SnapshotStore
from qlab.longterm import klines, universe

BTC = "BTC"


def pair_indicators(bars: pl.DataFrame, ma_days: Sequence[int]) -> pl.DataFrame:
    """``open_time_ms``, ``ret_1d``, ``above_ma_{L}`` (booléen ou nul), ``volume_quote``."""
    if not ma_days or any(n < 1 for n in ma_days):
        raise DataError("ma_days : longueurs ≥ 1 jour attendues")
    df = bars.sort("open_time_ms")
    at = pl.from_epoch("open_time_ms", time_unit="ms")
    consecutive = pl.col("open_time_ms").diff() == MS_PER_DAY
    columns = [
        pl.col("open_time_ms"),
        pl.when(consecutive).then(pl.col("close").log().diff()).alias("ret_1d"),
        pl.col("volume_quote"),
    ]
    for n in ma_days:
        window = f"{n}d"
        mean = pl.col("close").rolling_mean_by(at, window_size=window)
        count = pl.col("close").is_not_null().cast(pl.Int32).rolling_sum_by(at, window_size=window)
        columns.append(pl.when(count >= n).then(pl.col("close") > mean).alias(f"above_ma_{n}"))
    return df.select(columns)


def market_state(paths: DataPaths, members: pl.DataFrame, ma_days: Sequence[int]) -> pl.DataFrame:
    """Une ligne par date de ``members`` (``date_ms``, ``base``, ``symbol``)."""
    frames = [
        pair_indicators(klines.load(paths, universe.INTERVAL, s), ma_days).with_columns(
            symbol=pl.lit(s)
        )
        for s in members["symbol"].unique().sort()
    ]
    if not frames:
        raise DataError("univers vide : aucun membre")
    joined = members.join(
        pl.concat(frames).rename({"open_time_ms": "date_ms"}), on=["date_ms", "symbol"]
    )
    btc_volume = pl.when(pl.col("base") == BTC).then(pl.col("volume_quote")).otherwise(0.0)
    return (
        joined.group_by("date_ms")
        .agg(
            pl.len().alias("n_assets"),
            *(pl.col(f"above_ma_{n}").mean().alias(f"breadth_{n}") for n in ma_days),
            pl.col("ret_1d").std().alias("dispersion"),
            (btc_volume.sum() / pl.col("volume_quote").sum()).alias("btc_volume_share"),
        )
        .sort("date_ms")
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="résumé annuel de l'état du marché")
    show.add_argument("--ma-days", default="50,100", help="longueurs de moyenne, ex. 50,100")
    show.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _action(ctx: Context) -> int:
    ma_days = [int(x) for x in ctx.args.ma_days.split(",")]
    snapshot = SnapshotStore(ctx.paths, ctx.config.base.exchange(ctx.args.exchange).name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    obs = universe.observed(ctx.paths, snapshot.by_symbol(), ctx.config.longterm.universe)
    state = market_state(ctx.paths, obs.members, ma_days)
    year = pl.from_epoch("date_ms", time_unit="ms").dt.year().alias("année")
    summary = state.group_by(year).agg(pl.exclude("date_ms").mean().round(3)).sort("année")
    print(f"Moyennes annuelles de l'état du marché ({state.height} jours) :")
    print(summary)
    ctx.journal.info("market_state.show", {"days": state.height, "ma_days": list(ma_days)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.market_state",
        description="État du marché sur l'univers observé (largeur, dispersion, part de BTC)",
        component="market_state",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
