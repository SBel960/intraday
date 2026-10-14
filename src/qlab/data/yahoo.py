"""Séries quotidiennes des marchés du volet multi-actifs (Yahoo Finance, graphique quotidien).

Accès public, sans clé, **non documenté** : une réponse de forme inattendue arrête la commande
avec une erreur claire, rien n'est deviné. Une requête par marché, toute l'histoire depuis
``markets.start_date`` ; le fichier est réécrit en entier (quelques centaines de Ko).

Prix **ajustés** : open, high, low et close sont multipliés par ``adjclose / close`` du jour
(dividendes, divisions) ; le rendement d'un ETF est ainsi celui d'un porteur qui réinvestit.
Un jour sans clôture est retiré (le panel le marque comme trou). ``t_ms`` = minuit UTC du jour
de cotation (l'heure de la source est celle de l'ouverture locale).

Commande : ``python -m qlab.data.yahoo --config config fetch [--only SPY,TLT]``
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.parse
from collections.abc import Sequence
from typing import Any

import polars as pl

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.config import MarketsConfig
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.timeutils import MS_PER_DAY, date_str, date_to_ms, now_ms

HEADERS = {"User-Agent": "Mozilla/5.0 (qlab research)"}  # la source refuse un client anonyme
COLUMNS = ("t_ms", "open", "high", "low", "close", "volume")


def chart_url(cfg: MarketsConfig, ticker: str, end_ms: int) -> str:
    query = urllib.parse.urlencode(
        {
            "period1": date_to_ms(cfg.start_date) // 1000,
            "period2": end_ms // 1000,
            "interval": "1d",
        }
    )
    return f"{cfg.chart_url}{urllib.parse.quote(ticker)}?{query}"


def _result(blob: Any, ticker: str) -> Any:
    chart = blob.get("chart") if isinstance(blob, dict) else None
    if not isinstance(chart, dict) or chart.get("error"):
        detail = chart.get("error") if isinstance(chart, dict) else blob
        raise DataError(f"{ticker} : réponse refusée par la source ({detail})")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1:
        raise DataError(f"{ticker} : réponse sans série")
    return results[0]


def parse(body: bytes, ticker: str) -> pl.DataFrame:
    """Réponse JSON → (``t_ms``, ``open``, ``high``, ``low``, ``close``, ``volume``) ajustés,
    un jour par ligne, triés."""
    res = _result(json.loads(body), ticker)
    try:
        quote = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        raw = pl.DataFrame(
            {
                "t_ms": [int(t) * 1000 // MS_PER_DAY * MS_PER_DAY for t in res["timestamp"]],
                **{k: quote[k] for k in ("open", "high", "low", "close", "volume")},
                "adjclose": adj,
            },
            schema_overrides={k: pl.Float64 for k in ("open", "high", "low", "close", "volume")}
            | {"adjclose": pl.Float64},
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise DataError(f"{ticker} : forme de réponse inattendue ({exc!r})") from exc
    factor = pl.col("adjclose") / pl.col("close")
    return (
        raw.filter(pl.col("close") > 0, pl.col("adjclose") > 0)
        .with_columns((pl.col(c) * factor).alias(c) for c in ("open", "high", "low", "close"))
        .unique("t_ms", keep="last")
        .sort("t_ms")
        .select(COLUMNS)
    )


def fetch(cfg: MarketsConfig, ticker: str, end_ms: int) -> pl.DataFrame:
    return parse(http.get(chart_url(cfg, ticker, end_ms), headers=HEADERS).body, ticker)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("fetch", help="télécharger les séries de tous les marchés")
    run.add_argument("--only", help="tickers, ex. SPY,TLT")


def _action(ctx: Context) -> int:
    cfg = ctx.config.markets
    tickers = [a.ticker for a in cfg.assets]
    if ctx.args.only:
        tickers = [t for t in tickers if t in ctx.args.only.split(",")]
    end = now_ms()
    for i, ticker in enumerate(tickers):
        if i:
            time.sleep(cfg.request_interval_s)
        bars = fetch(cfg, ticker, end)
        if bars.is_empty():
            raise DataError(f"{ticker} : aucune séance")
        buffer = io.BytesIO()
        bars.write_parquet(buffer, compression="zstd")
        write_atomic(ctx.paths.markets_daily(ticker), buffer.getvalue(), overwrite=True)
        first, last = date_str(int(bars["t_ms"][0])), date_str(int(bars["t_ms"][-1]))
        print(f"{ticker:8} {first} → {last} : {bars.height} séances", flush=True)
    ctx.journal.info("markets.fetch", {"tickers": len(tickers)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.data.yahoo",
        description="Séries quotidiennes ajustées des marchés du volet multi-actifs",
        component="markets_fetch",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
