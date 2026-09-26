"""Spreads réels : relevés du meilleur prix acheteur / vendeur (``bookTicker``) des paires.

Le coût d'un ordre compte ½ spread (LT.4). Sans mesure, le gate de coûts prend une hypothèse
prudente de la config ; sur de petites paires EUR, le vrai spread peut être bien plus large.
Binance ne publie pas l'historique des carnets : on relève **maintenant**, régulièrement
(minuteur horaire, ``ops/``), et la médiane des relevés sert d'estimation pour tout
l'historique (limite écrite dans les rapports).

- ``sample`` : un relevé (une requête publique, poids 4) des paires cotées dans la devise du
  compte (``symbols.quote_asset``) et des paires tradées ; spread relatif
  (ask − bid) / milieu ; carnet vide (bid ou ask nul) ignoré ;
- ``medians`` : médiane par paire sur les relevés, seulement avec au moins ``min_samples``
  relevés (sinon la paire garde l'hypothèse de la config).

Commande : ``python -m qlab.exchange.spreads --config config sample | show``
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.errors import ExchangeError
from qlab.core.paths import DataPaths
from qlab.core.records import append_record, read_records
from qlab.core.timeutils import now_ms
from qlab.exchange.snapshots import SnapshotStore

ENDPOINT = "/api/v3/ticker/bookTicker"


def spreads_from(book: Iterable[Mapping[str, Any]], symbols: set[str]) -> dict[str, float]:
    """Spread relatif de chaque paire de ``symbols`` présente dans la réponse ``bookTicker``."""
    out = {}
    for row in book:
        symbol = row.get("symbol")
        if symbol not in symbols:
            continue
        try:
            bid, ask = Decimal(str(row["bidPrice"])), Decimal(str(row["askPrice"]))
        except (KeyError, InvalidOperation) as exc:
            raise ExchangeError(f"bookTicker illisible pour {symbol}") from exc
        if bid <= 0 or ask < bid:
            continue  # carnet vide ou incohérent : pas de mesure
        out[str(symbol)] = float((ask - bid) / ((ask + bid) / 2))
    return out


def fetch(url: str, *, notify: Callable[[str], None], **http_options: Any) -> list[dict[str, Any]]:
    """Réponse ``bookTicker`` de toutes les paires (``http_options`` : doublures de test)."""
    body = http.get(url, notify=notify, **http_options).body
    try:
        book = json.loads(body)
    except ValueError as exc:
        raise ExchangeError("bookTicker : réponse illisible") from exc
    if not isinstance(book, list):
        raise ExchangeError("bookTicker : liste attendue")
    return book


def medians(paths: DataPaths, min_samples: int) -> dict[str, float]:
    """Médiane des spreads relevés, par paire ayant au moins ``min_samples`` relevés."""
    series: dict[str, list[float]] = {}
    for record in read_records(paths.spreads):
        for symbol, value in record["spreads"].items():
            series.setdefault(symbol, []).append(float(value))
    return {s: statistics.median(v) for s, v in series.items() if len(v) >= min_samples}


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample", help="un relevé des spreads (minuteur horaire)")
    show = sub.add_parser("show", help="médiane par paire")
    show.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _action(ctx: Context) -> int:
    base = ctx.config.base
    exchange = base.exchange(ctx.args.exchange)
    if ctx.args.cmd == "show":
        found = medians(ctx.paths, ctx.args.min_samples)
        for symbol, value in sorted(found.items(), key=lambda kv: kv[1]):
            print(f"{symbol:14s} {value:.4%}")
        print(f"{len(found)} paires avec au moins {ctx.args.min_samples} relevés")
        return 0
    snapshot = SnapshotStore(ctx.paths, exchange.name).latest()
    if snapshot is None:
        raise ExchangeError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    quoted = {
        s["symbol"] for s in snapshot.symbols() if s["quoteAsset"] == base.symbols.quote_asset
    }
    wanted = quoted | set(base.symbols.trade)
    measured = spreads_from(fetch(exchange.rest_url + ENDPOINT, notify=ctx.notify), wanted)
    append_record(ctx.paths.spreads, {"ts_ms": now_ms(), "spreads": measured})
    print(f"{len(measured)} spreads relevés sur {len(wanted)} paires")
    ctx.journal.info("spreads.sample", {"pairs": len(measured)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.exchange.spreads",
        description="Relevés des spreads réels (bookTicker)",
        component="spreads",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
