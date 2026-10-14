"""Paper en temps réel des annonces Binance (phase 9) : aucun ordre, trades simulés au vrai prix.

Seules comptent les annonces **reçues** par ``events/news.py poll`` (``received_ms``) depuis
``events.paper.start_date`` : rien de l'historique, rien de connu d'avance. Règles figées le
2026-10-14 (``events.paper.rules``) : un retrait ⇒ vente à découvert du contrat perpétuel
``{actif}USDT`` pendant 72 h ; une cotation ⇒ achat pendant 24 h. Chaque passage
(``run``, minuteur ``ops/qlab-events``) :

1. **signal** : annonce nouvelle ⇒ entrée prévue à l'heure pleine qui suit sa réception ;
   écartée (et dite pourquoi) si le contrat n'existe pas, fait moins de ``min_volume_quote``
   sur 24 h, ou si ``max_lines`` positions sont déjà prises ;
2. **clôture** : après ``hold_h`` heures, entrée = ouverture de la bougie horaire d'entrée,
   sortie = ouverture de la bougie de sortie (ou dernière clôture si le contrat a été retiré
   entre-temps), financement réglé pendant la détention, liquidation à 1× si le plus haut
   double l'entrée (vente) ; coût ``2 × leg_cost_frac``. Les prix sont relus des bougies
   publiques après coup : aucun prix n'est figé au moment du signal.

Tout est écrit dans ``meta/paper/events.jsonl`` (ajout seulement) ; l'état se relit du journal.
Commandes : ``python -m qlab.live.paper_events --config config run`` ; ``… status``
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.config import EventPaperConfig
from qlab.core.errors import ExchangeError
from qlab.core.records import Record, append_record, read_records
from qlab.core.timeutils import MS_PER_HOUR, date_str, date_to_ms, now_ms
from qlab.events import classify
from qlab.events.news import stored

JOURNAL = "events"
SETTLE_MS = 60_000  # une bougie horaire est lue une minute après son ouverture


class Market(Protocol):
    def volume_24h(self, symbol: str) -> float | None: ...
    def hour_bars(self, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]: ...
    def funding(self, symbol: str, start_ms: int, end_ms: int) -> float: ...


@dataclass(frozen=True, slots=True)
class BinanceFutures:
    """Données publiques des contrats USDⓈ-M (``timing.futures_rest_url``)."""

    base_url: str

    def _get(self, path: str, **query: object) -> Any:
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(query)}"
        return json.loads(http.get(url).body)

    def volume_24h(self, symbol: str) -> float | None:
        try:
            return float(self._get("/fapi/v1/ticker/24hr", symbol=symbol)["quoteVolume"])
        except ExchangeError:  # HTTP 400 : contrat inconnu
            return None

    def hour_bars(self, symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
        query = {"symbol": symbol, "interval": "1h", "startTime": start_ms, "endTime": end_ms}
        return list(self._get("/fapi/v1/klines", **query, limit=1000))

    def funding(self, symbol: str, start_ms: int, end_ms: int) -> float:
        rows = self._get(
            "/fapi/v1/fundingRate",
            symbol=symbol,
            startTime=start_ms,
            endTime=end_ms - 1,
            limit=1000,
        )
        return sum(float(r["fundingRate"]) for r in rows)


def new_signals(cfg: EventPaperConfig, news: list[Record], journal: list[Record]) -> pl.DataFrame:
    """Annonces reçues depuis le début du paper, d'un type suivi, pas encore au journal."""
    live = [r for r in news if r["received_ms"] is not None]
    table = classify.events_table(live, 0).filter(
        pl.col("effective_ms") >= date_to_ms(cfg.start_date),
        pl.col("kind").is_in([r.kind for r in cfg.rules]),
    )
    seen = {(j["kind"], j["base"], j["received_ms"]) for j in journal if j["type"] == "signal"}
    keep = [
        (k, b, e) not in seen for k, b, e in table.select("kind", "base", "effective_ms").rows()
    ]
    return table.filter(pl.Series(keep, dtype=pl.Boolean))


def _open_ids(journal: list[Record]) -> set[int]:
    pending = {j["id"] for j in journal if j["type"] == "signal" and j["decision"] == "pending"}
    return pending - {j["id"] for j in journal if j["type"] == "close"}


def decide(cfg: EventPaperConfig, row: dict[str, Any], market: Market, busy: int) -> Record:
    rule = next(r for r in cfg.rules if r.kind == row["kind"])
    entry = -(-row["effective_ms"] // MS_PER_HOUR) * MS_PER_HOUR
    symbol = f"{row['base']}{cfg.quote}"
    volume = market.volume_24h(symbol)
    reason = (
        "aucun contrat perpétuel"
        if volume is None
        else f"illiquide ({volume:,.0f} {cfg.quote} sur 24 h)"
        if volume < cfg.min_volume_quote
        else "lignes pleines"
        if busy >= cfg.max_lines
        else ""
    )
    return {
        "type": "signal",
        "kind": row["kind"],
        "base": row["base"],
        "symbol": symbol,
        "received_ms": row["effective_ms"],
        "side": rule.side,
        "entry_ms": entry,
        "exit_ms": entry + rule.hold_h * MS_PER_HOUR,
        "hypothesis": f"{rule.hypothesis} {rule.params}",
        "decision": "skipped" if reason else "pending",
        "reason": reason,
    }


def settle(cfg: EventPaperConfig, signal: Record, market: Market) -> Record:
    """Clôture d'une position arrivée à échéance (voir le module, étape 3)."""
    bars = market.hour_bars(signal["symbol"], signal["entry_ms"], signal["exit_ms"])
    held = [b for b in bars if b[0] < signal["exit_ms"]]
    entry = float(held[0][1])
    exit_bar = [b for b in bars if b[0] == signal["exit_ms"]]
    exit_price = float(exit_bar[0][1]) if exit_bar else float(held[-1][4])
    peak = max(float(b[2]) for b in held)
    fund = market.funding(signal["symbol"], signal["entry_ms"], signal["exit_ms"])
    side = signal["side"]
    liquidated = side < 0 and peak >= 2 * entry
    pnl = -1.0 if liquidated else max(side * (exit_price / entry - 1) - side * fund, -1.0)
    stake = cfg.capital_quote / cfg.max_lines
    cost = 2 * cfg.leg_cost_frac
    return {
        "type": "close",
        "id": signal["id"],
        "entry_price": entry,
        "exit_price": exit_price,
        "peak": peak,
        "funding": fund,
        "liquidated": liquidated,
        "return_frac": pnl - cost,
        "net_quote": stake * (pnl - cost),
    }


def step(
    cfg: EventPaperConfig, news: list[Record], journal: list[Record], market: Market, now: int
) -> list[Record]:
    """Enregistrements à ajouter au journal pour ce passage."""
    out: list[Record] = []
    busy = len(_open_ids(journal))
    next_id = 1 + max((j["id"] for j in journal if j["type"] == "signal"), default=0)
    for row in new_signals(cfg, news, journal).iter_rows(named=True):
        rec = {**decide(cfg, row, market, busy), "id": next_id}
        busy += rec["decision"] == "pending"
        next_id += 1
        out.append(rec)
    signals = {j["id"]: j for j in journal if j["type"] == "signal"}
    for i in sorted(_open_ids(journal)):
        if now >= signals[i]["exit_ms"] + SETTLE_MS:
            out.append(settle(cfg, signals[i], market))
    return out


def _run(ctx: Context) -> int:
    cfg = ctx.config.events.paper
    path = ctx.paths.paper_journal(JOURNAL)
    market = BinanceFutures(ctx.config.timing.futures_rest_url)
    added = step(cfg, stored(ctx.paths), read_records(path), market, now_ms())
    for rec in added:
        append_record(path, rec)
    ctx.journal.info("paper_events.run", {"added": len(added)})
    return 0


def _status(ctx: Context) -> int:
    journal = read_records(ctx.paths.paper_journal(JOURNAL))
    closes = {j["id"]: j for j in journal if j["type"] == "close"}
    for s in (j for j in journal if j["type"] == "signal"):
        c = closes.get(s["id"])
        state = (
            s["reason"]
            if s["decision"] == "skipped"
            else (
                f"{c['return_frac']:+.1%} ({c['net_quote']:+.2f} {ctx.config.events.paper.quote})"
                if c
                else "en cours"
            )
        )
        print(f"{date_str(s['received_ms'])} {s['kind']:9} {s['symbol']:14} {state}")
    total = sum(c["net_quote"] for c in closes.values())
    print(f"{len(closes)} trade(s) clos, résultat {total:+.2f} {ctx.config.events.paper.quote}")
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="un passage : nouveaux signaux, clôtures échues")
    sub.add_parser("status", help="trades du paper et résultat")


def _action(ctx: Context) -> int:
    return _run(ctx) if ctx.args.cmd == "run" else _status(ctx)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.live.paper_events",
        description="Paper en temps réel des annonces Binance (aucun ordre)",
        component="paper_events",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
