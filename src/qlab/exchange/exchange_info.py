"""Récupère ``exchangeInfo`` (règles de toutes les paires), mesure l'horloge, versionne, alerte.

``python -m qlab.exchange.exchange_info --config config fetch [--if-due]`` :

1. ``--if-due`` : ne fait rien si le dernier snapshot a moins de ``snapshot_refresh_hours``
   (lancement quotidien automatique, voir ``ops/``) ;
2. télécharge via ``core/http.py`` (attend le retour du réseau, respecte les limites Binance) ;
3. mesure l'écart entre l'horloge locale et l'heure du serveur (``/api/v3/time``, réponse de
   quelques octets) ; au-delà de ``max_clock_offset_ms``, alerte (journal + message) ;
4. enregistre une nouvelle version si le contenu a changé (``exchange/snapshots.py``) et signale
   les différences ; **alerte** si une paire tradée (``symbols.trade``) est touchée.

``show [--symbol S]`` : résumé du dernier snapshot. Tout est tracé dans ``logs/exchange_info/``.

Frais : sans clé API, Binance ne donne pas ceux du compte ; le snapshot enregistre le barème de
repli de la config (``"origin": "config"``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.config import ExchangeConfig
from qlab.core.errors import DataError, ExchangeError
from qlab.core.timeutils import ms_to_iso, now_ms
from qlab.exchange.fees import account_rates, describe, effective_by_symbol, fees_record
from qlab.exchange.snapshots import (
    Snapshot,
    SnapshotStore,
    diff_snapshots,
    validate_exchange_info,
)

ENDPOINT = "/api/v3/exchangeInfo"
# Mesure d'horloge sur un endpoint minuscule : la réponse exchangeInfo (~17 Mo) met ~2 s à
# arriver et fausserait le point milieu (le serveur date sa réponse au début de l'envoi).
TIME_ENDPOINT = "/api/v3/time"
MS_PER_HOUR = 3_600_000


@dataclass(frozen=True, slots=True)
class ClockOffset:
    """Écart horloge serveur − horloge locale (ms ; négatif = PC en avance), estimé au milieu
    de l'aller-retour ; ``uncertainty_ms`` = demi-aller-retour."""

    offset_ms: int
    uncertainty_ms: int


def clock_offset(server_time_ms: int, sent_ms: int, received_ms: int) -> ClockOffset:
    """Écart d'horloge par la méthode du point milieu (comme NTP, sans correction de symétrie)."""
    if received_ms < sent_ms:
        raise DataError(
            f"horloge locale revenue en arrière pendant la requête ({sent_ms} → {received_ms})"
        )
    return ClockOffset(server_time_ms - (sent_ms + received_ms) // 2, (received_ms - sent_ms) // 2)


def measure_clock(url: str, *, notify: Callable[[str], None], **http_options: Any) -> ClockOffset:
    """Écart d'horloge mesuré sur ``url`` (``/api/v3/time`` : ``{"serverTime": ms}``)."""
    result = http.get(url, notify=notify, **http_options)
    try:
        server_time = json.loads(result.body)["serverTime"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ExchangeError(f"réponse d'heure serveur illisible : {url}") from exc
    if not isinstance(server_time, int) or isinstance(server_time, bool):
        raise ExchangeError(f"serverTime non entier : {url}")
    return clock_offset(server_time, result.sent_ms, result.received_ms)


def fetch_exchange_info(
    url: str, *, notify: Callable[[str], None], **http_options: Any
) -> tuple[dict[str, Any], http.HttpResult]:
    """Télécharge et valide ``exchangeInfo`` ; ``http_options`` passe à ``http.get`` (tests)."""
    result = http.get(url, notify=notify, **http_options)
    try:
        info = json.loads(result.body)
    except ValueError as exc:
        raise ExchangeError(f"réponse non JSON de {url}") from exc
    validate_exchange_info(info)
    return info, result


def is_due(latest: Snapshot | None, now: int, refresh_hours: int) -> bool:
    """Faut-il un nouveau snapshot ? Oui s'il n'y en a pas ou s'il date d'au moins la période."""
    return latest is None or now - latest.fetched_ms >= refresh_hours * MS_PER_HOUR


def _alert(ctx: Context, kind: str, message: str, data: dict[str, Any]) -> None:
    """Alerte : message visible (stderr) et avertissement dans le journal."""
    print(f"ALERTE : {message}", file=sys.stderr, flush=True)
    ctx.journal.warning(kind, {"message": message, **data})


def _summary(snap: Snapshot, versions: int) -> str:
    trading = snap.symbols()
    quotes: dict[str, int] = {}
    for s in trading:
        quotes[s["quoteAsset"]] = quotes.get(s["quoteAsset"], 0) + 1
    top = ", ".join(f"{q} {n}" for q, n in sorted(quotes.items(), key=lambda x: -x[1])[:6])
    head = (
        f"{ms_to_iso(snap.fetched_ms)} | {len(snap.symbols(None))} paires, {len(trading)} en "
        f"cotation ({top}) | {versions} version(s)"
    )
    return "\n".join([head, *describe(snap.fees)])


def _check_clock(ctx: Context, exchange: ExchangeConfig, clock: ClockOffset) -> None:
    ctx.journal.info(
        "clock.offset", {"offset_ms": clock.offset_ms, "uncertainty_ms": clock.uncertainty_ms}
    )
    # Alerte seulement si l'écart dépasse le seuil même dans le cas le plus favorable.
    if abs(clock.offset_ms) - clock.uncertainty_ms > exchange.max_clock_offset_ms:
        _alert(
            ctx,
            "clock.drift",
            f"horloge locale décalée de {-clock.offset_ms:+d} ms (± {clock.uncertainty_ms}) "
            f"par rapport à {exchange.name}, seuil {exchange.max_clock_offset_ms} ms ; "
            "resynchroniser Windows : w32tm /resync /force",
            {"offset_ms": clock.offset_ms, "uncertainty_ms": clock.uncertainty_ms},
        )


def _report_changes(ctx: Context, previous: Snapshot, snap: Snapshot) -> None:
    diff = diff_snapshots(previous, snap)
    print(
        f"Changements depuis {previous.path.name} : +{len(diff.added)} paire(s), "
        f"-{len(diff.removed)}, {len(diff.status_changed)} statut(s), "
        f"{len(diff.filters_changed)} filtre(s), frais {'modifiés' if diff.fees_changed else '='}"
    )
    ctx.journal.info(
        "snapshot.diff",
        {
            "previous": previous.path.name,
            "added": list(diff.added),
            "removed": list(diff.removed),
            "status_changed": [list(c) for c in diff.status_changed],
            "filters_changed": list(diff.filters_changed),
            "fees_changed": diff.fees_changed,
        },
    )
    touched = diff.touching(ctx.config.base.symbols.trade)
    if touched:
        details = [f"{s} : {b} → {a}" for s, b, a in diff.status_changed if s in touched]
        details += [f"{s} : filtres modifiés" for s in diff.filters_changed if s in touched]
        details += [f"{s} : disparue" for s in diff.removed if s in touched]
        _alert(
            ctx,
            "snapshot.traded_pairs_changed",
            "paire(s) tradée(s) touchée(s) : " + " ; ".join(details),
            {"symbols": list(touched)},
        )


def _fetch(ctx: Context, exchange: ExchangeConfig, store: SnapshotStore) -> Snapshot:
    now = now_ms()
    latest = store.latest()
    if (
        ctx.args.if_due
        and latest is not None
        and not is_due(latest, now, exchange.snapshot_refresh_hours)
    ):
        print(
            f"Pas encore dû : dernier snapshot {latest.path.name} "
            f"(période {exchange.snapshot_refresh_hours} h)"
        )
        ctx.journal.info("fetch.skipped", {"latest": latest.path.name})
        return latest

    notify = ctx.notify
    clock = measure_clock(exchange.rest_url + TIME_ENDPOINT, notify=notify)
    _check_clock(ctx, exchange, clock)
    url = exchange.rest_url + ENDPOINT
    info, result = fetch_exchange_info(url, notify=notify)
    rates = account_rates(
        ctx.config.base.secrets_file,
        exchange,
        ctx.config.base.symbols.trade,
        server_time_ms=lambda: now_ms() + clock.offset_ms,
        notify=notify,
    )
    if rates is None:
        print("Pas de clé d'API : frais de repli de la config")
    else:
        effective = effective_by_symbol(rates, exchange)
        ctx.journal.info(
            "fees.account", {s: {"maker": f.maker, "taker": f.taker} for s, f in effective.items()}
        )
    saved = store.save(info, fees_record(exchange, rates), now_ms(), url)
    ctx.journal.info(
        "fetch.ok",
        {
            "file": saved.snapshot.path.name,
            "written": saved.written,
            "attempts": result.attempts,
            "used_weight_1m": result.headers.get("x-mbx-used-weight-1m"),
        },
    )
    if not saved.written:
        print(f"Inchangé, dernier snapshot : {saved.snapshot.path.name}")
    else:
        print(f"Nouveau snapshot : {saved.snapshot.path.name}")
        if saved.previous is not None:
            _report_changes(ctx, saved.previous, saved.snapshot)
    return saved.snapshot


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    parser.add_argument("action", choices=["fetch", "show"])
    parser.add_argument(
        "--if-due", action="store_true", help="fetch : seulement si le dernier snapshot est ancien"
    )
    parser.add_argument("--symbol", help="affiche les filtres de cette paire")


def _action(ctx: Context) -> int:
    exchange = ctx.config.base.exchange(ctx.args.exchange)
    store = SnapshotStore(ctx.paths, exchange.name)
    if ctx.args.action == "fetch":
        snap = _fetch(ctx, exchange, store)
    else:
        latest = store.latest()
        if latest is None:
            raise DataError(f"aucun snapshot : lancer d'abord « fetch » ({store.dir})")
        snap = latest
    print(_summary(snap, len(store.paths())))
    if ctx.args.symbol:
        f = snap.filters(ctx.args.symbol)
        print(
            f"{f.symbol} : tick {f.tick_size}, pas {f.step_size}, "
            f"minNotional {f.min_notional}, qté {f.min_qty}–{f.max_qty}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.exchange.exchange_info",
        description="Snapshots versionnés d'exchangeInfo",
        component="exchange_info",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
