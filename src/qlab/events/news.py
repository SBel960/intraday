"""Annonces Binance (cotations, retraits), horodatées à la **réception** (phase 9, règle 1).

Source : liste paginée publique et non documentée de chaque rubrique (``events.catalogs``) ;
une réponse de forme inattendue arrête la commande, rien n'est deviné. Chaque annonce est
écrite une seule fois (dédoublonnage sur l'identifiant de la source) dans
``events/news/date=AAAA-MM-JJ/binance_cms.jsonl``, en ajout seulement :

``source``, ``source_id``, ``code``, ``catalog``, ``title``, ``published_ms``, ``received_ms``
(``None`` pour une annonce rechargée de l'historique : on ne sait pas quand on l'aurait reçue),
``backfilled``, ``sha256`` (titre + code : preuve que rien n'a été réécrit). Le classement
(type, actifs) est fait à la lecture par ``events/classify.py`` : de nouvelles règles ne
réécrivent jamais le RAW.

- ``backfill`` : toutes les pages de chaque rubrique (historique) ;
- ``poll`` : première page de chaque rubrique ; ``received_ms`` = heure du serveur de
  ``events.exchange`` (horloge locale corrigée), refus si l'horloge dérive au-delà de son
  ``max_clock_offset_ms`` (CLAUDE.md).

Commandes : ``python -m qlab.events.news --config config backfill`` ; ``… poll``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
from collections.abc import Sequence
from typing import Any

from qlab.core import http
from qlab.core.cli import Context, run_command
from qlab.core.config import EventCatalog, EventsConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.records import Record, append_record, read_records
from qlab.core.timeutils import date_str, now_ms
from qlab.exchange.exchange_info import TIME_ENDPOINT, measure_clock

SOURCE = "binance_cms"
HEADERS = {"User-Agent": "Mozilla/5.0 (qlab research)"}


def page_url(cfg: EventsConfig, catalog: EventCatalog, page: int) -> str:
    query = {"type": 1, "catalogId": catalog.id, "pageNo": page, "pageSize": cfg.page_size}
    return f"{cfg.list_url}?{urllib.parse.urlencode(query)}"


def parse_page(body: bytes, catalog: EventCatalog) -> list[dict[str, Any]]:
    """Réponse d'une page → annonces (``source_id``, ``code``, ``title``, ``published_ms``)."""
    try:
        blob = json.loads(body)
        catalogs = blob["data"]["catalogs"]
        articles = catalogs[0]["articles"] if catalogs else []
        return [
            {
                "source_id": int(a["id"]),
                "code": str(a["code"]),
                "title": str(a["title"]),
                "published_ms": int(a["releaseDate"]),
                "catalog": catalog.name,
            }
            for a in articles
        ]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataError(f"rubrique {catalog.name} : forme de réponse inattendue ({exc!r})") from exc


def record(article: dict[str, Any], received_ms: int | None) -> dict[str, Any]:
    digest = hashlib.sha256(f"{article['title']}\n{article['code']}".encode()).hexdigest()
    return {
        "source": SOURCE,
        **article,
        "received_ms": received_ms,
        "backfilled": received_ms is None,
        "sha256": digest,
    }


def stored(paths: DataPaths) -> list[Record]:
    """Toutes les annonces enregistrées, tous jours confondus."""
    files = sorted(paths.events_news_dir.glob("date=*/binance_cms.jsonl"))
    return [r for f in files for r in read_records(f)]


def store(paths: DataPaths, articles: list[dict[str, Any]], received_ms: int | None) -> int:
    """Ajoute les annonces encore inconnues ; renvoie leur nombre."""
    known = {r["source_id"] for r in stored(paths)}
    new = [a for a in articles if a["source_id"] not in known]
    for a in sorted(new, key=lambda a: a["published_ms"]):
        day = date_str(received_ms if received_ms is not None else a["published_ms"])
        append_record(paths.events_news(day), record(a, received_ms))
    return len(new)


def _fetch(cfg: EventsConfig, catalog: EventCatalog, page: int) -> list[dict[str, Any]]:
    return parse_page(http.get(page_url(cfg, catalog, page), headers=HEADERS).body, catalog)


def _backfill(ctx: Context) -> int:
    cfg = ctx.config.events
    for catalog in cfg.catalogs:
        page, total = 1, 0
        while articles := _fetch(cfg, catalog, page):
            total += store(ctx.paths, articles, None)
            page += 1
            time.sleep(cfg.request_interval_s)
        print(f"{catalog.name} : {total} nouvelles annonces ({page - 1} pages)", flush=True)
    return 0


def _poll(ctx: Context) -> int:
    cfg = ctx.config.events
    ex = ctx.config.base.exchange(cfg.exchange)
    clock = measure_clock(ex.rest_url + TIME_ENDPOINT, notify=ctx.notify)
    if abs(clock.offset_ms) - clock.uncertainty_ms > ex.max_clock_offset_ms:
        raise DataError(
            f"horloge décalée de {-clock.offset_ms:+d} ms (seuil {ex.max_clock_offset_ms}) : "
            "réception non horodatée ; resynchroniser Windows (w32tm /resync /force)"
        )
    for catalog in cfg.catalogs:
        articles = _fetch(cfg, catalog, 1)
        new = store(ctx.paths, articles, now_ms() + clock.offset_ms)
        ctx.journal.info("events.poll", {"catalog": catalog.name, "new": new})
        print(f"{catalog.name} : {new} nouvelle(s) annonce(s)", flush=True)
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill", help="recharger tout l'historique des rubriques")
    sub.add_parser("poll", help="relever la première page de chaque rubrique (réception)")


def _action(ctx: Context) -> int:
    return _backfill(ctx) if ctx.args.cmd == "backfill" else _poll(ctx)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.events.news",
        description="Annonces Binance horodatées à la réception (phase 9)",
        component="events_news",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
