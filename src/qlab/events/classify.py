"""Classement des annonces par règles déterministes sur le titre (phase 9, version 1).

Pas de modèle de langage (règle 2 de la phase 9). Règles versionnées (``RULES_VERSION``) et
appliquées à la lecture : de nouvelles règles ne réécrivent jamais le RAW
(``events/news.py``). Un titre qu'aucune règle ne reconnaît devient ``other``, compté dans les
rapports pour améliorer les règles sans jamais deviner.

| rubrique | titre | type | actifs |
|---|---|---|---|
| retraits | « Binance Will Delist A, B and C on AAAA-MM-JJ » | ``delisting`` | A, B, C |
| retraits | « Notice of Removal of … Trading Pairs … » | ``pair_removal`` (l'actif reste) | — |
| retraits | « Binance Futures Will Delist … » | ``perp_delisting`` | bases des ``…USDT`` cités |
| cotations | « Binance Will List Nom (TICKER) », « Binance Lists TICKER » | ``listing`` | TICKER |
| cotations | « Binance Futures Will Launch … » | ``perp_launch`` | — |

Un retrait qui vise des produits (« … Margin Pairs », « Futures Contracts », « Options »,
« Earn ») et non le jeton lui-même n'est pas un ``delisting``.

``events_table`` : annonces → une ligne par (type, actif) avec l'heure **effective** : réception
si connue, sinon publication + ``events.latency_margin_s`` (règle 1).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from qlab.core.records import Record

RULES_VERSION = 1
_TICKER = r"[A-Z0-9]{2,15}"
_DELIST = re.compile(r"^Binance Will Delist (.+?) on \d{4}-\d{2}-\d{2}")
_PRODUCTS = re.compile(r"Margin|Futures|Options|Earn|Products|Pairs")
_PAIR_REMOVAL = re.compile(r"^Notice of Removal of .*Trading Pairs")
_PERP_DELIST = re.compile(r"^Binance Futures Will Delist")
_LIST = re.compile(r"^Binance Will List (.+)")
_LISTS = re.compile(rf"^Binance Lists ({_TICKER})\b")
_PERP_LAUNCH = re.compile(r"^Binance Futures Will Launch")
_PARENS = re.compile(rf"\(({_TICKER})\)")
_LEADING = re.compile(rf"^({_TICKER})\b")


@dataclass(frozen=True, slots=True)
class Classified:
    kind: str
    bases: tuple[str, ...]


def _tickers(text: str) -> tuple[str, ...]:
    parts = re.split(r",\s*|\s+and\s+|\s*&\s*", text)
    return tuple(p.strip() for p in parts if re.fullmatch(_TICKER, p.strip()))


def _delisting_rubric(title: str) -> Classified | None:
    if (m := _DELIST.match(title)) and not _PRODUCTS.search(m[1]):
        return Classified("delisting", _tickers(m[1]))
    if _PAIR_REMOVAL.match(title):
        return Classified("pair_removal", ())
    if _PERP_DELIST.match(title):
        return Classified("perp_delisting", tuple(re.findall(rf"({_TICKER})USDT\b", title)))
    return None


def _listing_rubric(title: str) -> Classified | None:
    if m := _LIST.match(title):
        return Classified("listing", tuple(_PARENS.findall(m[1])) or tuple(_LEADING.findall(m[1])))
    if m := _LISTS.match(title):
        return Classified("listing", (m[1],))
    if _PERP_LAUNCH.match(title):
        return Classified("perp_launch", ())
    return None


_RUBRICS = {"delisting": _delisting_rubric, "listing": _listing_rubric}


def classify(title: str, catalog: str) -> Classified:
    """Type et actifs d'une annonce (voir le tableau du module)."""
    rules = _RUBRICS.get(catalog)
    found = rules(title) if rules is not None else None
    return found if found is not None else Classified("other", ())


def events_table(records: Sequence[Record], latency_margin_s: int) -> pl.DataFrame:
    """(``kind``, ``base``, ``published_ms``, ``effective_ms``, ``backfilled``,
    ``rules_version``) : une ligne par actif visé, triées par heure effective."""
    rows = []
    for r in records:
        c = classify(r["title"], r["catalog"])
        received = r["received_ms"]
        effective = (
            received if received is not None else r["published_ms"] + latency_margin_s * 1000
        )
        rows += [
            {
                "kind": c.kind,
                "base": base,
                "published_ms": r["published_ms"],
                "effective_ms": effective,
                "backfilled": r["backfilled"],
                "rules_version": RULES_VERSION,
            }
            for base in c.bases
        ]
    schema = {
        "kind": pl.String,
        "base": pl.String,
        "published_ms": pl.Int64,
        "effective_ms": pl.Int64,
        "backfilled": pl.Boolean,
        "rules_version": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema).sort("effective_ms")
