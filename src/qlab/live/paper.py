"""Paper trading long terme : une décision par barre journalière, journal complet (LT.7).

``python -m qlab.live.paper --config config --trial "lt_ts_momentum · lookback_days=30" CMD``

- ``start`` : ouvre le portefeuille (capital du premier palier, ou ``--tier``) ; refusé si le
  journal existe déjà ;
- ``run`` (minuteur ``qlab-paper``, 00:05 UTC) :
  1. horloge mesurée face au serveur ; refus au-delà de ``max_clock_offset_ms`` (décider sur
     une heure fausse, c'est risquer de prendre une barre pas encore close) ;
  2. état rebâti depuis le journal (``paper_book.replay``) ; fiche modifiée depuis ``start`` ⇒
     refus (ce ne serait plus l'essai jugé) ;
  3. bougies : archives + dernières bougies REST (``klines_rest``) ; ouverture de la barre en
     cours = prix d'exécution ;
  4. signal de l'essai sur la dernière barre close, règles d'ordre du backtest (snapshot le
     plus récent, spreads médians), puis ``paper_book.day`` ;
  5. enregistrement ajouté au journal ``meta/paper/{book}.jsonl`` (sous verrou : une seule
     décision par barre, même si deux passages se chevauchent).
  Barre déjà traitée ⇒ rien à faire. PC éteint plusieurs jours ⇒ seule la dernière barre est
  traitée ; les barres manquées sont comptées dans l'enregistrement (``missed_bars``).
- ``status`` : état du portefeuille ; ``resume --reason "…"`` : lève un arrêt verrouillé.

Limite : le financement des contrats (fiches qui s'en servent) reste celui des archives.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.money import config_decimal
from qlab.core.records import Record, append_record, read_records
from qlab.core.timeutils import MS_PER_DAY, date_str, now_ms
from qlab.exchange.exchange_info import measure_clock
from qlab.exchange.snapshots import SnapshotStore
from qlab.exchange.spreads import medians as spread_medians
from qlab.live import paper_book as pb
from qlab.live.cockpit import trial_by_name
from qlab.longterm import klines, klines_rest, lt_backtest, strategies
from qlab.longterm import signals as sg
from qlab.research.hypothesis import Hypothesis, load_all

INTERVAL = "1d"
TIME_ENDPOINT = "/api/v3/time"


def book_name(trial: str) -> str:
    """Nom de fichier du journal tiré du nom de l'essai (``[a-z0-9_]``)."""
    return re.sub(r"[^a-z0-9]+", "_", trial.lower()).strip("_")


def extend_market(
    market: strategies.Market,
    recent: dict[str, klines_rest.Recent],
    archives: dict[str, pl.DataFrame],
) -> strategies.Market:
    """Clôtures et volumes des paires tradées prolongés par les bougies REST closes."""
    bars = {s: klines_rest.extend(archives[s], recent[s].closed) for s in archives}
    return dataclasses.replace(market, closes=sg.wide(bars), volumes=sg.wide(bars, "volume_quote"))


def today(
    market: strategies.Market,
    recent: dict[str, klines_rest.Recent],
    trial: tuple[Hypothesis, dict[str, float]],
    server_ms: int,
) -> pb.Market:
    """Données du jour pour ``paper_book.day`` : dernière barre close, cible, ouvertures."""
    h, params = trial
    weights = strategies.strategy_for(h).build(market, params)
    last = weights.row(-1, named=True)
    bar = int(last[sg.DATE])
    closes = market.closes.filter(market.closes[sg.DATE] == bar).row(0, named=True)
    opens = {
        s: float(r.current_open)
        for s, r in recent.items()
        if r.current_open is not None and r.current_open_ms == bar + MS_PER_DAY
    }
    return pb.Market(
        bar,
        bar + MS_PER_DAY - 1,
        {s: float(v) for s, v in closes.items() if s != sg.DATE and v is not None},
        opens,
        {s: float(w) for s, w in last.items() if s != sg.DATE and w > 0},
        server_ms,
    )


def _journal(ctx: Context) -> tuple[Path, list[Record]]:
    path = ctx.paths.paper_journal(book_name(ctx.args.trial))
    return path, read_records(path) if path.exists() else []


def _start(ctx: Context, h: Hypothesis) -> int:
    path, records = _journal(ctx)
    if records:
        raise DataError(f"portefeuille déjà ouvert : {path} (un seul start par essai)")
    base = ctx.config.base
    tier = next(
        (t for t in base.capital_tiers if t.name == (ctx.args.tier or base.capital_tiers[0].name)),
        None,
    )
    if tier is None:
        raise DataError(f"palier inconnu : {ctx.args.tier}")
    record = {
        "kind": "start",
        "run_ms": now_ms(),
        "trial": ctx.args.trial,
        "fingerprint": h.fingerprint,
        "quote": base.symbols.quote_asset,
        "tier": tier.name,
        "capital": str(config_decimal(tier.capital_quote)),
    }
    append_record(path, record, precheck=_empty)
    print(f"Portefeuille ouvert : {record['capital']} {record['quote']} ({tier.name}) → {path}")
    return 0


def _empty(existing: list[Record]) -> None:
    if existing:
        raise DataError("portefeuille déjà ouvert")


def _run(ctx: Context, h: Hypothesis, params: dict[str, float]) -> int:
    path, records = _journal(ctx)
    book = pb.replay(records)
    if records[0]["fingerprint"] != h.fingerprint:
        raise DataError(f"{h.id} : fiche modifiée depuis « start » ; ce n'est plus l'essai jugé")
    ex = ctx.config.base.exchange(ctx.args.exchange)
    clock = measure_clock(ex.rest_url + TIME_ENDPOINT, notify=ctx.notify)
    if abs(clock.offset_ms) - clock.uncertainty_ms > ex.max_clock_offset_ms:
        raise DataError(
            f"horloge décalée de {-clock.offset_ms:+d} ms (seuil {ex.max_clock_offset_ms}) : "
            "aucune décision ; resynchroniser Windows (w32tm /resync /force)"
        )
    server = now_ms() + clock.offset_ms
    snapshot = SnapshotStore(ctx.paths, ex.name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    symbols = ctx.config.base.symbols.trade
    archives = {s: klines.load(ctx.paths, INTERVAL, s) for s in symbols}
    recent = {
        s: klines_rest.fetch(
            ex.rest_url, s, INTERVAL, _after(archives[s]), server, notify=ctx.notify
        )
        for s in symbols
    }
    base_market = strategies.load_market(ctx.paths, ctx.config, snapshot, (), ex.name)
    market = extend_market(base_market, recent, archives)
    data = today(market, recent, (h, params), server)
    if book.last_bar_ms is not None and data.bar_ms <= book.last_bar_ms:
        print(f"Barre du {date_str(data.bar_ms)} déjà traitée : rien à faire.")
        return 0
    spreads = spread_medians(ctx.paths, ctx.config.longterm.costs.spread_min_samples)
    rules = lt_backtest.pair_rules(ctx.config, snapshot, spreads)
    plan = pb.Plan(
        strategies.policy_for(h),
        int(market.closes[sg.DATE][0]),
        rules,
        ctx.config.longterm.risk,
    )
    record, executions = pb.day(book, data, plan)
    missed = _missed(records, data.bar_ms)
    record |= {
        "trial": ctx.args.trial,
        "missed_bars": missed,
        "clock_offset_ms": clock.offset_ms,
        "snapshot": snapshot.path.name,
        "anomalies": {s: r.anomalies for s, r in recent.items()},
    }
    append_record(path, record, precheck=lambda old: _not_done(old, data.bar_ms))
    _print_day(record, book)
    summary = {"bar": date_str(data.bar_ms), "risk": record["risk"]["action"]}
    ctx.journal.info("paper.day", {**summary, "executions": len(executions)})
    return 0


def _after(archive: pl.DataFrame) -> int:
    """Ouverture de la première barre après les archives."""
    last = archive["open_time_ms"].max()
    if not isinstance(last, int):
        raise DataError("archives de bougies vides : lancer « klines_build build »")
    return last + MS_PER_DAY


def _missed(records: Sequence[Record], bar_ms: int) -> int:
    days = [int(r["bar_ms"]) for r in records if r.get("kind") == "day"]
    return max(0, (bar_ms - days[-1]) // MS_PER_DAY - 1) if days else 0


def _not_done(existing: list[Record], bar_ms: int) -> None:
    if any(r.get("kind") == "day" and int(r["bar_ms"]) >= bar_ms for r in existing):
        raise DataError("barre déjà traitée par un autre passage")


def _print_day(record: Record, book: pb.Book) -> None:
    risk = record["risk"]
    print(f"Barre du {date_str(record['bar_ms'])} : valeur {record['value_close']} {book.quote}")
    print(f"Garde-fous : {risk['action']}" + "".join(f" — {r}" for r in risk["reasons"]))
    for e in record["executions"]:
        done = "exécuté" if e["accepted"] else f"refusé ({e['reason']})"
        print(f"  {e['side']} {e['symbol']} {e['qty']} @ {e['price']} : {done}")
    print(
        f"Soldes : {record['after']['cash']} {book.quote} "
        + " ".join(f"{s} {q}" for s, q in record["after"]["held"].items())
    )


def _status(ctx: Context) -> int:
    path, records = _journal(ctx)
    book = pb.replay(records)
    days = [r for r in records if r.get("kind") == "day"]
    print(f"Journal : {path} ({len(days)} barre(s))")
    print(f"Capital de départ : {records[0]['capital']} {book.quote} ({records[0]['tier']})")
    if days:
        last = days[-1]
        value = f"{last['value_close']} {book.quote}"
        when, risk = date_str(last["bar_ms"]), last["risk"]["action"]
        print(f"Dernière barre : {when}, valeur {value}, garde-fous : {risk}")
    print(
        f"Soldes : {book.cash} {book.quote} "
        + " ".join(f"{s} {q}" for s, q in book.held.items() if q)
    )
    print("ARRÊT VERROUILLÉ (lever : resume --reason …)" if book.halted else "Actif")
    return 0


def _resume(ctx: Context) -> int:
    path, records = _journal(ctx)
    if not pb.replay(records).halted:
        raise DataError("aucun arrêt à lever")
    append_record(path, {"kind": "resume", "run_ms": now_ms(), "reason": ctx.args.reason})
    print("Arrêt levé : le prochain passage suivra de nouveau le signal.")
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial", required=True, help="nom exact de l'essai (rapports de vague)")
    parser.add_argument("--hypotheses", type=Path, default=Path("hypotheses"))
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    start = sub.add_parser("start", help="ouvrir le portefeuille")
    start.add_argument("--tier", default=None, help="palier (défaut : le premier)")
    sub.add_parser("run", help="passage quotidien (minuteur)")
    sub.add_parser("status", help="état du portefeuille")
    resume = sub.add_parser("resume", help="lever un arrêt verrouillé")
    resume.add_argument("--reason", required=True, help="pourquoi on relance (journalisé)")


def _action(ctx: Context) -> int:
    h, params = trial_by_name(load_all(ctx.args.hypotheses), ctx.args.trial)
    if ctx.args.cmd == "start":
        return _start(ctx, h)
    if ctx.args.cmd == "run":
        return _run(ctx, h, params)
    return _status(ctx) if ctx.args.cmd == "status" else _resume(ctx)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="qlab.live.paper",
        description="paper trading long terme (une décision par barre)",
        component="paper",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
