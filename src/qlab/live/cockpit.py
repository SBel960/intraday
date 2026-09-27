"""Cockpit local : page de suivi en direct (``cockpit.html``) et l'état de qlab qu'elle affiche.

``python -m qlab.live.cockpit --config config --trial "lt_ts_momentum · lookback_days=30"``
puis ouvrir l'adresse affichée dans le navigateur (Windows : WSL2 relaie ``localhost``).

- **Prix, bougies, spreads, transactions en direct** : la page les reçoit elle-même des flux
  publics de l'échange (sans clé) ; ce module ne sort pas sur le réseau.
- **État de qlab** (``/api/state``, recalculé au plus toutes les ``CACHE_MS``) : signal de
  l'essai suivi à la dernière clôture locale, fraîcheur des données, spreads médians relevés,
  minuteurs systemd ``qlab-*``, registre des essais, limites de risque, disque ;
- **paper trading** (``paper_state``) : portefeuille rebâti depuis le journal
  (``paper_book.replay``), valeurs de clôture, derniers passages, prochain rééquilibrage. La page
  le valorise aux prix en direct ; sans journal, elle garde un portefeuille **fictif** (capital
  du palier réparti selon le signal).

Le serveur n'écoute que ``127.0.0.1`` : rien n'est exposé au réseau local. Il ne lit aucune clé
et ne passe aucun ordre.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import threading
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.config import QlabConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.records import Record, read_records
from qlab.core.timeutils import MS_PER_DAY, MS_PER_MIN, US_PER_MS, date_str, now_ms
from qlab.exchange.snapshots import SnapshotStore
from qlab.exchange.spreads import medians as spread_medians
from qlab.live import paper_book, paper_report
from qlab.live.paper import book_name
from qlab.longterm import signals as sg
from qlab.longterm import strategies
from qlab.longterm.allocation import Policy, is_calendar_day
from qlab.research.hypothesis import Hypothesis, load_all
from qlab.research.trials import TrialRegistry, TrialResult

PAGE = Path(__file__).with_name("cockpit.html")
HOST = "127.0.0.1"  # jamais 0.0.0.0 : la page n'est visible que de cette machine
CACHE_MS = MS_PER_MIN
TIMER_PREFIX = "qlab-"
SYSTEMCTL = ("systemctl", "--user", "list-timers", "--all", "--output=json", "--no-pager")


def signal(
    market: strategies.Market, hypothesis: Hypothesis, params: Mapping[str, float]
) -> dict[str, Any]:
    """Poids voulus à la dernière clôture disponible, et ces clôtures (base du portefeuille
    fictif : sa valeur suit les prix en direct depuis cette décision)."""
    weights = strategies.strategy_for(hypothesis).build(market, params)
    if weights.is_empty():
        raise DataError("aucune clôture locale : construire d'abord les klines 1d")
    last = weights.row(-1, named=True)
    closes = market.closes.filter(pl.col(sg.DATE) == last[sg.DATE]).row(0, named=True)
    assets = [c for c in weights.columns if c != sg.DATE]
    return {
        "date": date_str(last[sg.DATE]),
        "close_ms": last[sg.DATE] + MS_PER_DAY - 1,
        "weights": {a: float(last[a]) for a in assets},
        "closes": {a: closes[a] for a in assets},
        "policy_days": strategies.policy_for(hypothesis).period_days,
    }


def paper_state(
    paths: DataPaths, trial: str, anchor_ms: int, policy: Policy, min_days: int
) -> dict[str, Any] | None:
    """Portefeuille du paper d'après son journal ; ``None`` s'il n'a pas été ouvert."""
    path = paths.paper_journal(book_name(trial))
    if not path.exists():
        return None
    records = read_records(path)
    book = paper_book.replay(records)
    days = [r for r in records if r.get("kind") == "day"]
    run = paper_report.paper_run(records) if days else None
    yesterday = now_ms() // MS_PER_DAY * MS_PER_DAY - MS_PER_DAY
    last = yesterday if book.last_bar_ms is None else book.last_bar_ms
    return {
        "started_ms": records[0]["run_ms"],
        "capital": records[0]["capital"],
        "tier": records[0]["tier"],
        "cash": str(book.cash),
        "held": {s: str(q) for s, q in book.held.items() if q},
        "halted": book.halted,
        "days": 0 if run is None else run.days,
        "min_days": min_days,
        "missed_bars": 0 if run is None else run.missed_bars,
        "equity": [[r["bar_ms"], r["value_close"]] for r in days],
        "recent": [_passage(r) for r in reversed(days[-7:])],
        "next_rebalance_bar_ms": next_rebalance(last, anchor_ms, policy),
    }


def next_rebalance(last_bar_ms: int, anchor_ms: int, policy: Policy) -> int | None:
    """Première barre de rééquilibrage après ``last_bar_ms`` (calendaire seulement)."""
    if policy.kind != "calendar":
        return None
    days = (last_bar_ms + k * MS_PER_DAY for k in range(1, policy.period_days + 1))
    return next(d for d in days if is_calendar_day(d, anchor_ms, policy.period_days))


def _passage(record: Record) -> dict[str, Any]:
    orders = [
        f"{e['side']} {e['symbol']} {e['qty']}"
        + ("" if e["accepted"] else f" refusé ({e['reason']})")
        for e in record["executions"]
    ]
    risk = record["risk"]
    return {
        "bar_ms": record["bar_ms"],
        "value": record["value_close"],
        "action": risk["action"],
        "reasons": risk["reasons"],
        "orders": orders,
        "missed": record.get("missed_bars", 0),
    }


def qlab_timers(raw: str) -> list[dict[str, Any]]:
    """Minuteurs ``qlab-*`` de ``systemctl --output=json`` (µs) en ms ; 0 ⇒ ``None``."""
    out = []
    for t in json.loads(raw or "[]"):
        if str(t.get("unit", "")).startswith(TIMER_PREFIX):
            ms = {k: (t[k] // US_PER_MS if t.get(k) else None) for k in ("next", "last")}
            out.append({"unit": t["unit"], "service": t.get("activates"), **ms})
    return out


def _systemd_timers() -> list[dict[str, Any]]:
    try:
        done = subprocess.run(SYSTEMCTL, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return []  # pas de systemd (CI) : la page l'indique
    return qlab_timers(done.stdout)


def collect(
    config: QlabConfig, paths: DataPaths, hypotheses_dir: Path, trial: str, exchange: str
) -> dict[str, Any]:
    """Tout l'état affiché par la page (hors prix en direct)."""
    ex = config.base.exchange(exchange)
    store = SnapshotStore(paths, ex.name)
    snapshot = store.latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    hypothesis, params = strategies.trial_by_name(load_all(hypotheses_dir), trial)
    market = strategies.load_market(paths, config, snapshot, (), exchange)
    registry = TrialRegistry(paths.trials)
    recorded = registry.find("longterm", hypothesis.id, params)
    min_samples = config.longterm.costs.spread_min_samples
    spreads = spread_medians(paths, min_samples) if paths.spreads.exists() else {}
    tier = config.base.capital_tiers[0]
    risk = config.longterm.risk
    disk = shutil.disk_usage(paths.root)
    return {
        "generated_ms": now_ms(),
        "exchange": {
            "name": ex.name,
            "rest_url": ex.rest_url,
            "ws_url": ex.ws_url,
            "max_clock_offset_ms": ex.max_clock_offset_ms,
        },
        "quote": config.base.symbols.quote_asset,
        "symbols": list(config.base.symbols.trade),
        "trial": {
            "name": trial,
            "params": params,
            "sharpe_annual": None if recorded is None else _annual(recorded.result),
        },
        "signal": signal(market, hypothesis, params),
        "capital": {"tier": tier.name, "amount": tier.capital_quote},
        "risk": {
            "max_open_positions": risk.max_open_positions,
            "max_day_loss_frac": risk.max_day_loss_frac,
            "max_drawdown_frac": risk.max_drawdown_frac,
            "max_bar_age_hours": risk.max_bar_age_hours,
        },
        "spreads": {
            "medians": {s: spreads[s] for s in config.base.symbols.trade if s in spreads},
            "samples": sum(1 for _ in read_records(paths.spreads)) if paths.spreads.exists() else 0,
            "min_samples": min_samples,
            "fallback": config.longterm.costs.fallback_spread_frac,
        },
        "snapshot": {"name": snapshot.path.name, "fetched_ms": snapshot.fetched_ms},
        "trials": {"n": registry.n_trials("longterm")},
        "paper": paper_state(
            paths,
            trial,
            int(market.closes[sg.DATE][0]),
            strategies.policy_for(hypothesis),
            config.longterm.acceptance.paper_min_days,
        ),
        "timers": _systemd_timers(),
        "disk": {"used": disk.used, "total": disk.total},
    }


def _annual(result: TrialResult) -> float:
    return result.sharpe * math.sqrt(result.periods_per_year)


class _State:
    """État mis en cache (``CACHE_MS``) : recharger les klines à chaque rafraîchissement de
    la page serait inutile, elles changent une fois par jour."""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._lock = threading.Lock()
        self._body = b""
        self._at = 0

    def body(self) -> bytes:
        with self._lock:
            if now_ms() - self._at >= CACHE_MS:
                a = self._ctx.args
                data = collect(self._ctx.config, self._ctx.paths, a.hypotheses, a.trial, a.exchange)
                self._body, self._at = json.dumps(data).encode(), now_ms()
            return self._body


def _handler(state: _State, page: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                try:
                    self._send(state.body(), "application/json")
                except DataError as e:
                    msg = json.dumps({"error": str(e)}).encode()
                    self._send(msg, "application/json", HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self._send(b"introuvable", "text/plain", HTTPStatus.NOT_FOUND)

        def _send(self, body: bytes, kind: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return  # une ligne par requête (toutes les 15 s) noierait le terminal

    return Handler


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial", required=True, help="nom exact de l'essai suivi")
    parser.add_argument("--hypotheses", type=Path, default=Path("hypotheses"))
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    parser.add_argument("--port", type=int, default=8765)


def _action(ctx: Context) -> int:
    state = _State(ctx)
    state.body()  # erreur de données visible tout de suite, pas au premier affichage
    server = ThreadingHTTPServer((HOST, ctx.args.port), _handler(state, PAGE.read_bytes()))
    url = f"http://localhost:{ctx.args.port}"
    ctx.journal.info("cockpit.start", {"url": url, "trial": ctx.args.trial})
    print(f"Cockpit : {url}  (Ctrl+C pour arrêter)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Cockpit arrêté.")
    finally:
        server.server_close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="qlab.live.cockpit",
        description="cockpit local en direct (127.0.0.1)",
        component="cockpit",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
