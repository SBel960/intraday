"""Table des stratégies long terme : fiche → signal, données, règle de rééquilibrage.

Une seule table pour le gate de coûts (``lt_costs``) et le backtest : chaque fiche de
``hypotheses/`` (volet long terme) y a une entrée ; une fiche sans entrée est une erreur.

- **Données** (``Market``) : clôtures des paires tradées (``symbols.trade``) sur grille
  journalière, financement moyen de leurs contrats perpétuels (``{base}USDT``), largeur du
  marché calculée sur l'univers observé (``market_state``), calendrier du marché
  (``trading_days_per_year`` : la fenêtre « 1 an » de la fiche financement).
- **Rééquilibrage** : calendaire, tous les ``horizon_s`` de la fiche (en jours), sauf quand le
  signal fixe lui-même ses dates (fin de mois : chaque jour).
- Momentum transversal : la fiche le teste sur tout le marché (``research/ic.py``) et
  l'applique aux paires tradées ; c'est cette application qui est rejouée ici.

Commande (gate LT officiel, rapport dans ``reports/``) :
``python -m qlab.longterm.strategies --config config costs [--hypotheses hypotheses]``
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qlab.core.cli import Context, run_command
from qlab.core.config import QlabConfig, SignalsConfig
from qlab.core.errors import DataError
from qlab.core.files import write_atomic
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, MS_PER_S, date_str, now_ms
from qlab.exchange.snapshots import Snapshot, SnapshotStore
from qlab.longterm import funding, klines, lt_costs, market_state, universe
from qlab.longterm import signals as sg
from qlab.longterm.allocation import Policy
from qlab.research.hypothesis import Hypothesis, load_all


@dataclass(frozen=True, slots=True)
class Market:
    closes: pl.DataFrame  # paires tradées, grille journalière (signals.wide)
    funding: pl.DataFrame  # date_ms, funding_1d (moyenne des contrats des paires tradées)
    breadth: Mapping[int, pl.DataFrame]  # ma_days → date_ms, breadth
    days_per_year: int
    settings: SignalsConfig


Params = Mapping[str, float]


def _days(params: Params, name: str) -> int:
    value = params[name]
    if value != int(value):
        raise DataError(f"{name} = {value} : nombre de jours entier attendu")
    return int(value)


def _ts_momentum(m: Market, p: Params) -> pl.DataFrame:
    return sg.ts_momentum(m.closes, _days(p, "lookback_days"))


def _xs_momentum(m: Market, p: Params) -> pl.DataFrame:
    present = m.closes.select(sg.DATE, *(pl.col(c).is_not_null() for c in m.closes.columns[1:]))
    skip = m.settings.xs_skip_days
    return sg.xs_momentum(m.closes, present, _days(p, "lookback_days"), skip, _days(p, "top_k"))


def _low_volatility(m: Market, p: Params) -> pl.DataFrame:
    return sg.low_volatility(m.closes, _days(p, "vol_lookback_days"))


def _short_reversal(m: Market, p: Params) -> pl.DataFrame:
    return sg.short_reversal(m.closes, _days(p, "drop_lookback_days"))


def _funding_leverage(m: Market, p: Params) -> pl.DataFrame:
    return sg.funding_leverage(m.closes, m.funding, p["funding_quantile"], m.days_per_year)


def _market_breadth(m: Market, p: Params) -> pl.DataFrame:
    return sg.market_breadth(m.closes, m.breadth[_days(p, "ma_days")], p["min_breadth"])


def _turn_of_month(m: Market, p: Params) -> pl.DataFrame:
    return sg.turn_of_month(m.closes, _days(p, "pre_days"), _days(p, "post_days"))


@dataclass(frozen=True, slots=True)
class Strategy:
    build: Callable[[Market, Params], pl.DataFrame]
    rebalance_days: int | None = None  # None : horizon de la fiche


REGISTRY: dict[str, Strategy] = {
    "lt_ts_momentum": Strategy(_ts_momentum),
    "lt_xs_momentum": Strategy(_xs_momentum),
    "lt_low_volatility": Strategy(_low_volatility),
    "lt_short_reversal": Strategy(_short_reversal),
    "lt_funding_leverage": Strategy(_funding_leverage),
    "lt_market_breadth": Strategy(_market_breadth),
    "lt_turn_of_month": Strategy(_turn_of_month, rebalance_days=1),
}


def policy_for(hypothesis: Hypothesis) -> Policy:
    """Calendaire : période = horizon de la fiche en jours (ou celle imposée par le signal)."""
    strategy = strategy_for(hypothesis)
    if strategy.rebalance_days is not None:
        return Policy("calendar", period_days=strategy.rebalance_days)
    days, rest = divmod(hypothesis.horizon_s * MS_PER_S, MS_PER_DAY)
    if rest or days < 1:
        raise DataError(f"{hypothesis.id} : horizon_s doit être un nombre entier de jours")
    return Policy("calendar", period_days=days)


def strategy_for(hypothesis: Hypothesis) -> Strategy:
    if hypothesis.id not in REGISTRY:
        raise DataError(f"{hypothesis.id} : aucune stratégie dans longterm/strategies.py")
    return REGISTRY[hypothesis.id]


def load_market(
    paths: DataPaths,
    config: QlabConfig,
    snapshot: Snapshot,
    ma_days: Sequence[int],
    exchange: str,
) -> Market:
    """Charge les données de toutes les fiches (largeur seulement pour ``ma_days``)."""
    trade = config.base.symbols.trade
    closes = sg.wide({s: klines.load(paths, universe.INTERVAL, s) for s in trade})
    known = snapshot.by_symbol()
    contracts = [f"{known[s]['baseAsset']}USDT" for s in trade if s in known]
    rates = pl.concat([funding.daily(funding.load(paths, c)) for c in contracts])
    mean_rate = rates.group_by("date_ms").agg(pl.col("funding_1d").mean()).sort("date_ms")
    breadth: dict[int, pl.DataFrame] = {}
    if ma_days:
        members = universe.observed(paths, known, config.longterm.universe).members
        state = market_state.market_state(paths, members, ma_days)
        breadth = {n: state.select("date_ms", breadth=f"breadth_{n}") for n in ma_days}
    days = config.base.exchange(exchange).trading_days_per_year
    return Market(closes, mean_rate, breadth, days, config.longterm.signals)


def trial_name(hypothesis: Hypothesis, params: Params) -> str:
    return f"{hypothesis.id} · " + ", ".join(f"{k}={v:g}" for k, v in params.items())


def cost_rows(
    config: QlabConfig, snapshot: Snapshot, market: Market, hypotheses: Sequence[Hypothesis]
) -> list[lt_costs.Row]:
    """Gate LT : chaque essai (fiche × paramètres) à chaque palier de capital."""
    cfg = config.longterm.costs
    tiers = [(t, lt_costs.tier_costs(config, snapshot, t, {})) for t in config.base.capital_tiers]
    rows = []
    for h in hypotheses:
        policy, strategy = policy_for(h), strategy_for(h)
        for params in h.grid():
            weights = strategy.build(market, params)
            for tier, costs in tiers:
                res = lt_costs.simulate(weights, market.closes, policy, costs, market.days_per_year)
                rows.append(
                    lt_costs.Row(trial_name(h, params), tier.name, res, lt_costs.gate(res, h, cfg))
                )
    return rows


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cmd", required=True)
    costs = sub.add_parser("costs", help="gate de coûts LT de toutes les fiches long terme")
    costs.add_argument("--hypotheses", type=Path, default=Path("hypotheses"))
    costs.add_argument("--exchange", default="binance", help="nom dans base.yaml")


def _action(ctx: Context) -> int:
    config = ctx.config
    snapshot = SnapshotStore(ctx.paths, config.base.exchange(ctx.args.exchange).name).latest()
    if snapshot is None:
        raise DataError("aucun snapshot exchangeInfo : lancer d'abord exchange_info fetch")
    fiches = [h for h in load_all(ctx.args.hypotheses) if h.volet == "longterm"]
    ma_days = sorted(
        {int(v) for h in fiches for p in h.parameters if p.name == "ma_days" for v in p.values}
    )
    market = load_market(ctx.paths, config, snapshot, ma_days, ctx.args.exchange)
    rows = cost_rows(config, snapshot, market, fiches)
    now = now_ms()
    body = lt_costs.render(rows, config.longterm.costs, spread_measured=False)
    report = ctx.paths.reports / f"lt_costs_{date_str(now)}.md"
    title = f"# Gate de coûts long terme — {date_str(now)} (snapshot {snapshot.path.name})\n\n"
    write_atomic(report, (title + body + "\n").encode("utf-8"), overwrite=True)
    kept = sorted({r.trial for r in rows if r.verdict.passed})
    print(body)
    print(f"\nRapport : {report}")
    ctx.journal.info("strategies.costs", {"trials": len(rows), "passed_somewhere": len(kept)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.longterm.strategies",
        description="Stratégies long terme : gate de coûts des fiches",
        component="strategies",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
