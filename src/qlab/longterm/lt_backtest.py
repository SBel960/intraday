"""Backtest long terme barre à barre (SPEC_LONG_TERME LT.5), en argent réel simulé.

Chaque jour ``t`` de la grille (``signals.wide``) :

1. **ouverture de t** : apport éventuel (``flows``, ex. DCA), puis exécution des échanges
   décidés la veille, **au prix d'ouverture** de t, ventes d'abord : prix × (1 ± impact), impact
   = ½ spread + slippage (+ conversion), frais taker réels de la paire ; quantité arrondie au
   pas et ordre refusé sous ``minNotional`` par ``exchange/lot.py`` (rejet compté par motif) ;
   achat limité au cash disponible frais compris ; pas de prix d'ouverture ⇒ ordre manqué ;
2. **clôture de t** : valeur V_t = cash + Σ quantité × dernière clôture connue ;
3. **décision** avec les seules données ≤ t : poids actuels, poids cibles de la ligne t,
   politique (``allocation``) et δ_min = minNotional / V_t (vraie valeur du portefeuille).

Montants et quantités en ``Decimal`` ; poids en ``float`` (fractions). Les règles d'ordre
viennent du snapshot ``exchangeInfo`` actuel, appliqué au passé : limite assumée, écrite dans le
rapport (Binance ne publie pas l'historique des filtres).

Rendements : TWR jour par jour (apports neutralisés : r_t = (V_t − F_t) / V_{t−1} − 1) ; le MWR
se calcule sur ``flows`` et la valeur finale (``lt_report``). ``peek`` fabrique la version
« qui voit l'avenir » d'une stratégie : elle doit faire mieux (test anti-fuite de LT.5).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import polars as pl

from qlab.core.config import QlabConfig
from qlab.core.errors import DataError
from qlab.exchange.fees import pair_fees
from qlab.exchange.lot import Side, SymbolFilters, check_order
from qlab.exchange.snapshots import Snapshot
from qlab.longterm import allocation
from qlab.longterm.signals import DATE

ZERO = Decimal(0)
_CLOSED = 1e-9  # poids restant sous lequel une vente solde la position


@dataclass(frozen=True, slots=True)
class PairRules:
    filters: SymbolFilters
    fee_frac: Decimal  # frais taker réels
    impact_frac: Decimal  # ½ spread + slippage + conversion, par ordre


def pair_rules(
    config: QlabConfig, snapshot: Snapshot, spreads: Mapping[str, float]
) -> dict[str, PairRules]:
    """Règles des paires tradées d'après le snapshot (filtres, frais réels) et la config."""
    cfg = config.longterm.costs
    out = {}
    for symbol in config.base.symbols.trade:
        spread = spreads.get(symbol, cfg.fallback_spread_frac)
        impact = spread / 2 + cfg.slippage_frac + cfg.eur_conversion_cost_frac
        fees = pair_fees(snapshot.fees, symbol)
        out[symbol] = PairRules(snapshot.filters(symbol), _dec(fees.taker), _dec(impact))
    return out


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


@dataclass(frozen=True, slots=True)
class Fill:
    date_ms: int
    asset: str
    side: str
    qty: Decimal
    price: Decimal
    notional: Decimal
    fee: Decimal


@dataclass(slots=True)
class Result:
    dates: list[int]
    equity: list[Decimal] = field(default_factory=list)  # valeur à la clôture
    flows: list[Decimal] = field(default_factory=list)  # apport à l'ouverture du jour
    fills: list[Fill] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)  # motif lot.py → nombre
    missed: int = 0  # ordres sans prix d'ouverture
    skipped: int = 0  # échanges voulus sous δ_min, non tentés (seraient refusés par Binance)

    @property
    def fees(self) -> Decimal:
        return sum((f.fee for f in self.fills), ZERO)

    def twr_returns(self) -> np.ndarray:
        """r_t = (V_t − F_t) / V_{t−1} − 1, t ≥ 1 (jours sans valeur précédente exclus)."""
        v = np.array([float(x) for x in self.equity])
        f = np.array([float(x) for x in self.flows])
        prev = v[:-1]
        return np.where(prev > 0, (v[1:] - f[1:]) / np.where(prev > 0, prev, 1) - 1, 0.0)


@dataclass(slots=True)
class _Book:
    """Portefeuille en cours : cash et quantités."""

    cash: Decimal
    qty: dict[str, Decimal]
    last: dict[str, Decimal]  # dernière clôture (ou ouverture) connue

    def value(self) -> Decimal:
        return self.cash + sum((q * self.last[a] for a, q in self.qty.items() if q), ZERO)

    def weights(self, total: Decimal) -> dict[str, float]:
        return {a: float(q * self.last[a] / total) for a, q in self.qty.items() if q > 0}


def _execute(
    book: _Book,
    res: Result,
    trade: allocation.Trade,
    *,
    day: int,
    open_price: float,
    value: Decimal,
    rules: PairRules,
) -> None:
    """Un échange à l'ouverture (voir l'en-tête, point 1)."""
    if not np.isfinite(open_price):
        res.missed += 1
        return
    held = book.qty.get(trade.asset, ZERO)
    side: Side
    wanted = abs(_dec(trade.delta_w)) * value
    if trade.delta_w < 0:
        side, price = "SELL", _dec(open_price) * (1 - rules.impact_frac)
        closing = book.weights(value).get(trade.asset, 0.0) + trade.delta_w < _CLOSED
        qty = held if closing else min(wanted / price, held)
    else:
        side, price = "BUY", _dec(open_price) * (1 + rules.impact_frac)
        qty = min(wanted, book.cash / (1 + rules.fee_frac)) / price
    check = check_order(rules.filters, side, "MARKET", qty, price)
    if not check.accepted:
        res.rejected[str(check.reason)] += 1
        return
    notional = check.qty * price
    fee = notional * rules.fee_frac
    sign = 1 if side == "BUY" else -1
    book.cash -= sign * notional + fee
    book.qty[trade.asset] = held + sign * check.qty
    book.last[trade.asset] = _dec(open_price)
    res.fills.append(Fill(day, trade.asset, side, check.qty, price, notional, fee))


def run(
    weights: pl.DataFrame,
    opens: pl.DataFrame,
    closes: pl.DataFrame,
    policy: allocation.Policy,
    rules: Mapping[str, PairRules],
    *,
    initial_quote: Decimal,
    flows: Mapping[int, Decimal] | None = None,
) -> Result:
    """Rejoue ``weights`` avec de l'argent : voir l'en-tête du module."""
    assets = [c for c in weights.columns if c != DATE]
    dates: list[int] = weights[DATE].to_list()
    if opens[DATE].to_list() != dates or closes[DATE].to_list() != dates:
        raise DataError("poids, ouvertures et clôtures : même grille de dates attendue")
    missing = sorted(set(assets) - set(rules))
    if missing or initial_quote < 0:
        raise DataError(f"règles d'ordre inconnues {missing} ou capital initial négatif")
    w, o, c = (df.select(assets).to_numpy() for df in (weights, opens, closes))
    book = _Book(initial_quote, {a: ZERO for a in assets}, {a: ZERO for a in assets})
    res, decided_value = Result(dates), ZERO
    pending: tuple[allocation.Trade, ...] = ()
    for t, day in enumerate(dates):
        flow = (flows or {}).get(day, ZERO)
        book.cash += flow
        for trade in pending:
            i = assets.index(trade.asset)
            _execute(
                book,
                res,
                trade,
                day=day,
                open_price=o[t, i],
                value=decided_value,
                rules=rules[trade.asset],
            )
        for i, a in enumerate(assets):
            if np.isfinite(c[t, i]):
                book.last[a] = _dec(c[t, i])
        value = book.value()
        res.equity.append(value)
        res.flows.append(flow)
        pending, decided_value = (), value
        if value > 0:
            target = {a: float(x) for a, x in zip(assets, w[t], strict=True) if x > 0}
            delta_min = {a: float(rules[a].filters.min_notional / value) for a in assets}
            decision = allocation.decide(
                policy,
                date_ms=day,
                anchor_ms=dates[0],
                current=book.weights(value),
                target=target,
                delta_min=delta_min,
            )
            pending = decision.trades
            res.skipped += len(decision.skipped)
    return res


def peek(weights: pl.DataFrame, bars: int = 1) -> pl.DataFrame:
    """Poids décalés de ``bars`` vers le passé : la stratégie « voit » l'avenir. Sert au test
    anti-fuite : ce tricheur doit battre la vraie stratégie, sinon le pipeline est suspect."""
    if bars < 1:
        raise DataError("bars ≥ 1 attendu")
    return weights.select(DATE, pl.exclude(DATE).shift(-bars).fill_null(0.0))
