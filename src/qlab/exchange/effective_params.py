"""Paramètres effectifs, recalculés à chaque appel : palier, frais, limites en devise, δ_min.

Deux montants distincts, jamais confondus :
- **capital apporté** (dépôts − retraits, ``ledger.py``) ⇒ **palier** (``tier_for``) : une
  perte ne fait jamais redescendre de palier ;
- **valeur du portefeuille** (``equity``) ⇒ **limites en devise** et **bande minimale**
  ``δ_min = minNotional / equity`` par paire (SPEC_LONG_TERME LT.4). Sans portefeuille (pas
  encore de paper trading), on prend le capital apporté et on le signale.

Frais et filtres viennent du snapshot ``exchangeInfo`` **en vigueur à la date demandée**
(point-in-time) ; un snapshot appliqué avant sa date est signalé.

``compute`` est pur (sert aussi aux backtests, avec un capital simulé) ; ``current`` lit le
registre et les snapshots. Commande :
``python -m qlab.exchange.effective_params --config config [--equity 48.20] [--date YYYY-MM-DD]``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from qlab.core.cli import Context, run_command
from qlab.core.config import CapitalTier, QlabConfig
from qlab.core.errors import DataError
from qlab.core.ledger import Ledger
from qlab.core.money import config_decimal, format_amount, parse_amount
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, date_to_ms, ms_to_iso, now_ms
from qlab.exchange.fees import fallback_fees, pair_fees
from qlab.exchange.lot import SymbolFilters
from qlab.exchange.snapshots import Snapshot, SnapshotStore


@dataclass(frozen=True, slots=True)
class PairParams:
    """Une paire tradée : filtres en vigueur, ``δ_min`` (fraction du portefeuille), frais
    réellement payés (``fee_origin`` : ``account`` = frais du compte, ``config`` = repli)."""

    symbol: str
    status: str
    filters: SymbolFilters
    delta_min_frac: Decimal
    fee_maker_frac: Decimal
    fee_taker_frac: Decimal
    fee_origin: str

    @property
    def tradable(self) -> bool:
        return self.status == "TRADING"


@dataclass(frozen=True, slots=True)
class EffectiveParams:
    """Paramètres effectifs à ``at_ms``. Montants en devise de cotation, fractions sans unité.

    ``fee_*`` : barème de repli (config) ; les frais réels de chaque paire sont dans ``pairs``.
    """

    at_ms: int
    net_deposits_quote: Decimal
    equity_quote: Decimal
    equity_assumed: bool
    tier: CapitalTier
    fees_origin: str
    fee_maker_frac: Decimal
    fee_taker_frac: Decimal
    max_daily_loss_quote: Decimal
    max_order_quote: Decimal
    snapshot_file: str
    snapshot_extrapolated: bool
    pairs: tuple[PairParams, ...]
    warnings: tuple[str, ...]


def _pairs(config: QlabConfig, snapshot: Snapshot, equity: Decimal) -> tuple[PairParams, ...]:
    """Paires tradées : statut, filtres, δ_min et frais réels (ou repli) en vigueur."""
    known = snapshot.by_symbol()
    pairs = []
    for symbol in config.base.symbols.trade:
        if symbol not in known:
            raise DataError(f"{symbol} (symbols.trade) absent du snapshot {snapshot.path.name}")
        filters = snapshot.filters(symbol)
        fees = pair_fees(snapshot.fees, symbol)
        pairs.append(
            PairParams(
                symbol,
                known[symbol]["status"],
                filters,
                filters.min_notional / equity,
                config_decimal(fees.maker),
                config_decimal(fees.taker),
                fees.origin,
            )
        )
    return tuple(pairs)


def _warnings(
    config: QlabConfig,
    pairs: tuple[PairParams, ...],
    max_order: Decimal,
    *,
    equity_assumed: bool,
    snapshot_file: str,
    snapshot_extrapolated: bool,
) -> tuple[str, ...]:
    """Alertes : données supposées ou anciennes, paire hors cotation, capital trop petit."""
    warnings = []
    if equity_assumed:
        warnings.append("valeur du portefeuille inconnue : capital apporté utilisé à la place")
    if snapshot_extrapolated:
        warnings.append(
            f"snapshot {snapshot_file} postérieur à la date demandée : règles "
            "actuelles appliquées au passé"
        )
    widest_band = config_decimal(max(config.longterm.rebalance.band_fracs))
    for pair in pairs:
        if not pair.tradable:
            warnings.append(f"{pair.symbol} n'est pas en cotation (statut {pair.status})")
        if pair.delta_min_frac > widest_band:
            warnings.append(
                f"{pair.symbol} : δ_min {pair.delta_min_frac:.1%} > bande la plus large "
                f"{widest_band:.0%} : rééquilibrage impossible à ce capital"
            )
        if max_order < pair.filters.min_notional:
            warnings.append(f"{pair.symbol} : ordre max {max_order} < minNotional")
    dca = config_decimal(config.longterm.dca.amount_quote)
    if pairs and dca < max(p.filters.min_notional for p in pairs):
        warnings.append(f"DCA de {dca} sous le minNotional d'au moins une paire")
    return tuple(warnings)


def compute(
    config: QlabConfig,
    snapshot: Snapshot,
    *,
    at_ms: int,
    net_deposits_quote: Decimal,
    equity_quote: Decimal | None = None,
    snapshot_extrapolated: bool = False,
) -> EffectiveParams:
    """Calcule les paramètres effectifs (fonction pure, aucun accès disque ni réseau).

    ``equity_quote`` : valeur du portefeuille ; ``None`` ⇒ capital apporté (signalé).
    Paire tradée absente du snapshot ⇒ ``DataError`` (config incohérente avec Binance).
    """
    base = config.base
    tier = base.tier_for(net_deposits_quote)  # refuse ≤ 0 ou non-Decimal
    equity = net_deposits_quote if equity_quote is None else equity_quote
    if not isinstance(equity, Decimal) or not equity.is_finite() or equity <= 0:
        raise DataError(f"valeur du portefeuille doit être un Decimal > 0 (reçu {equity!r})")
    pairs = _pairs(config, snapshot, equity)
    max_order = equity * config_decimal(base.risk.max_order_frac)
    fallback = fallback_fees(snapshot.fees)
    return EffectiveParams(
        at_ms=at_ms,
        net_deposits_quote=net_deposits_quote,
        equity_quote=equity,
        equity_assumed=equity_quote is None,
        tier=tier,
        fees_origin=str(snapshot.fees.get("origin", "config")),
        fee_maker_frac=config_decimal(fallback.maker),
        fee_taker_frac=config_decimal(fallback.taker),
        max_daily_loss_quote=equity * config_decimal(base.risk.max_daily_loss_frac),
        max_order_quote=max_order,
        snapshot_file=snapshot.path.name,
        snapshot_extrapolated=snapshot_extrapolated,
        pairs=pairs,
        warnings=_warnings(
            config,
            pairs,
            max_order,
            equity_assumed=equity_quote is None,
            snapshot_file=snapshot.path.name,
            snapshot_extrapolated=snapshot_extrapolated,
        ),
    )


def current(
    config: QlabConfig,
    paths: DataPaths,
    *,
    at_ms: int,
    exchange: str = "binance",
    equity_quote: Decimal | None = None,
) -> EffectiveParams:
    """Paramètres effectifs à ``at_ms`` d'après le registre des apports et les snapshots."""
    net = Ledger(paths.ledger).net_deposits_quote(at_ms)
    if net <= 0:
        raise DataError(
            f"capital apporté au {ms_to_iso(at_ms)} : {format_amount(net)} ; enregistrer un "
            "dépôt d'abord (python -m qlab.core.ledger ... deposit)"
        )
    source = config.base.exchange(exchange).name
    snapshot, extrapolated = SnapshotStore(paths, source).snapshot_at(at_ms)
    return compute(
        config,
        snapshot,
        at_ms=at_ms,
        net_deposits_quote=net,
        equity_quote=equity_quote,
        snapshot_extrapolated=extrapolated,
    )


# --- commande ----------------------------------------------------------------------------


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exchange", default="binance", help="nom dans base.yaml")
    parser.add_argument("--equity", help="valeur actuelle du portefeuille (devise de cotation)")
    parser.add_argument("--date", help="YYYY-MM-DD (fin de journée UTC) ; défaut : maintenant")


def _action(ctx: Context) -> int:
    a = ctx.args
    try:
        at_ms = date_to_ms(a.date) + MS_PER_DAY - 1 if a.date else now_ms()
    except ValueError as exc:
        raise DataError(str(exc)) from exc
    equity = None if a.equity is None else parse_amount(a.equity)
    p = current(ctx.config, ctx.paths, at_ms=at_ms, exchange=a.exchange, equity_quote=equity)
    q = ctx.config.base.symbols.quote_asset
    print(f"Au {ms_to_iso(p.at_ms)} — snapshot {p.snapshot_file}")
    print(
        f"Capital apporté {format_amount(p.net_deposits_quote)} {q} ⇒ palier {p.tier.name} "
        f"(drawdown max {p.tier.max_drawdown_frac:.0%})"
    )
    print(
        f"Portefeuille {format_amount(p.equity_quote)} {q}"
        + (" (supposé = capital apporté)" if p.equity_assumed else "")
    )
    print(f"Frais de repli (config) : maker {p.fee_maker_frac:.4%}, taker {p.fee_taker_frac:.4%}")
    print(
        f"Perte journalière max {p.max_daily_loss_quote:.2f} {q} ; ordre max "
        f"{p.max_order_quote:.2f} {q}"
    )
    for pair in p.pairs:
        min_notional = format_amount(pair.filters.min_notional.normalize())
        print(
            f"  {pair.symbol:<10} {pair.status:<8} minNotional {min_notional} "
            f"⇒ δ_min {pair.delta_min_frac:.1%} | frais maker {pair.fee_maker_frac:.4%} "
            f"taker {pair.fee_taker_frac:.4%} ({pair.fee_origin})"
        )
    for w in p.warnings:
        print(f"ATTENTION : {w}")
    ctx.journal.info(
        "params.computed",
        {
            "at_ms": p.at_ms,
            "tier": p.tier.name,
            "equity_quote": format_amount(p.equity_quote),
            "net_deposits_quote": format_amount(p.net_deposits_quote),
            "snapshot": p.snapshot_file,
            "fees_origin": p.fees_origin,
            "delta_min": {x.symbol: str(x.delta_min_frac) for x in p.pairs},
            "warnings": list(p.warnings),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(
        argv,
        prog="python -m qlab.exchange.effective_params",
        description="Paramètres effectifs (palier, frais, limites, δ_min)",
        component="effective_params",
        add_arguments=_add_arguments,
        action=_action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
