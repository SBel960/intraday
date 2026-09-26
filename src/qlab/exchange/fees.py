"""Frais appliqués : le **seul** module qui connaît leur format dans un snapshot.

Format (``Snapshot.fees``) ::

    {"origin": "account" | "config",
     "maker_frac", "taker_frac", "bnb_discount_frac", "pay_in_bnb",   # barème de repli (config)
     "effective_maker_frac", "effective_taker_frac",                  # repli après remise BNB
     "by_symbol": {"BTCEUR": {..., "effective_maker_frac", "effective_taker_frac"}}}

``by_symbol`` contient les frais réels du compte pour les paires tradées (clé d'API présente) ;
toute autre paire prend le barème de repli. Les anciens snapshots sans ``by_symbol`` restent
lisibles (repli partout).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlab.core.config import ExchangeConfig
from qlab.core.errors import DataError
from qlab.exchange.account import Account, CommissionRates, load_credentials


@dataclass(frozen=True, slots=True)
class PairFees:
    """Frais réellement payés sur une paire (fractions) et leur origine."""

    maker: float
    taker: float
    origin: str  # "account" (frais réels du compte) ou "config" (barème de repli)


def effective_by_symbol(
    rates: Mapping[str, CommissionRates], exchange: ExchangeConfig
) -> dict[str, PairFees]:
    """Frais effectifs par paire (remise BNB si ``fees.pay_in_bnb`` et active sur le compte)."""
    out = {}
    for symbol, r in rates.items():
        maker, taker = r.effective(exchange.fees.pay_in_bnb)
        out[symbol] = PairFees(maker, taker, "account")
    return out


def fees_record(
    exchange: ExchangeConfig, rates: Mapping[str, CommissionRates] | None = None
) -> dict[str, Any]:
    """Bloc « frais » d'un snapshot : repli de la config + frais réels par paire (si ``rates``)."""
    f = exchange.fees
    effective = effective_by_symbol(rates or {}, exchange)
    by_symbol = {
        symbol: {
            "maker_frac": r.maker,
            "taker_frac": r.taker,
            "tax_maker_frac": r.tax_maker,
            "tax_taker_frac": r.tax_taker,
            "special_maker_frac": r.special_maker,
            "special_taker_frac": r.special_taker,
            "bnb_multiplier": r.bnb_multiplier,
            "bnb_enabled": r.bnb_enabled,
            "effective_maker_frac": effective[symbol].maker,
            "effective_taker_frac": effective[symbol].taker,
        }
        for symbol, r in sorted((rates or {}).items())
    }
    return {
        "origin": "account" if by_symbol else "config",
        "by_symbol": by_symbol,
        "maker_frac": f.maker_frac,
        "taker_frac": f.taker_frac,
        "bnb_discount_frac": f.bnb_discount_frac,
        "pay_in_bnb": f.pay_in_bnb,
        "effective_maker_frac": f.effective_maker_frac,
        "effective_taker_frac": f.effective_taker_frac,
    }


def _rates(block: object, where: str) -> tuple[float, float]:
    """(maker, taker) effectifs d'un bloc ; format invalide : ``DataError`` (jamais KeyError)."""
    if not isinstance(block, Mapping):
        raise DataError(f"frais {where} : bloc attendu, reçu {type(block).__name__}")
    values = []
    for key in ("effective_maker_frac", "effective_taker_frac"):
        v = block.get(key)
        if not isinstance(v, int | float) or isinstance(v, bool) or not 0 <= v < 1:
            raise DataError(f"frais {where} : {key} absent ou hors [0, 1[ ({v!r})")
        values.append(float(v))
    return values[0], values[1]


def fallback_fees(fees: Mapping[str, Any]) -> PairFees:
    """Barème de repli (config) enregistré dans un bloc « frais »."""
    return PairFees(*_rates(fees, "de repli"), "config")


def pair_fees(fees: Mapping[str, Any], symbol: str) -> PairFees:
    """Frais d'une paire lus dans un bloc « frais » : réels si connus, sinon repli."""
    by_symbol = fees.get("by_symbol", {})  # absent (anciens snapshots) : aucun frais réel
    if not isinstance(by_symbol, Mapping):
        raise DataError("frais : by_symbol doit être un bloc paire → frais")
    if symbol in by_symbol:
        return PairFees(*_rates(by_symbol[symbol], symbol), "account")
    return fallback_fees(fees)


def describe(fees: Mapping[str, Any]) -> list[str]:
    """Lignes lisibles : repli, puis frais réels par paire."""
    fallback = fallback_fees(fees)
    lines = [f"Frais de repli (config) : maker {fallback.maker:.4%}, taker {fallback.taker:.4%}"]
    for symbol in sorted(fees.get("by_symbol", {})):
        f = pair_fees(fees, symbol)
        lines.append(f"Frais réels {symbol} : maker {f.maker:.4%}, taker {f.taker:.4%}")
    return lines


def account_rates(
    secrets_file: Path,
    exchange: ExchangeConfig,
    symbols: Sequence[str],
    *,
    server_time_ms: Callable[[], int],
    notify: Callable[[str], None],
) -> dict[str, CommissionRates] | None:
    """Frais réels de ``symbols`` si une clé d'API est configurée ; sinon ``None``.

    La clé est d'abord vérifiée en lecture seule (refusée sinon). Une clé présente mais en
    échec fait échouer l'appel : jamais de retour silencieux au barème de repli.
    """
    creds = load_credentials(secrets_file)
    if creds is None:
        return None
    account = Account(exchange.rest_url, creds, server_time_ms=server_time_ms, notify=notify)
    account.ensure_read_only()
    return {s: account.commission(s) for s in symbols}
