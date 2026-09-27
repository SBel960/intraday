"""Interface d'ordres : un seul contrat pour le paper trading et, plus tard, le réel.

``live/paper.py`` ne connaît que ``Broker`` (``balances``, ``submit``) : passer du paper au réel
ne changera pas sa logique.

- ``PaperBroker`` : exécution simulée avec **le même modèle que le backtest** (``PairRules`` de
  ``lt_backtest.pair_rules`` : prix de référence ± impact, filtres de lot de ``exchange/lot.py``,
  frais taker réels prélevés en devise de cotation). Un test vérifie qu'il exécute exactement
  comme ``lt_backtest.run`` : les résultats du paper se comparent à ceux du backtest (LT.6).
  Le prix de référence est fourni par l'appelant (ouverture de la barre suivante, comme en
  backtest). Un ordre refusé ne change rien au portefeuille et dit pourquoi.
- **Réel impossible** : ``open_broker("live", ...)`` refuse toujours (``LiveDisabledError``). Il
  énumère ce qui manquerait (réglage explicite ``allow_live``, droits de la clé vérifiés :
  trading spot seul, ni retrait ni transfert, clé restreinte à une IP), puis refuse encore :
  l'exécution réelle ne sera écrite qu'après 2 à 3 mois de paper trading concluants (LT.7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol

from qlab.core.errors import DataError, QlabError
from qlab.exchange.account import FORBIDDEN_RIGHTS
from qlab.exchange.lot import Side, check_order
from qlab.longterm.lt_backtest import PairRules

ZERO = Decimal(0)
TRADE_RIGHT = "enableSpotAndMarginTrading"
# Droits interdits en réel : tout ce que refuse la lecture seule, sauf le trading spot.
LIVE_FORBIDDEN = tuple(r for r in FORBIDDEN_RIGHTS if r != TRADE_RIGHT)
NOT_BUILT = "exécution réelle non écrite : elle vient après 2 à 3 mois de paper trading concluants"


class LiveDisabledError(QlabError):
    """Le réel a été demandé : refus, avec la liste de ce qui manque."""


@dataclass(frozen=True, slots=True)
class Order:
    """Ordre au marché. ``qty`` : quantité voulue, arrondie au pas à l'exécution ;
    ``ref_price`` : prix de référence (ouverture de la barre), avant impact."""

    symbol: str
    side: Side
    qty: Decimal
    ref_price: Decimal


@dataclass(frozen=True, slots=True)
class Execution:
    """Résultat d'un ordre ; refusé ⇒ ``qty``, ``notional``, ``fee`` nuls et ``reason``."""

    order: Order
    ts_ms: int
    accepted: bool
    qty: Decimal
    price: Decimal
    notional: Decimal
    fee: Decimal
    reason: str | None = None


class Broker(Protocol):
    def balances(self) -> dict[str, Decimal]: ...

    def submit(self, order: Order, *, ts_ms: int) -> Execution: ...


class PaperBroker:
    """Portefeuille simulé : ``cash`` en devise ``quote``, ``holdings`` en quantités d'actif
    (clé : la paire, ex. ``BTCEUR``). ``live/paper.py`` le reconstruit depuis son journal."""

    def __init__(
        self,
        quote: str,
        cash: Decimal,
        rules: Mapping[str, PairRules],
        holdings: Mapping[str, Decimal] | None = None,
    ) -> None:
        held = dict(holdings or {})
        if cash < 0 or any(q < 0 for q in held.values()):
            raise DataError("solde négatif : cash et quantités ≥ 0 attendus")
        unknown = sorted(set(held) - set(rules))
        if unknown:
            raise DataError(f"positions sans règles d'ordre : {unknown}")
        self.quote, self._cash, self._rules = quote, cash, dict(rules)
        self._held = {s: held.get(s, ZERO) for s in self._rules}

    def balances(self) -> dict[str, Decimal]:
        return {self.quote: self._cash, **self._held}

    def submit(self, order: Order, *, ts_ms: int) -> Execution:
        rules = self._rules.get(order.symbol)
        if rules is None:
            raise DataError(f"{order.symbol} : aucune règle d'ordre (paire hors univers)")
        if order.qty <= 0 or order.ref_price <= 0:
            raise DataError(f"ordre invalide : quantité et prix > 0 attendus ({order})")
        buy = order.side == "BUY"
        price = order.ref_price * (1 + rules.impact_frac if buy else 1 - rules.impact_frac)
        check = check_order(rules.filters, order.side, "MARKET", order.qty, price)
        if not check.accepted:
            return _refused(order, ts_ms, price, str(check.reason))
        notional = check.qty * price
        fee = notional * rules.fee_frac
        held = self._held[order.symbol]
        if (buy and notional + fee > self._cash) or (not buy and check.qty > held):
            return _refused(order, ts_ms, price, "insufficient_balance")
        sign = 1 if buy else -1
        self._cash -= sign * notional + fee
        self._held[order.symbol] = held + sign * check.qty
        return Execution(order, ts_ms, True, check.qty, price, notional, fee)


def _refused(order: Order, ts_ms: int, price: Decimal, reason: str) -> Execution:
    return Execution(order, ts_ms, False, ZERO, price, ZERO, ZERO, reason)


def live_blockers(allow_live: bool, rights: Mapping[str, Any] | None) -> list[str]:
    """Ce qui interdit le réel, hors exécution non écrite. ``rights`` : réponse de
    ``/sapi/v1/account/apiRestrictions`` (``Account.signed_get``), ``None`` si non vérifiés."""
    out = [] if allow_live else ["réglage explicite du réel absent (allow_live)"]
    if rights is None:
        return [*out, "droits de la clé non vérifiés"]
    if rights.get(TRADE_RIGHT) is not True:
        out.append("clé sans droit de trading spot")
    extra = [r for r in LIVE_FORBIDDEN if rights.get(r)]
    if extra:
        out.append(f"clé avec des droits interdits : {', '.join(extra)}")
    if rights.get("ipRestrict") is not True:
        out.append("clé non restreinte à l'adresse IP de ce PC")
    return out


def open_broker(
    mode: Literal["paper", "live"],
    *,
    quote: str,
    cash: Decimal,
    rules: Mapping[str, PairRules],
    allow_live: bool = False,
    rights: Mapping[str, Any] | None = None,
) -> Broker:
    """Paper : ``PaperBroker``. Réel : toujours ``LiveDisabledError`` (voir l'en-tête)."""
    if mode == "paper":
        return PaperBroker(quote, cash, rules)
    if mode != "live":
        raise DataError(f"mode inconnu : {mode!r} (paper ou live)")
    reasons = [*live_blockers(allow_live, rights), NOT_BUILT]
    raise LiveDisabledError("réel impossible : " + " ; ".join(reasons))
