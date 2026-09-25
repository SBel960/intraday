"""Contraintes de lot Binance spot : arrondis ``tickSize`` / ``stepSize``, bornes, ``minNotional``.

Tout en ``Decimal`` (jamais de flottant pour un prix ou une quantité). Les filtres viennent
d'``exchangeInfo`` (snapshot versionné par ``exchange_info.py``), jamais du code.

Règles d'arrondi (toujours en défaveur de l'exécution, jamais de l'utilisateur) :
- prix limite d'achat arrondi **vers le bas** au tick, de vente **vers le haut** : on ne paie
  jamais plus, on ne vend jamais moins que le prix voulu ;
- quantité arrondie **vers le bas** au pas : on ne dépense jamais plus que prévu. Le reste
  demeure en cash (``leftover_quote``), ce qui compte à petit capital (0,00001 BTC ≈ 0,9 €).

Un ordre qui viole un filtre n'est pas une exception : c'est un **rejet** avec un code stable
(``RejectReason``), compté par ``lt_costs.py``. Les erreurs de programmation (valeur négative,
flottant) lèvent ``DataError``.

Non couverts : ``PERCENT_PRICE_BY_SIDE`` (bande autour du prix moyen, sans effet sur nos ordres
proches du marché), ``ICEBERG_PARTS``, ``TRAILING_DELTA``, ``MAX_NUM_*`` (non utilisés).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Literal

from qlab.core.errors import DataError

Side = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET"]
Rounding = Literal["down", "up"]
RejectReason = Literal[
    "price_zero_after_rounding",
    "qty_zero_after_rounding",
    "qty_below_min",
    "qty_above_max",
    "price_below_min",
    "price_above_max",
    "notional_below_min",
    "notional_above_max",
]
ZERO = Decimal(0)


def _dec(name: str, value: object, *, positive: bool = False) -> Decimal:
    """Decimal fini ≥ 0 (> 0 si ``positive``) ; accepte un texte Binance (``"0.01000000"``)."""
    if isinstance(value, str):
        try:
            value = Decimal(value)
        except InvalidOperation as exc:
            raise DataError(f"{name} : nombre illisible {value!r}") from exc
    if not isinstance(value, Decimal):
        raise DataError(f"{name} : Decimal ou texte attendu, reçu {type(value).__name__}")
    if not value.is_finite() or value < 0 or (positive and value == 0):
        raise DataError(f"{name} doit être fini et {'> 0' if positive else '≥ 0'} : {value}")
    return value


def round_to_step(x: Decimal, step: Decimal, rounding: Rounding) -> Decimal:
    """Multiple de ``step`` le plus proche de ``x`` vers le bas ou le haut. ``step = 0`` : ``x``."""
    _dec("x", x)
    _dec("step", step)
    if step == 0:
        return x
    mode = ROUND_FLOOR if rounding == "down" else ROUND_CEILING
    return ((x / step).to_integral_value(rounding=mode) * step).quantize(step)


@dataclass(frozen=True, slots=True)
class SymbolFilters:
    """Filtres d'une paire. Une valeur 0 désactive la contrainte (convention Binance), sauf
    ``step_size`` / ``tick_size`` à 0 qui signifient « pas d'arrondi »."""

    symbol: str
    tick_size: Decimal
    min_price: Decimal
    max_price: Decimal
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    market_step_size: Decimal
    market_min_qty: Decimal
    market_max_qty: Decimal
    min_notional: Decimal
    max_notional: Decimal
    apply_min_to_market: bool
    apply_max_to_market: bool

    def __post_init__(self) -> None:
        for name in (
            "tick_size",
            "min_price",
            "max_price",
            "step_size",
            "min_qty",
            "max_qty",
            "market_step_size",
            "market_min_qty",
            "market_max_qty",
            "min_notional",
            "max_notional",
        ):
            _dec(name, getattr(self, name))

    @classmethod
    def from_binance(cls, symbol: str, filters: Sequence[Mapping[str, object]]) -> SymbolFilters:
        """Construit depuis la liste ``filters`` d'``exchangeInfo`` ; filtre manquant : erreur."""
        by_type = {f.get("filterType"): f for f in filters}
        missing = [t for t in ("PRICE_FILTER", "LOT_SIZE", "NOTIONAL") if t not in by_type]
        if missing:
            raise DataError(f"{symbol} : filtres absents d'exchangeInfo : {missing}")
        price, lot, notional = by_type["PRICE_FILTER"], by_type["LOT_SIZE"], by_type["NOTIONAL"]
        market = by_type.get("MARKET_LOT_SIZE", {"minQty": "0", "maxQty": "0", "stepSize": "0"})
        for key in ("applyMinToMarket", "applyMaxToMarket"):
            if not isinstance(notional.get(key), bool):
                raise DataError(f"{symbol} : NOTIONAL.{key} booléen attendu")
        try:
            return cls(
                symbol=symbol,
                tick_size=_dec("tickSize", price["tickSize"]),
                min_price=_dec("minPrice", price["minPrice"]),
                max_price=_dec("maxPrice", price["maxPrice"]),
                step_size=_dec("stepSize", lot["stepSize"]),
                min_qty=_dec("minQty", lot["minQty"]),
                max_qty=_dec("maxQty", lot["maxQty"]),
                market_step_size=_dec("MARKET_LOT_SIZE.stepSize", market["stepSize"]),
                market_min_qty=_dec("MARKET_LOT_SIZE.minQty", market["minQty"]),
                market_max_qty=_dec("MARKET_LOT_SIZE.maxQty", market["maxQty"]),
                min_notional=_dec("minNotional", notional["minNotional"]),
                max_notional=_dec("maxNotional", notional["maxNotional"]),
                apply_min_to_market=bool(notional["applyMinToMarket"]),
                apply_max_to_market=bool(notional["applyMaxToMarket"]),
            )
        except KeyError as exc:
            raise DataError(f"{symbol} : champ absent d'un filtre : {exc}") from exc

    def round_price(self, price: Decimal, side: Side) -> Decimal:
        """Prix limite au tick : achat vers le bas, vente vers le haut."""
        return round_to_step(
            _dec("price", price, positive=True), self.tick_size, "down" if side == "BUY" else "up"
        )

    def round_qty(self, qty: Decimal, order_type: OrderType) -> Decimal:
        """Quantité vers le bas au pas (LOT_SIZE, puis MARKET_LOT_SIZE pour un ordre marché)."""
        q = round_to_step(_dec("qty", qty), self.step_size, "down")
        if order_type == "MARKET":
            q = round_to_step(q, self.market_step_size, "down")
        return q


@dataclass(frozen=True, slots=True)
class OrderCheck:
    """Ordre après arrondis. ``reason`` est ``None`` si accepté.

    ``notional_quote`` = qty × price (devise de cotation) ; pour un ordre marché, ``price`` est
    le prix de référence fourni (le prix d'exécution réel peut différer).
    """

    accepted: bool
    side: Side
    order_type: OrderType
    qty: Decimal
    price: Decimal
    notional_quote: Decimal
    reason: RejectReason | None


def check_order(
    filters: SymbolFilters, side: Side, order_type: OrderType, qty: Decimal, price: Decimal
) -> OrderCheck:
    """Arrondit ``qty`` (et ``price`` si LIMIT) puis vérifie tous les filtres couverts.

    ``price`` : prix limite (LIMIT) ou prix de référence pour le notionnel (MARKET).
    """
    if side not in ("BUY", "SELL") or order_type not in ("LIMIT", "MARKET"):
        raise DataError(f"side / order_type invalides : {side!r} / {order_type!r}")
    limit = order_type == "LIMIT"
    p = filters.round_price(price, side) if limit else _dec("price", price, positive=True)
    q = filters.round_qty(qty, order_type)
    notional = q * p

    f = filters
    market_min = f.apply_min_to_market
    market_max = f.apply_max_to_market
    # Premier filtre violé, dans l'ordre : prix, quantité, bornes de prix, notionnel.
    checks: tuple[tuple[bool, RejectReason], ...] = (
        (limit and p == 0, "price_zero_after_rounding"),
        (q == 0, "qty_zero_after_rounding"),
        (q < f.min_qty or (not limit and q < f.market_min_qty), "qty_below_min"),
        (
            bool(f.max_qty and q > f.max_qty)
            or bool(not limit and f.market_max_qty and q > f.market_max_qty),
            "qty_above_max",
        ),
        (limit and bool(f.min_price) and p < f.min_price, "price_below_min"),
        (limit and bool(f.max_price) and p > f.max_price, "price_above_max"),
        ((limit or market_min) and notional < f.min_notional, "notional_below_min"),
        (
            bool(f.max_notional) and (limit or market_max) and notional > f.max_notional,
            "notional_above_max",
        ),
    )
    reason = next((r for violated, r in checks if violated), None)
    return OrderCheck(reason is None, side, order_type, q, p, notional, reason)


def qty_for_quote(
    filters: SymbolFilters, quote_amount: Decimal, price: Decimal, order_type: OrderType
) -> tuple[Decimal, Decimal]:
    """Quantité achetable avec ``quote_amount`` au prix ``price``, arrondie vers le bas.

    Retourne ``(qty, leftover_quote)`` : ``leftover_quote`` = montant non investi à cause du pas.
    """
    amount = _dec("quote_amount", quote_amount)
    p = _dec("price", price, positive=True)
    qty = filters.round_qty(amount / p, order_type)
    return qty, amount - qty * p
