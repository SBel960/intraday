"""Tests de qlab.exchange.lot : filtres BTCEUR réels recopiés, valeurs calculées à la main.

Repères (prix 87 654,32 €, pas 0,00001 BTC, minNotional 5 €) :
- 50 € / 87 654,32 = 0,000570422… BTC → arrondi bas 0,00057 ; notionnel 0,00057 × 87 654,32
  = 49,9629624 € ; reste en cash 50 − 49,9629624 = 0,0370376 €.
- 5 € / 87 654,32 = 0,0000570… → 0,00005 ; notionnel 4,382716 € < 5 € ⇒ rejet : un ordre
  de 5 € pile ne passe pas après arrondi.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from qlab.core.errors import DataError
from qlab.exchange.lot import SymbolFilters, check_order, qty_for_quote, round_to_step

D = Decimal
PX = D("87654.32")

# Filtres BTCEUR d'exchangeInfo (relevés le 2026-09-25), recopiés à la main.
BTCEUR_FILTERS: list[dict[str, Any]] = [
    {
        "filterType": "PRICE_FILTER",
        "minPrice": "0.01000000",
        "maxPrice": "1000000.00000000",
        "tickSize": "0.01000000",
    },
    {
        "filterType": "LOT_SIZE",
        "minQty": "0.00001000",
        "maxQty": "9000.00000000",
        "stepSize": "0.00001000",
    },
    {"filterType": "ICEBERG_PARTS", "limit": 100},
    {
        "filterType": "MARKET_LOT_SIZE",
        "minQty": "0.00000000",
        "maxQty": "6.12719641",
        "stepSize": "0.00000000",
    },
    {
        "filterType": "NOTIONAL",
        "minNotional": "5.00000000",
        "applyMinToMarket": True,
        "maxNotional": "9000000.00000000",
        "applyMaxToMarket": False,
        "avgPriceMins": 5,
    },
]


@pytest.fixture
def btceur() -> SymbolFilters:
    return SymbolFilters.from_binance("BTCEUR", BTCEUR_FILTERS)


# --- parsing -----------------------------------------------------------------------------


def test_parse_binance_filters(btceur: SymbolFilters) -> None:
    assert btceur.tick_size == D("0.01")
    assert btceur.step_size == D("0.00001")
    assert btceur.min_notional == D("5")
    assert btceur.market_max_qty == D("6.12719641")
    assert btceur.market_step_size == 0  # désactivé
    assert btceur.apply_min_to_market is True and btceur.apply_max_to_market is False


def test_market_lot_size_optional() -> None:
    filters = SymbolFilters.from_binance(
        "X", [f for f in BTCEUR_FILTERS if f["filterType"] != "MARKET_LOT_SIZE"]
    )
    assert filters.market_max_qty == 0 and filters.market_step_size == 0


# --- arrondis ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "step", "rounding", "expected"),
    [
        ("87654.321", "0.01", "down", "87654.32"),
        ("87654.321", "0.01", "up", "87654.33"),
        ("87654.32", "0.01", "up", "87654.32"),  # déjà multiple : inchangé
        ("0.000570422", "0.00001", "down", "0.00057"),
        ("0.000570422", "0.00001", "up", "0.00058"),
        ("0.00000999", "0.00001", "down", "0.00000"),
        ("1.5", "0", "down", "1.5"),  # pas nul : pas d'arrondi
        ("0", "0.01", "up", "0.00"),
    ],
)
def test_round_to_step(x: str, step: str, rounding: Any, expected: str) -> None:
    out = round_to_step(D(x), D(step), rounding)
    assert out == D(expected)
    assert str(out) == expected or D(step) == 0  # exposant du pas conservé : "0.00057"


def test_price_rounding_by_side(btceur: SymbolFilters) -> None:
    assert btceur.round_price(D("87654.325"), "BUY") == D("87654.32")  # ne paie pas plus
    assert btceur.round_price(D("87654.321"), "SELL") == D("87654.33")  # ne vend pas moins


# --- ordres ------------------------------------------------------------------------------


def test_qty_for_50_eur(btceur: SymbolFilters) -> None:
    qty, leftover = qty_for_quote(btceur, D("50"), PX, "LIMIT")
    assert qty == D("0.00057")
    assert leftover == D("0.0370376")
    check = check_order(btceur, "BUY", "LIMIT", qty, PX)
    assert check.accepted and check.reason is None
    assert check.notional_quote == D("49.9629624")


def test_five_euros_rejected_after_rounding(btceur: SymbolFilters) -> None:
    qty, _ = qty_for_quote(btceur, D("5"), PX, "LIMIT")
    check = check_order(btceur, "BUY", "LIMIT", qty, PX)
    assert qty == D("0.00005")
    assert check.notional_quote == D("4.3827160")
    assert not check.accepted and check.reason == "notional_below_min"


def test_limit_price_rounded_in_check(btceur: SymbolFilters) -> None:
    check = check_order(btceur, "SELL", "LIMIT", D("0.001234567"), D("87654.321"))
    assert check.price == D("87654.33")
    assert check.qty == D("0.00123")
    assert check.notional_quote == D("0.00123") * D("87654.33")
    assert check.accepted


@pytest.mark.parametrize(
    "case",
    [
        ("BUY", "LIMIT", "0.000009", "87654.32", "qty_zero_after_rounding"),
        ("BUY", "LIMIT", "9000.00001", "1", "qty_above_max"),
        ("BUY", "MARKET", "6.2", "87654.32", "qty_above_max"),  # MARKET_LOT_SIZE.maxQty
        ("BUY", "LIMIT", "1000", "0.001", "price_zero_after_rounding"),  # 0,001 → 0,00
        ("SELL", "LIMIT", "1", "1000000.01", "price_above_max"),
        ("BUY", "MARKET", "0.00005", "87654.32", "notional_below_min"),  # applyMinToMarket
        ("BUY", "LIMIT", "100", "90000.01", "notional_above_max"),
    ],
)
def test_rejections(btceur: SymbolFilters, case: tuple[Any, Any, str, str, str]) -> None:
    side, order_type, qty, price, reason = case
    check = check_order(btceur, side, order_type, D(qty), D(price))
    assert not check.accepted
    assert check.reason == reason


def test_price_below_min() -> None:
    filters = SymbolFilters.from_binance(
        "X",
        [
            {"filterType": "PRICE_FILTER", "minPrice": "1", "maxPrice": "0", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0", "maxQty": "0", "stepSize": "0"},
            {
                "filterType": "NOTIONAL",
                "minNotional": "0",
                "applyMinToMarket": False,
                "maxNotional": "0",
                "applyMaxToMarket": False,
            },
        ],
    )
    assert check_order(filters, "BUY", "LIMIT", D("1"), D("0.99")).reason == "price_below_min"
    # maxPrice / maxQty / maxNotional à 0 : désactivés
    assert check_order(filters, "BUY", "LIMIT", D("10") ** 9, D("10") ** 9).accepted


def test_price_zero_after_rounding_without_min_price() -> None:
    """Sans minPrice, un prix d'achat arrondi à 0,00 doit quand même être rejeté."""
    filters = SymbolFilters.from_binance(
        "X",
        [
            {"filterType": "PRICE_FILTER", "minPrice": "0", "maxPrice": "0", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0", "maxQty": "0", "stepSize": "0"},
            {
                "filterType": "NOTIONAL",
                "minNotional": "0",
                "applyMinToMarket": False,
                "maxNotional": "0",
                "applyMaxToMarket": False,
            },
        ],
    )
    check = check_order(filters, "BUY", "LIMIT", D("1000"), D("0.009"))
    assert check.price == D("0.00")
    assert check.reason == "price_zero_after_rounding"


def test_market_max_notional_not_applied(btceur: SymbolFilters) -> None:
    """applyMaxToMarket = false : un gros ordre marché n'est pas borné par maxNotional."""
    check = check_order(btceur, "BUY", "MARKET", D("6"), D("2000000"))
    assert check.notional_quote == D("12000000")
    assert check.accepted


def test_market_price_not_rounded(btceur: SymbolFilters) -> None:
    check = check_order(btceur, "BUY", "MARKET", D("0.001"), D("87654.321"))
    assert check.price == D("87654.321")  # prix de référence, pas un prix d'ordre


# --- cas limites -------------------------------------------------------------------------


def test_zero_quote_amount(btceur: SymbolFilters) -> None:
    qty, leftover = qty_for_quote(btceur, D("0"), PX, "LIMIT")
    assert qty == 0 and leftover == 0
    assert check_order(btceur, "BUY", "LIMIT", qty, PX).reason == "qty_zero_after_rounding"


def test_exact_min_notional_accepted() -> None:
    filters = SymbolFilters.from_binance(
        "X",
        [
            {"filterType": "PRICE_FILTER", "minPrice": "0", "maxPrice": "0", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0", "maxQty": "0", "stepSize": "0.1"},
            {
                "filterType": "NOTIONAL",
                "minNotional": "5",
                "applyMinToMarket": True,
                "maxNotional": "0",
                "applyMaxToMarket": False,
            },
        ],
    )
    assert check_order(filters, "BUY", "LIMIT", D("0.5"), D("10")).accepted  # 5,00 pile


# --- cas d'erreur ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [D("-1"), D("NaN"), D("Infinity"), 1.5, "abc"])
def test_bad_values(btceur: SymbolFilters, bad: Any) -> None:
    with pytest.raises(DataError):
        check_order(btceur, "BUY", "LIMIT", bad, PX)
    with pytest.raises(DataError):
        check_order(btceur, "BUY", "LIMIT", D("1"), bad)


def test_zero_price_rejected(btceur: SymbolFilters) -> None:
    with pytest.raises(DataError, match="> 0"):
        qty_for_quote(btceur, D("50"), D("0"), "LIMIT")


def test_bad_side_or_type(btceur: SymbolFilters) -> None:
    with pytest.raises(DataError, match="invalides"):
        check_order(btceur, "buy", "LIMIT", D("1"), PX)  # type: ignore[arg-type]
    with pytest.raises(DataError, match="invalides"):
        check_order(btceur, "BUY", "STOP", D("1"), PX)  # type: ignore[arg-type]


def test_missing_filter() -> None:
    with pytest.raises(DataError, match=r"filtres absents.*NOTIONAL"):
        SymbolFilters.from_binance("X", BTCEUR_FILTERS[:2])


def test_missing_field_or_bad_bool() -> None:
    broken = [dict(f) for f in BTCEUR_FILTERS]
    del broken[0]["tickSize"]
    with pytest.raises(DataError, match="champ absent"):
        SymbolFilters.from_binance("X", broken)
    broken = [dict(f) for f in BTCEUR_FILTERS]
    broken[4]["applyMinToMarket"] = "true"
    with pytest.raises(DataError, match="booléen"):
        SymbolFilters.from_binance("X", broken)


def test_negative_filter_rejected() -> None:
    broken = [dict(f) for f in BTCEUR_FILTERS]
    broken[1]["stepSize"] = "-0.001"
    with pytest.raises(DataError, match="stepSize"):
        SymbolFilters.from_binance("X", broken)
