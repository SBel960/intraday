"""Tests de qlab.core.money : saisie, bornes, écriture décimale, conversion depuis la config."""

from __future__ import annotations

from decimal import Decimal

import pytest

from qlab.core.errors import DataError
from qlab.core.money import check_amount, config_decimal, format_amount, parse_amount


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "0.00",
        "-5",
        "+5",
        "abc",
        "NaN",
        "Infinity",
        "",
        " 5",
        "5.",
        ".5",
        "1,5",
        "1e2",
        "1e999999999999",  # Decimal le juge fini : refusé par la notation
        "1000000000000",  # 13 chiffres : ≥ 10¹²
        "0.0000000000000000001",  # 19 décimales
    ],
)
def test_bad_amounts(text: str) -> None:
    with pytest.raises(DataError, match="montant"):
        parse_amount(text)


def test_amount_bounds_accepted() -> None:
    assert parse_amount("999999999999") == Decimal("999999999999")  # 10¹² − 1
    assert parse_amount("0.000000000000000001") == Decimal("1E-18")  # 18 décimales


def test_format_never_scientific() -> None:
    assert format_amount(Decimal("1E-8")) == "0.00000001"
    assert format_amount(Decimal("1E+3")) == "1000"
    assert format_amount(Decimal("-50")) == "-50"


@pytest.mark.parametrize("amount", [Decimal("1E+12"), Decimal("1E-19"), Decimal("NaN"), Decimal(0)])
def test_check_amount_bounds(amount: Decimal) -> None:
    with pytest.raises(DataError, match="montant"):
        check_amount(amount)


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.05, "0.05"), (0.001, "0.001"), (10.0, "10.0"), (0.1 + 0.2, "0.30000000000000004")],
)
def test_config_decimal_is_the_written_value(x: float, expected: str) -> None:
    """0.05 écrit dans le YAML donne exactement 0.05 (pas 0.05000000000000000277…)."""
    assert config_decimal(x) == Decimal(expected)
    assert Decimal(0.05) != Decimal("0.05")  # noqa: RUF032 — démontre le piège évité
