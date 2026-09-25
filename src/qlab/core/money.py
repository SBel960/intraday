"""Montants d'argent : ``Decimal`` uniquement, jamais de flottant.

Utilisé par tout ce qui manipule de l'argent (registre des apports, paramètres effectifs, coûts,
rapports, paper trading). Un montant saisi est en notation décimale simple, borné ; il est
toujours réécrit en notation décimale (jamais ``1E-8``).
"""

from __future__ import annotations

import re
from decimal import Decimal

from qlab.core.errors import DataError

MAX_INT_DIGITS = 12  # < 10¹² en devise de cotation : au-delà, c'est une faute de frappe
MAX_DECIMALS = 18
_AMOUNT_RE = re.compile(rf"^\d{{1,{MAX_INT_DIGITS}}}(\.\d{{1,{MAX_DECIMALS}}})?$")


def check_amount(amount: Decimal) -> Decimal:
    """Vérifie un montant : fini, > 0, < 10¹², au plus 18 décimales ; sinon ``DataError``."""
    if not amount.is_finite() or amount <= 0:
        raise DataError(f"montant doit être fini et > 0 : {amount}")
    if amount.adjusted() >= MAX_INT_DIGITS:
        raise DataError(f"montant trop grand (≥ 10^{MAX_INT_DIGITS}) : {amount}")
    exponent = amount.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -MAX_DECIMALS:
        raise DataError(f"montant avec plus de {MAX_DECIMALS} décimales : {amount}")
    return amount


def parse_amount(text: str) -> Decimal:
    """Montant en notation décimale simple (``50``, ``12.34`` ; ni signe ni exposant)."""
    if not _AMOUNT_RE.match(text):
        raise DataError(f"montant illisible (attendu ex. 50 ou 12.34) : {text!r}")
    return check_amount(Decimal(text))


def format_amount(amount: Decimal) -> str:
    """Notation décimale, jamais scientifique : ``0.00000001`` et non ``1E-8``."""
    return format(amount, "f")


def config_decimal(x: float) -> Decimal:
    """Nombre de la config (``float`` YAML : fraction, montant) → ``Decimal`` exact écrit.

    ``Decimal(0.05)`` vaudrait 0.05000000000000000277… ; ``repr`` donne ``0.05``, ce qui est
    exactement la valeur écrite dans le YAML.
    """
    return Decimal(repr(x))
