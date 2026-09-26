"""Tests de qlab.exchange.fees : format du bloc « frais », frais par paire, repli."""

from __future__ import annotations

from pathlib import Path

import pytest

from qlab.core.config import ExchangeConfig, load_base
from qlab.core.errors import DataError
from qlab.exchange.account import CommissionRates
from qlab.exchange.fees import account_rates, describe, fees_record, pair_fees

BTC = CommissionRates("BTCUSDT", 0.001, 0.00095, 0.0, 0.0, 0.0, 0.0, 0.75, True)


@pytest.fixture
def exchange(config_dir: Path) -> ExchangeConfig:
    return load_base(config_dir / "base.yaml").exchange("binance")


def test_without_key_everything_falls_back(exchange: ExchangeConfig) -> None:
    fees = fees_record(exchange)
    assert fees["origin"] == "config" and fees["by_symbol"] == {}
    f = pair_fees(fees, "BTCUSDT")
    assert (f.maker, f.taker, f.origin) == (0.001, 0.001, "config")


def test_real_fees_per_pair(exchange: ExchangeConfig) -> None:
    """pay_in_bnb faux (config de test) : frais standard, sans remise."""
    fees = fees_record(exchange, {"BTCUSDT": BTC})
    assert fees["origin"] == "account"
    assert fees["by_symbol"]["BTCUSDT"]["bnb_multiplier"] == 0.75
    real = pair_fees(fees, "BTCUSDT")
    assert (real.maker, real.taker, real.origin) == (0.001, 0.00095, "account")
    other = pair_fees(fees, "ETHUSDT")  # non interrogée : repli
    assert (other.taker, other.origin) == (0.001, "config")


def test_bnb_discount_applied_when_configured(config_dir: Path) -> None:
    path = config_dir / "base.yaml"
    path.write_text(path.read_text().replace("pay_in_bnb: false", "pay_in_bnb: true"))
    fees = fees_record(load_base(path).exchange("binance"), {"BTCUSDT": BTC})
    f = pair_fees(fees, "BTCUSDT")
    assert f.maker == pytest.approx(0.00075) and f.taker == pytest.approx(0.0007125)


def test_old_snapshot_format_still_readable() -> None:
    """Snapshot antérieur à by_symbol : repli partout, sans erreur."""
    old = {"origin": "config", "effective_maker_frac": 0.001, "effective_taker_frac": 0.001}
    assert pair_fees(old, "BTCEUR").origin == "config"
    assert describe(old) == ["Frais de repli (config) : maker 0.1000%, taker 0.1000%"]


def test_describe(exchange: ExchangeConfig) -> None:
    assert describe(fees_record(exchange, {"BTCUSDT": BTC})) == [
        "Frais de repli (config) : maker 0.1000%, taker 0.1000%",
        "Frais réels BTCUSDT : maker 0.1000%, taker 0.0950%",
    ]


def test_account_rates_without_secrets_file(exchange: ExchangeConfig, tmp_path: Path) -> None:
    assert (
        account_rates(
            tmp_path / "absent.env",
            exchange,
            ["BTCUSDT"],
            server_time_ms=lambda: 0,
            notify=lambda _: None,
        )
        is None
    )


# --- non-régression de l'audit (2026-09-26) -----------------------------------------------


@pytest.mark.parametrize(
    "fees",
    [
        {},
        {"effective_maker_frac": 0.001},
        {"effective_maker_frac": "0.001", "effective_taker_frac": 0.001},
        {"effective_maker_frac": 0.001, "effective_taker_frac": 1.5},
        {"effective_maker_frac": 0.001, "effective_taker_frac": True},
        {"effective_maker_frac": 0.001, "effective_taker_frac": 0.001, "by_symbol": []},
        {
            "effective_maker_frac": 0.001,
            "effective_taker_frac": 0.001,
            "by_symbol": {"BTCUSDT": {"effective_maker_frac": 0.001}},
        },
        {"effective_maker_frac": 0.001, "effective_taker_frac": 0.001, "by_symbol": {"BTCUSDT": 3}},
    ],
)
def test_malformed_fees_raise_data_error(fees: dict[str, object]) -> None:
    """Bloc de frais mal formé : erreur claire (DataError), jamais KeyError (« bug »)."""
    with pytest.raises(DataError, match="frais"):
        pair_fees(fees, "BTCUSDT")
        describe(fees)
