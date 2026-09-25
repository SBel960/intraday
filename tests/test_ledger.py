"""Tests de qlab.core.ledger : sommes à la main, lecture stricte, commande."""

from __future__ import annotations

import fcntl
import os
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest

from qlab.core.errors import DataError
from qlab.core.ledger import Flow, Ledger, main, parse_amount

T = 1_704_067_200_000  # 2024-01-01T00:00:00Z
DAY = 86_400_000


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "meta" / "ledger.jsonl")


# --- cas nominal -------------------------------------------------------------------------


def test_exact_line_and_round_trip(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    flow = Flow(T, "deposit", Decimal("50.00"), "virement")
    ledger.append(flow)
    assert ledger.path.read_bytes() == (
        b'{"amount_fiat":null,"amount_quote":"50.00","fiat":null,"kind":"deposit",'
        b'"note":"virement","ts_ms":1704067200000}\n'
    )
    assert ledger.flows() == (flow,)
    assert ledger.flows()[0].amount_quote == Decimal("50.00")  # pas de flottant


def test_net_deposits_by_hand(tmp_path: Path) -> None:
    """50 + 150,50 − 20,25 = 180,25 ; avant le retrait : 200,50 ; avant tout : 0."""
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T, "deposit", Decimal("50")))
    ledger.append(Flow(T + DAY, "deposit", Decimal("150.50")))
    ledger.append(Flow(T + 2 * DAY, "withdrawal", Decimal("20.25")))
    assert ledger.net_deposits_quote() == Decimal("180.25")
    assert ledger.net_deposits_quote(at_ms=T + DAY) == Decimal("200.50")  # borne incluse
    assert ledger.net_deposits_quote(at_ms=T - 1) == Decimal(0)
    assert [f.signed_quote for f in ledger.flows()] == [
        Decimal("50"),
        Decimal("150.50"),
        Decimal("-20.25"),
    ]


def test_decimal_exactness(tmp_path: Path) -> None:
    """0,1 + 0,2 = 0,3 exactement (en flottant : 0,30000000000000004)."""
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T, "deposit", Decimal("0.1")))
    ledger.append(Flow(T, "deposit", Decimal("0.2")))  # même instant : autorisé
    assert ledger.net_deposits_quote() == Decimal("0.3")


def test_withdrawing_more_than_deposited(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T, "deposit", Decimal("50")))
    ledger.append(Flow(T + 1, "withdrawal", Decimal("60")))  # gains retirés
    assert ledger.net_deposits_quote() == Decimal("-10")


# --- cas limites -------------------------------------------------------------------------


def test_tiny_amount_never_scientific(tmp_path: Path) -> None:
    """Decimal('1E-8') s'écrit 0.00000001 dans le fichier et se relit à l'identique."""
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T, "deposit", Decimal("1E-8")))
    assert b'"amount_quote":"0.00000001"' in ledger.path.read_bytes()
    assert ledger.flows()[0].amount_quote == Decimal("0.00000001")


def test_amount_bounds_accepted() -> None:
    assert parse_amount("999999999999") == Decimal("999999999999")  # 10¹² − 1
    assert parse_amount("0.000000000000000001") == Decimal("1E-18")  # 18 décimales


@pytest.mark.parametrize("amount", [Decimal("1E+12"), Decimal("1E-19"), Decimal("NaN")])
def test_flow_rejects_out_of_bounds(amount: Decimal) -> None:
    with pytest.raises(DataError, match="montant"):
        Flow(T, "deposit", amount)


def test_missing_ledger_is_empty(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.flows() == ()
    assert ledger.net_deposits_quote() == Decimal(0)
    assert not ledger.path.exists()


def test_from_data_root(tmp_path: Path) -> None:
    assert Ledger.from_data_root(tmp_path).path == tmp_path / "meta" / "ledger.jsonl"


def test_unicode_note(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T, "deposit", Decimal("5"), "épargne € septembre"))
    assert ledger.flows()[0].note == "épargne € septembre"


# --- cas d'erreur ------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        ((1.5, "deposit", Decimal("1")), "ts_ms doit être un entier"),
        ((True, "deposit", Decimal("1")), "ts_ms doit être un entier"),
        ((T, "gift", Decimal("1")), "kind doit être"),
        ((T, "deposit", 1.0), "doit être un Decimal"),
        ((T, "deposit", Decimal("-1")), "montant"),
        ((T, "deposit", Decimal("1"), "a\nb"), "une seule ligne"),
    ],
)
def test_bad_flows(args: tuple[object, ...], msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        Flow(*args)  # type: ignore[arg-type]


def test_chronological_order_enforced(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T + DAY, "deposit", Decimal("50")))
    with pytest.raises(DataError, match="antérieur au dernier"):
        ledger.append(Flow(T, "deposit", Decimal("10")))
    assert len(ledger.flows()) == 1  # rien d'écrit


@pytest.mark.parametrize(
    ("content", "msg"),
    [
        (
            b'{"amount_quote":"5","kind":"deposit","amount_fiat":null,"fiat":null,"note":"","ts_ms":1',
            "tronquée",
        ),
        (b"pas du json\n", "JSON illisible"),
        (b'{"amount_quote":"5","kind":"deposit","ts_ms":1}\n', "champs attendus"),
        (
            b'{"amount_quote":5.0,"kind":"deposit","amount_fiat":null,"fiat":null,"note":"","ts_ms":1}\n',
            "pas de flottant",
        ),
        (
            b'{"amount_quote":"-5","kind":"deposit","amount_fiat":null,"fiat":null,"note":"","ts_ms":1}\n',
            "ligne 1 : montant",
        ),
        (
            b'{"amount_quote":"5","kind":"gift","amount_fiat":null,"fiat":null,"note":"","ts_ms":1}\n',
            "kind doit être",
        ),
        (
            b'{"amount_quote":"5","kind":"deposit","amount_fiat":null,"fiat":null,"note":"","ts_ms":2}\n'
            b'{"amount_quote":"5","kind":"deposit","amount_fiat":null,"fiat":null,"note":"","ts_ms":1}\n',
            "ligne 2 : flux antérieur",
        ),
    ],
)
def test_strict_reading(tmp_path: Path, content: bytes, msg: str) -> None:
    ledger = _ledger(tmp_path)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_bytes(content)
    with pytest.raises(DataError, match=msg):
        ledger.flows()


def test_no_append_on_corrupt_ledger(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_bytes(b'{"amount_quote":"5"')  # crash pendant une écriture
    with pytest.raises(DataError, match="tronquée"):
        ledger.append(Flow(T, "deposit", Decimal("1")))
    assert ledger.path.read_bytes() == b'{"amount_quote":"5"'  # intact


# --- montant fiat ------------------------------------------------------------------------


def test_fiat_round_trip(tmp_path: Path) -> None:
    """49,85 USDC reçus pour 50,00 EUR versés : les deux montants sont gardés, exacts."""
    ledger = _ledger(tmp_path)
    flow = Flow(T, "deposit", Decimal("49.85"), "SEPA", Decimal("50.00"), "EUR")
    ledger.append(flow)
    assert b'"amount_fiat":"50.00","amount_quote":"49.85","fiat":"EUR"' in ledger.path.read_bytes()
    assert ledger.flows() == (flow,)
    assert ledger.net_deposits_quote() == Decimal("49.85")  # le capital reste en devise de cotation


@pytest.mark.parametrize(
    ("amount_fiat", "fiat", "msg"),
    [
        (Decimal("50"), None, "les deux ou aucun"),
        (None, "EUR", "les deux ou aucun"),
        (Decimal("50"), "eur", "code ISO"),
        (Decimal("50"), "EURO", "code ISO"),
        (Decimal("0"), "EUR", "montant"),
        (50.0, "EUR", "doit être un Decimal"),
    ],
)
def test_bad_fiat(amount_fiat: object, fiat: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        Flow(T, "deposit", Decimal("1"), "", amount_fiat, fiat)  # type: ignore[arg-type]


def test_fiat_amount_as_float_in_file_rejected(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_bytes(
        b'{"amount_fiat":50.0,"amount_quote":"5","fiat":"EUR","kind":"deposit",'
        b'"note":"","ts_ms":1}\n'
    )
    with pytest.raises(DataError, match="pas de flottant"):
        ledger.flows()


# --- verrou ------------------------------------------------------------------------------


def test_append_waits_for_lock(tmp_path: Path) -> None:
    """Pendant qu'un autre écrivain tient le verrou, append attend ; il écrit après."""
    ledger = _ledger(tmp_path)
    ledger.append(Flow(T, "deposit", Decimal("50")))
    holder = os.open(ledger.path, os.O_RDONLY)
    fcntl.flock(holder, fcntl.LOCK_EX)
    worker = threading.Thread(target=ledger.append, args=(Flow(T + 1, "deposit", Decimal("1")),))
    worker.start()
    time.sleep(0.3)
    try:
        assert worker.is_alive()  # bloqué par le verrou
        assert len(ledger.flows()) == 1  # rien d'écrit pendant l'attente
    finally:
        os.close(holder)  # libère le verrou
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert ledger.net_deposits_quote() == Decimal("51")


# --- commande ----------------------------------------------------------------------------


def test_cli_deposit_and_tiers(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = str(config_dir)
    assert main(["--config", cfg, "deposit", "50", "--date", "2024-01-01", "--note", "départ"]) == 0
    out = capsys.readouterr().out
    assert "2024-01-01T00:00:00.000Z  deposit" in out
    assert "Capital apporté : 50 USDT" in out
    assert "Palier : t0 (drawdown max 30%)" in out

    assert main(["--config", cfg, "deposit", "150", "--date", "2024-02-01"]) == 0
    assert "Palier : t1 (drawdown max 25%)" in capsys.readouterr().out

    assert main(["--config", cfg, "withdrawal", "250", "--date", "2024-03-01"]) == 0
    out = capsys.readouterr().out
    assert "Capital apporté : -50 USDT" in out
    assert "Palier : aucun" in out

    assert main(["--config", cfg, "show"]) == 0
    assert capsys.readouterr().out.count("\n") == 5  # 3 flux + capital + palier


def test_cli_fiat(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--config", str(config_dir), "deposit", "49.85", "--date", "2024-01-01"]
    assert main([*argv, "--fiat-amount", "50.00", "--fiat", "EUR"]) == 0
    assert "+49.85 USDT (50.00 EUR)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "msg"),
    [
        (["deposit", "5", "--fiat-amount", "5"], "les deux ou aucun"),
        (["deposit", "5", "--fiat", "EUR"], "les deux ou aucun"),
        (["deposit", "abc"], "montant illisible"),
        (["deposit", "0"], "fini et > 0"),
        (["deposit", "5", "--date", "2024-13-01"], "YYYY-MM-DD"),
    ],
)
def test_cli_errors(
    config_dir: Path, capsys: pytest.CaptureFixture[str], argv: list[str], msg: str
) -> None:
    assert main(["--config", str(config_dir), *argv]) == 1  # erreur attendue, pas un bug
    err = capsys.readouterr().err
    assert err.startswith("ERREUR (DataError)")
    assert msg in err


def test_cli_chronology_error(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = str(config_dir)
    assert main(["--config", cfg, "deposit", "50", "--date", "2024-02-01"]) == 0
    assert main(["--config", cfg, "deposit", "50", "--date", "2024-01-01"]) == 1
    assert "antérieur au dernier" in capsys.readouterr().err


# --- non-régression du break test (2026-09-25) --------------------------------------------


def test_lone_surrogate_in_note_is_data_error(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(DataError, match="non encodable en UTF-8"):
        ledger.append(Flow(T, "deposit", Decimal("1"), "\ud800"))
    assert ledger.flows() == ()
