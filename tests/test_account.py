"""Tests de qlab.exchange.account : identifiants, signature Ed25519, droits, frais — sans réseau."""

from __future__ import annotations

import base64
import json
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from qlab.core.errors import DataError, ExchangeError
from qlab.core.http import HttpResult
from qlab.core.secrets import Secret
from qlab.exchange.account import (
    FORBIDDEN_RIGHTS,
    Account,
    Credentials,
    load_credentials,
    parse_commission,
)

SERVER_MS = 1_790_380_000_000
# Réponse réelle de /api/v3/account/commission pour BTCEUR (relevée le 2026-09-26).
BTCEUR_COMMISSION = {
    "symbol": "BTCEUR",
    "standardCommission": {
        "maker": "0.00100000",
        "taker": "0.00095000",
        "buyer": "0",
        "seller": "0",
    },
    "specialCommission": {
        "maker": "0.00000000",
        "taker": "0.00000000",
        "buyer": "0",
        "seller": "0",
    },
    "taxCommission": {"maker": "0.00000000", "taker": "0.00000000", "buyer": "0", "seller": "0"},
    "discount": {
        "enabledForAccount": True,
        "enabledForSymbol": True,
        "discountAsset": "BNB",
        "discount": "0.75000000",
    },
}
READ_ONLY = {"ipRestrict": False, "enableReading": True, **dict.fromkeys(FORBIDDEN_RIGHTS, False)}


def _write_private(path: Path, data: bytes | str) -> Path:
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    path.chmod(0o600)
    return path


def _pem(key: Ed25519PrivateKey | ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _secrets(
    tmp_path: Path, key: Ed25519PrivateKey | ec.EllipticCurvePrivateKey, api_key: str = "APIKEY123"
) -> Path:
    pem = _write_private(tmp_path / "k.pem", _pem(key))
    return _write_private(
        tmp_path / ".env", f"BINANCE_API_KEY={api_key}\nBINANCE_PRIVATE_KEY_PATH={pem}\n"
    )


class _FakeGet:
    """Remplace ``http.get`` : enregistre URL et en-têtes, répond ``responses[path]``."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Any]] = []

    def __call__(self, url: str, **kw: Any) -> HttpResult:
        self.calls.append((url, kw.get("headers")))
        path = urllib.parse.urlsplit(url).path
        body = self.responses[path]
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return HttpResult(raw, {}, 0, 0, 1)


def _account(key: Ed25519PrivateKey, responses: dict[str, Any]) -> tuple[Account, _FakeGet]:
    get = _FakeGet(responses)
    creds = Credentials(Secret("APIKEY123"), key)
    return Account(
        "https://api.example",
        creds,
        server_time_ms=lambda: SERVER_MS,
        notify=lambda _: None,
        get=get,
    ), get


# --- identifiants ------------------------------------------------------------------------


def test_no_secrets_file_means_fallback(tmp_path: Path) -> None:
    assert load_credentials(tmp_path / "absent.env") is None


def test_load_credentials(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    creds = load_credentials(_secrets(tmp_path, key))
    assert creds is not None
    assert creds.api_key.reveal() == "APIKEY123"
    assert repr(creds) == "Credentials(***)" and "APIKEY123" not in repr(creds)


def test_private_key_must_be_private(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    secrets = _secrets(tmp_path, key)
    (tmp_path / "k.pem").chmod(0o644)
    with pytest.raises(DataError, match="chmod 600"):
        load_credentials(secrets)


def test_non_ed25519_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Ed25519 attendue"):
        load_credentials(_secrets(tmp_path, ec.generate_private_key(ec.SECP256R1())))


def test_garbage_key_rejected(tmp_path: Path) -> None:
    pem = _write_private(tmp_path / "k.pem", "pas une clé")
    env = _write_private(tmp_path / ".env", f"BINANCE_API_KEY=x\nBINANCE_PRIVATE_KEY_PATH={pem}\n")
    with pytest.raises(DataError, match="illisible"):
        load_credentials(env)


def test_missing_variable(tmp_path: Path) -> None:
    env = _write_private(tmp_path / ".env", "BINANCE_API_KEY=x\n")
    with pytest.raises(DataError, match="BINANCE_PRIVATE_KEY_PATH absent"):
        load_credentials(env)


# --- signature ---------------------------------------------------------------------------


def test_signed_request_is_verifiable(key: Ed25519PrivateKey) -> None:
    """La signature couvre exactement la chaîne de requête, horodatée à l'heure du serveur."""
    account, get = _account(key, {"/x": {"ok": True}})
    assert account.signed_get("/x", {"symbol": "BTCEUR"}) == {"ok": True}
    url, headers = get.calls[0]
    query = urllib.parse.urlsplit(url).query
    payload, _, sig = query.rpartition("&signature=")
    assert payload == f"symbol=BTCEUR&timestamp={SERVER_MS}&recvWindow=5000"
    key.public_key().verify(base64.b64decode(urllib.parse.unquote(sig)), payload.encode())
    assert headers == {"X-MBX-APIKEY": "APIKEY123"}


def test_non_json_response(key: Ed25519PrivateKey) -> None:
    account, _ = _account(key, {"/x": b"<html>"})
    with pytest.raises(ExchangeError, match="non JSON"):
        account.signed_get("/x")


# --- droits de la clé --------------------------------------------------------------------


def test_read_only_key_accepted(key: Ed25519PrivateKey) -> None:
    account, _ = _account(key, {"/sapi/v1/account/apiRestrictions": READ_ONLY})
    account.ensure_read_only()


@pytest.mark.parametrize("right", FORBIDDEN_RIGHTS)
def test_any_extra_right_refuses_the_key(key: Ed25519PrivateKey, right: str) -> None:
    account, _ = _account(key, {"/sapi/v1/account/apiRestrictions": {**READ_ONLY, right: True}})
    with pytest.raises(ExchangeError, match=f"clé API refusée .*{right}"):
        account.ensure_read_only()


@pytest.mark.parametrize("body", [{**READ_ONLY, "enableReading": False}, ["liste"], {}])
def test_unexpected_rights_response(key: Ed25519PrivateKey, body: Any) -> None:
    account, _ = _account(key, {"/sapi/v1/account/apiRestrictions": body})
    with pytest.raises(ExchangeError, match="sans droit de lecture"):
        account.ensure_read_only()


# --- frais -------------------------------------------------------------------------------


def test_real_commission_by_hand() -> None:
    """Frais réels BTCEUR : maker 0,100 %, taker 0,095 % ; en BNB × 0,75 ⇒ 0,075 % et 0,07125 %."""
    r = parse_commission(BTCEUR_COMMISSION)
    assert (r.maker, r.taker, r.bnb_multiplier, r.bnb_enabled) == (0.001, 0.00095, 0.75, True)
    assert r.effective(pay_in_bnb=False) == (0.001, 0.00095)
    maker, taker = r.effective(pay_in_bnb=True)
    assert maker == pytest.approx(0.00075) and taker == pytest.approx(0.0007125)


def test_tax_and_special_are_added() -> None:
    doc = json.loads(json.dumps(BTCEUR_COMMISSION))
    doc["taxCommission"]["taker"] = "0.0001"
    doc["specialCommission"]["taker"] = "0.00005"
    assert parse_commission(doc).effective(False)[1] == pytest.approx(0.0011)


def test_discount_disabled_for_symbol() -> None:
    doc = json.loads(json.dumps(BTCEUR_COMMISSION))
    doc["discount"]["enabledForSymbol"] = False
    assert parse_commission(doc).effective(pay_in_bnb=True) == (0.001, 0.00095)


@pytest.mark.parametrize(
    ("patch", "msg"),
    [
        ({"discount": None}, "discount"),
        ({"standardCommission": {"maker": "abc", "taker": "0"}}, "illisible"),
        ({"standardCommission": {"maker": "1.5", "taker": "0"}}, "hors bornes"),
        (
            {"discount": {"discount": "0", "enabledForAccount": True, "enabledForSymbol": True}},
            "multiplicateur",
        ),
    ],
)
def test_bad_commission(patch: dict[str, Any], msg: str) -> None:
    with pytest.raises(ExchangeError, match=msg):
        parse_commission({**BTCEUR_COMMISSION, **patch})


def test_commission_symbol_mismatch(key: Ed25519PrivateKey) -> None:
    account, _ = _account(key, {"/api/v3/account/commission": BTCEUR_COMMISSION})
    assert account.commission("BTCEUR").symbol == "BTCEUR"
    with pytest.raises(ExchangeError, match="reçus pour BTCEUR"):
        account.commission("ETHEUR")
