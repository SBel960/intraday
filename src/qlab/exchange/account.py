"""Compte Binance en **lecture seule** : signature Ed25519, droits de la clé, frais réels.

Identifiants (``base.secrets_file``, chmod 600) : ``BINANCE_API_KEY`` (identifiant public de la
clé) et ``BINANCE_PRIVATE_KEY_PATH`` (clé privée Ed25519 au format PEM, chmod 600, générée sur
le PC : elle ne quitte jamais la machine ; Binance ne connaît que la clé publique).

Sécurité : ``ensure_read_only`` refuse toute clé qui aurait un droit autre que la lecture
(trading, retrait, transfert, marge, contrats à terme…), quel que soit le réglage chez Binance.

Horodatage : Binance refuse une requête datée de plus d'1 s dans le futur ; l'horloge du PC
peut dériver. On signe donc avec l'**heure du serveur**, fournie par l'appelant
(``server_time_ms``, dérivée de la mesure d'horloge d'``exchange_info``).
"""

from __future__ import annotations

import base64
import json
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from qlab.core import http
from qlab.core.errors import DataError, ExchangeError
from qlab.core.secrets import Secret, check_private, load_env_file, require

KEY_VAR = "BINANCE_API_KEY"
PEM_VAR = "BINANCE_PRIVATE_KEY_PATH"
RECV_WINDOW_MS = 5_000  # fenêtre de validité d'une requête signée (défaut Binance)
COMMISSION_ENDPOINT = "/api/v3/account/commission"
RESTRICTIONS_ENDPOINT = "/sapi/v1/account/apiRestrictions"
# Tout droit autre que la lecture : la clé est refusée s'il en a un seul.
FORBIDDEN_RIGHTS = (
    "enableSpotAndMarginTrading",
    "enableWithdrawals",
    "enableInternalTransfer",
    "permitsUniversalTransfer",
    "enableMargin",
    "enableFutures",
    "enableVanillaOptions",
    "enablePortfolioMarginTrading",
    "enableFixApiTrade",
)


@dataclass(frozen=True, slots=True)
class Credentials:
    api_key: Secret
    private_key: Ed25519PrivateKey

    def __repr__(self) -> str:
        return "Credentials(***)"


def load_credentials(secrets_file: Path) -> Credentials | None:
    """Identifiants depuis ``secrets_file`` ; ``None`` si le fichier n'existe pas (repli)."""
    if not secrets_file.exists():
        return None
    values = load_env_file(secrets_file)
    api_key = require(values, KEY_VAR, secrets_file)
    pem_path = Path(require(values, PEM_VAR, secrets_file).reveal())
    check_private(pem_path)
    try:
        key = serialization.load_pem_private_key(pem_path.read_bytes(), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise DataError(f"clé privée illisible : {pem_path}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise DataError(f"{pem_path} : clé Ed25519 attendue (clé « auto-générée » Binance)")
    return Credentials(api_key, key)


@dataclass(frozen=True, slots=True)
class CommissionRates:
    """Frais d'une paire pour ce compte (fractions). ``bnb_multiplier`` : part payée si les
    frais sont réglés en BNB (0.75 = remise de 25 %) ; ``bnb_enabled`` : remise active."""

    symbol: str
    maker: float
    taker: float
    tax_maker: float
    tax_taker: float
    special_maker: float
    special_taker: float
    bnb_multiplier: float
    bnb_enabled: bool

    def effective(self, pay_in_bnb: bool) -> tuple[float, float]:
        """(maker, taker) réellement payés : standard (× remise BNB) + taxe + spécial."""
        mult = self.bnb_multiplier if pay_in_bnb and self.bnb_enabled else 1.0
        return (
            self.maker * mult + self.tax_maker + self.special_maker,
            self.taker * mult + self.tax_taker + self.special_taker,
        )


def _frac(doc: Mapping[str, Any], group: str, side: str) -> float:
    try:
        value = float(doc[group][side])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExchangeError(f"frais du compte : {group}.{side} absent ou illisible") from exc
    if not 0 <= value < 1:
        raise ExchangeError(f"frais du compte hors bornes : {group}.{side} = {value}")
    return value


def parse_commission(doc: Mapping[str, Any]) -> CommissionRates:
    """Réponse de ``/api/v3/account/commission`` → ``CommissionRates`` (validée)."""
    discount = doc.get("discount")
    if not isinstance(discount, dict):
        raise ExchangeError("frais du compte : bloc « discount » absent")
    try:
        mult = float(discount["discount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExchangeError("frais du compte : discount.discount absent ou illisible") from exc
    if not 0 < mult <= 1:
        raise ExchangeError(f"frais du compte : multiplicateur BNB hors ]0, 1] ({mult})")
    return CommissionRates(
        symbol=str(doc.get("symbol")),
        maker=_frac(doc, "standardCommission", "maker"),
        taker=_frac(doc, "standardCommission", "taker"),
        tax_maker=_frac(doc, "taxCommission", "maker"),
        tax_taker=_frac(doc, "taxCommission", "taker"),
        special_maker=_frac(doc, "specialCommission", "maker"),
        special_taker=_frac(doc, "specialCommission", "taker"),
        bnb_multiplier=mult,
        bnb_enabled=bool(discount.get("enabledForAccount"))
        and bool(discount.get("enabledForSymbol")),
    )


def _http_get(url: str, **options: Any) -> http.HttpResult:
    """``core/http.get`` retrouvé à chaque appel (et non figé à l'import) : substituable."""
    return http.get(url, **options)


class Account:
    """Accès signé en lecture seule. ``get`` : défaut ``core/http.get`` (injectable, tests)."""

    def __init__(
        self,
        rest_url: str,
        credentials: Credentials,
        *,
        server_time_ms: Callable[[], int],
        notify: Callable[[str], None],
        get: Callable[..., http.HttpResult] | None = None,
    ) -> None:
        self._rest_url, self._creds = rest_url.rstrip("/"), credentials
        self._server_time_ms, self._notify = server_time_ms, notify
        self._get = get or _http_get

    def signed_get(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        query = urllib.parse.urlencode(
            {**(params or {}), "timestamp": self._server_time_ms(), "recvWindow": RECV_WINDOW_MS}
        )
        signature = base64.b64encode(self._creds.private_key.sign(query.encode("ascii")))
        url = (
            f"{self._rest_url}{path}?{query}&signature="
            f"{urllib.parse.quote(signature.decode('ascii'), safe='')}"
        )
        result = self._get(
            url, headers={"X-MBX-APIKEY": self._creds.api_key.reveal()}, notify=self._notify
        )
        try:
            return json.loads(result.body)
        except ValueError as exc:
            raise ExchangeError(f"réponse non JSON sur {path}") from exc

    def ensure_read_only(self) -> None:
        """Refuse la clé si elle peut faire autre chose que lire (``ExchangeError``)."""
        rights = self.signed_get(RESTRICTIONS_ENDPOINT)
        if not isinstance(rights, dict) or rights.get("enableReading") is not True:
            raise ExchangeError("clé API sans droit de lecture ou réponse inattendue")
        extra = [r for r in FORBIDDEN_RIGHTS if rights.get(r)]
        if extra:
            raise ExchangeError(
                f"clé API refusée : elle a des droits au-delà de la lecture ({', '.join(extra)}) ;"
                " ne garder que « Activer la lecture » chez Binance"
            )

    def commission(self, symbol: str) -> CommissionRates:
        rates = parse_commission(self.signed_get(COMMISSION_ENDPOINT, {"symbol": symbol}))
        if rates.symbol != symbol:
            raise ExchangeError(f"frais demandés pour {symbol}, reçus pour {rates.symbol}")
        return rates
