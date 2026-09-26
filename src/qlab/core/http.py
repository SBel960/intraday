"""Client HTTP commun : attend le retour du réseau, respecte les limites de requêtes de Binance.

Tout accès réseau du projet (``exchangeInfo``, archives, bougies REST…) passe par ``get`` :
une seule politique de relance, testée une fois.

- réseau coupé, délai dépassé, HTTP 5xx : nouvel essai **sans fin**, attente 2, 4, 8… s
  plafonnée à ``MAX_BACKOFF_S`` (on attend que l'ordinateur retrouve la connexion) ;
- HTTP 429 : attente du délai ``Retry-After`` demandé par Binance, puis reprise ;
- HTTP 418 (IP bannie) et autres 4xx : arrêt immédiat (``ExchangeError``), insister aggraverait ;
- Ctrl-C interrompt toujours l'attente.

``opener``, ``sleep``, ``clock_ms`` et ``notify`` sont injectables : tests sans réseau ni attente.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from qlab.core.errors import ExchangeError
from qlab.core.timeutils import now_ms

TIMEOUT_S = 30
MAX_BACKOFF_S = 60  # attente maximale entre deux essais quand le réseau est coupé
RATE_LIMIT_WAIT_S = 60  # attente si Binance renvoie 429 sans Retry-After


class _Response(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...

    def read(self) -> bytes: ...
    def __enter__(self) -> _Response: ...
    def __exit__(self, *args: object) -> None: ...


Opener = Callable[..., _Response]


@dataclass(frozen=True, slots=True)
class HttpResult:
    """Réponse reçue. ``sent_ms`` / ``received_ms`` : horloge locale (epoch ms) autour de
    l'essai réussi, pour estimer l'écart avec l'heure du serveur."""

    body: bytes
    headers: dict[str, str]
    sent_ms: int
    received_ms: int
    attempts: int


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _retry_after_s(exc: urllib.error.HTTPError) -> int:
    """Délai demandé par Binance (en-tête ``Retry-After``, s) ; défaut ``RATE_LIMIT_WAIT_S``."""
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    try:
        return max(1, int(value)) if value is not None else RATE_LIMIT_WAIT_S
    except ValueError:
        return RATE_LIMIT_WAIT_S


def _public(url: str) -> str:
    """URL sans paramètres pour les messages : ni horodatage ni signature dans les journaux."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Corps de la réponse d'erreur (ex. ``{"code":-1021,"msg":…}`` de Binance), tronqué."""
    try:
        body = exc.read()
    except OSError:
        return ""
    text = body.decode("utf-8", errors="replace").strip()
    return f" : {text[:300]}" if text else ""


def get(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    opener: Opener | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock_ms: Callable[[], int] | None = None,
    notify: Callable[[str], None] = _stderr,
) -> HttpResult:
    """GET ``url`` en attendant le réseau aussi longtemps qu'il le faut (voir le module).

    ``headers`` : en-têtes de la requête (ex. clé d'API) ; jamais écrits dans les messages.
    ``clock_ms`` : horloge locale en epoch ms (défaut ``now_ms``, lue à chaque appel).
    Les messages d'erreur montrent l'URL **sans** ses paramètres (signature, horodatage).
    """
    target: str | urllib.request.Request = (
        urllib.request.Request(url, headers=dict(headers)) if headers else url
    )
    open_url = opener or urllib.request.urlopen
    clock = clock_ms or now_ms
    attempt = 0
    while True:
        attempt += 1
        sent_ms = clock()
        try:
            with open_url(target, timeout=TIMEOUT_S) as resp:
                body = resp.read()
                headers = {k.lower(): v for k, v in dict(resp.headers).items()}
            return HttpResult(body, headers, sent_ms, clock(), attempt)
        except urllib.error.HTTPError as exc:
            if exc.code == 418:
                raise ExchangeError(
                    f"HTTP 418 sur {_public(url)} : IP bannie temporairement par Binance "
                    "(trop de requêtes) ; ne pas relancer avant la levée du bannissement"
                ) from exc
            if exc.code == 429:
                wait_s = _retry_after_s(exc)
                notify(f"limite de requêtes Binance (HTTP 429) : reprise dans {wait_s} s")
                sleep(wait_s)
                continue
            if exc.code < 500:
                raise ExchangeError(
                    f"HTTP {exc.code} sur {_public(url)}{_error_detail(exc)}"
                ) from exc
            reason = f"erreur serveur (HTTP {exc.code})"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            reason = f"réseau indisponible ({exc})"
        wait_s = min(MAX_BACKOFF_S, 2**attempt)
        notify(f"{reason} : essai {attempt} échoué, nouvel essai dans {wait_s} s")
        sleep(wait_s)
