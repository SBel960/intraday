"""Tests de qlab.core.http : attente du réseau, limites Binance, sans réseau ni attente réelle."""

from __future__ import annotations

import http.client as http_client
import io
import itertools
import urllib.error
from typing import Any

import pytest

from qlab.core import http
from qlab.core.errors import ExchangeError

URL = "https://api.example/x"


class _Resp:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "err", headers or {}, io.BytesIO(b""))  # type: ignore[arg-type]


class Flaky:
    """Connexion qui échoue selon ``failures`` puis répond ; enregistre attentes et messages."""

    def __init__(
        self, failures: list[Exception], body: bytes = b"ok", headers: dict[str, str] | None = None
    ) -> None:
        self.failures = list(failures)
        self.body, self.headers = body, headers
        self.calls = 0
        self.waits: list[float] = []
        self.messages: list[str] = []
        self.clock = itertools.count(1_000, 10)  # horloge locale factice : +10 ms par lecture

    def open(self, url: str, timeout: int) -> _Resp:
        assert timeout == http.TIMEOUT_S
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _Resp(self.body, self.headers)

    def get(self) -> http.HttpResult:
        return http.get(
            URL,
            opener=self.open,
            sleep=self.waits.append,
            clock_ms=lambda: next(self.clock),
            notify=self.messages.append,
        )


def test_success_first_try() -> None:
    flaky = Flaky([], b"hello", {"X-MBX-USED-WEIGHT-1M": "20"})
    result = flaky.get()
    assert result.body == b"hello"
    assert result.headers == {"x-mbx-used-weight-1m": "20"}  # clés en minuscules
    assert (result.sent_ms, result.received_ms, result.attempts) == (1_000, 1_010, 1)
    assert flaky.waits == [] and flaky.messages == []


def test_waits_for_network_as_long_as_needed() -> None:
    """Réseau coupé 10 fois : attente 2, 4, 8, 16, 32 puis 60 s (plafond), puis succès."""
    flaky = Flaky([urllib.error.URLError("pas de réseau")] * 10)
    result = flaky.get()
    assert result.attempts == 11
    assert flaky.waits == [2, 4, 8, 16, 32, 60, 60, 60, 60, 60]
    assert "réseau indisponible" in flaky.messages[0]
    assert "essai 10 échoué, nouvel essai dans 60 s" in flaky.messages[-1]
    assert result.sent_ms > 1_000  # horodatage de l'essai réussi, pas du premier


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("lent"), ConnectionResetError("coupé"), http_error(500), http_error(503)],
)
def test_transient_failures_are_retried(failure: Exception) -> None:
    flaky = Flaky([failure, failure])
    assert flaky.get().attempts == 3
    assert flaky.waits == [2, 4]


@pytest.mark.parametrize(
    ("headers", "expected_wait"),
    [
        ({"Retry-After": "7"}, 7),
        ({}, 60),
        ({"Retry-After": "bientôt"}, 60),
        ({"Retry-After": "0"}, 1),
    ],
)
def test_rate_limit_waits_as_requested(headers: dict[str, str], expected_wait: int) -> None:
    flaky = Flaky([http_error(429, headers)])
    flaky.get()
    assert flaky.waits == [expected_wait]
    assert "HTTP 429" in flaky.messages[0]


@pytest.mark.parametrize(
    ("failure", "msg"),
    [
        (http_error(418), "HTTP 418 .*IP bannie"),
        (http_error(404), "HTTP 404"),
        (http_error(403), "HTTP 403"),
    ],
)
def test_fatal_http_errors_stop_immediately(failure: Exception, msg: str) -> None:
    flaky = Flaky([failure])
    with pytest.raises(ExchangeError, match=msg):
        flaky.get()
    assert flaky.waits == [] and flaky.calls == 1  # pas d'insistance


def test_ctrl_c_interrupts_waiting() -> None:
    def interrupted(_: float) -> None:
        raise KeyboardInterrupt

    flaky = Flaky([urllib.error.URLError("x")])
    with pytest.raises(KeyboardInterrupt):
        http.get(URL, opener=flaky.open, sleep=interrupted, notify=lambda _: None)


def test_default_notify_writes_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    flaky: Any = Flaky([urllib.error.URLError("x")])
    http.get(URL, opener=flaky.open, sleep=lambda _: None)
    assert "réseau indisponible" in capsys.readouterr().err


# --- en-têtes et messages d'erreur (clé d'API) --------------------------------------------


def test_headers_are_sent_in_a_request() -> None:
    seen: list[Any] = []

    def open_url(target: Any, timeout: int) -> _Resp:
        seen.append(target)
        return _Resp(b"ok")

    http.get(URL, headers={"X-MBX-APIKEY": "cle"}, opener=open_url, notify=lambda _: None)
    assert seen[0].get_header("X-mbx-apikey") == "cle"  # urllib normalise la casse
    http.get(URL, opener=open_url, notify=lambda _: None)
    assert seen[1] == URL  # sans en-tête : l'URL brute, comme avant


def test_error_shows_binance_message_but_not_query() -> None:
    signed = URL + "?timestamp=1&signature=SECRETSIG"
    body = b'{"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}'
    error = urllib.error.HTTPError(signed, 400, "Bad", {}, io.BytesIO(body))  # type: ignore[arg-type]
    flaky = Flaky([error])
    with pytest.raises(ExchangeError) as err:
        http.get(signed, opener=flaky.open, notify=lambda _: None)
    message = str(err.value)
    assert "-1021" in message and "recvWindow" in message
    assert "SECRETSIG" not in message and "timestamp=" not in message


def test_connection_cut_mid_transfer_is_retried() -> None:
    """Coupure au milieu d'un fichier (5,3 Mo reçus sur 12,2 Mo, vu sur Tardis) : on réessaie."""
    cut = http_client.IncompleteRead(b"x" * 10, 20)
    flaky = Flaky([cut, http_client.RemoteDisconnected("fermé")])
    assert flaky.get().attempts == 3
    assert flaky.waits == [2, 4]
    assert "réseau indisponible" in flaky.messages[0]
