"""Tests de qlab.core.errors : hiérarchie et codes de sortie de run_cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from qlab.core.config import ConfigError
from qlab.core.errors import (
    EXIT_BUG,
    EXIT_EXPECTED_ERROR,
    EXIT_OK,
    DataError,
    ExchangeError,
    QlabError,
    run_cli,
)
from qlab.core.jsonlog import JsonLog, read_log

T = 1_704_116_832_345  # 2024-01-01T13:47:12.345Z


def _raise(exc: BaseException) -> int:
    raise exc


# --- hiérarchie --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [ConfigError, DataError, ExchangeError])
def test_project_errors_are_qlab_errors(cls: type[Exception]) -> None:
    assert issubclass(cls, QlabError)


def test_config_error_still_a_value_error() -> None:
    """Compatibilité : le code qui attrape ValueError continue d'attraper ConfigError."""
    assert issubclass(ConfigError, ValueError)
    with pytest.raises(QlabError):
        raise ConfigError("x")


# --- codes de sortie ---------------------------------------------------------------------


def test_success_returns_entry_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(lambda: EXIT_OK) == 0
    assert run_cli(lambda: 3) == 3  # le code choisi par entry est conservé
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("exc", [ConfigError("clé manquante"), DataError("trou"), QlabError("x")])
def test_expected_error(exc: QlabError, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(lambda: _raise(exc)) == EXIT_EXPECTED_ERROR
    err = capsys.readouterr().err
    assert err == f"ERREUR ({type(exc).__name__}) : {exc}\n"  # une ligne, pas de trace


def test_bug(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(lambda: _raise(KeyError("inattendu"))) == EXIT_BUG
    err = capsys.readouterr().err
    assert err.startswith("BUG (KeyError) :")
    assert "Traceback (most recent call last)" in err  # trace complète, rien de masqué


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(5)])
def test_base_exceptions_not_intercepted(exc: BaseException) -> None:
    with pytest.raises(type(exc)):
        run_cli(lambda: _raise(exc))


# --- journalisation ----------------------------------------------------------------------


def test_errors_are_journaled(tmp_path: Path) -> None:
    with JsonLog(tmp_path, "cli", clock_ms=lambda: T) as journal:
        assert run_cli(lambda: _raise(DataError("trou 12:00")), journal=journal) == 1
        assert run_cli(lambda: _raise(ZeroDivisionError("div")), journal=journal) == 2
        assert run_cli(lambda: 0, journal=journal) == 0  # succès : rien d'écrit
    read = read_log(tmp_path / "cli" / "2024-01-01.jsonl")
    assert read.corrupt_lines == ()
    assert len(read.records) == 2
    expected, bug = (r["data"] for r in read.records)
    assert isinstance(expected, dict) and isinstance(bug, dict)
    assert expected == {"error": "DataError", "message": "trou 12:00", "bug": False}
    assert bug["error"] == "ZeroDivisionError" and bug["bug"] is True
    assert isinstance(bug["trace"], str) and "ZeroDivisionError: div" in bug["trace"]
    assert all(r["level"] == "error" and r["kind"] == "run.failed" for r in read.records)
