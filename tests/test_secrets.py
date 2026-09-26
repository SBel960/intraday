"""Tests de qlab.core.secrets : format strict, droits du fichier, secrets jamais affichés."""

from __future__ import annotations

from pathlib import Path

import pytest

from qlab.core.errors import DataError
from qlab.core.secrets import Secret, check_private, load_env_file, require


def _env(tmp_path: Path, text: str, mode: int = 0o600) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_load_and_reveal(tmp_path: Path) -> None:
    path = _env(tmp_path, "# commentaire\n\nBINANCE_API_KEY=abc123\nKEY_PATH=/x/y.pem\n")
    values = load_env_file(path)
    assert values["BINANCE_API_KEY"].reveal() == "abc123"
    assert require(values, "KEY_PATH", path).reveal() == "/x/y.pem"


def test_secret_never_printed() -> None:
    s = Secret("tres-secret")
    assert repr(s) == str(s) == f"{s}" == "Secret(***)"
    assert "tres-secret" not in repr({"k": s})


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660])
def test_readable_by_others_rejected(tmp_path: Path, mode: int) -> None:
    path = _env(tmp_path, "A=1\n", mode)
    with pytest.raises(DataError, match="chmod 600"):
        load_env_file(path)


def test_private_ok(tmp_path: Path) -> None:
    check_private(_env(tmp_path, "A=1\n", 0o600))
    check_private(_env(tmp_path, "A=1\n", 0o400))


@pytest.mark.parametrize(
    ("text", "msg"),
    [
        ("lowercase=1\n", "CLE=valeur"),
        ("export A=1\n", "CLE=valeur"),
        ("A=1\nA=2\n", "en double"),
        ('A="1"\n', "guillemets"),
        ("A=\n", "secret vide"),
    ],
)
def test_bad_format(tmp_path: Path, text: str, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        load_env_file(_env(tmp_path, text))


def test_missing_key_message_has_no_values(tmp_path: Path) -> None:
    path = _env(tmp_path, "OTHER=valeur-privee\n")
    with pytest.raises(DataError, match="BINANCE_API_KEY absent") as err:
        require(load_env_file(path), "BINANCE_API_KEY", path)
    assert "valeur-privee" not in str(err.value)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "absent")
