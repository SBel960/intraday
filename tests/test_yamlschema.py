"""Tests de qlab.core.yamlschema : lecture stricte YAML → dataclass, classe d'erreur choisie."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from qlab.core.yamlschema import SchemaError, build, read_yaml


class MyError(SchemaError):
    pass


@dataclass(frozen=True, slots=True)
class Inner:
    n: int

    def __post_init__(self) -> None:
        if self.n < 0:
            raise MyError("n doit être ≥ 0")


@dataclass(frozen=True, slots=True)
class Outer:
    name: str
    rate: float
    flag: bool
    items: tuple[Inner, ...]
    path: Path


RAW = {"name": "x", "rate": 1, "flag": True, "items": [{"n": 1}, {"n": 2}], "path": "~/a"}


def test_build_nominal() -> None:
    out = build(Outer, RAW, "o", error=MyError)
    assert out.rate == 1.0 and isinstance(out.rate, float)
    assert out.items == (Inner(1), Inner(2))
    assert out.path == Path("~/a").expanduser()


@pytest.mark.parametrize(
    ("raw", "msg"),
    [
        ({k: v for k, v in RAW.items() if k != "flag"}, r"o : clé\(s\) manquante\(s\) \['flag'\]"),
        ({**RAW, "extra": 1}, r"inconnue\(s\) \['extra'\]"),
        ({**RAW, "flag": 1}, "o.flag : bool attendu, reçu int"),
        ({**RAW, "rate": True}, "float attendu, reçu bool"),
        ({**RAW, "rate": float("nan")}, "nombre fini"),
        ({**RAW, "name": "  "}, "chaîne vide"),
        ({**RAW, "items": {"n": 1}}, "liste attendue"),
        ({**RAW, "items": [{"n": -1}]}, r"o.items\[0\] : n doit être ≥ 0"),
        ([1], "table attendue"),
    ],
)
def test_errors_use_the_given_class(raw: object, msg: str) -> None:
    with pytest.raises(MyError, match=msg):
        build(Outer, raw, "o", error=MyError)


def test_read_yaml(tmp_path: Path) -> None:
    (tmp_path / "ok.yaml").write_text("a: 1\n")
    assert read_yaml(tmp_path / "ok.yaml", error=MyError) == {"a": 1}
    (tmp_path / "bad.yaml").write_text("a: [")
    with pytest.raises(MyError, match="YAML invalide"):
        read_yaml(tmp_path / "bad.yaml", error=MyError)
    with pytest.raises(MyError, match="lecture impossible"):
        read_yaml(tmp_path / "absent.yaml", error=MyError)
    (tmp_path / "code.yaml").write_text("a: !!python/object/apply:os.system ['echo x']\n")
    with pytest.raises(MyError, match="YAML invalide"):  # chargeur sûr : rien n'est exécuté
        read_yaml(tmp_path / "code.yaml", error=MyError)


def test_unsupported_annotation_is_a_bug() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: complex

    with pytest.raises(TypeError, match="non géré"):
        build(Bad, {"x": 1}, "b", error=MyError)
