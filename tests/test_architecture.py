"""Règles d'architecture vérifiées automatiquement (voir « Règles d'architecture » dans
docs/ARBORESCENCE.md). Un échec ici signale du code spaghetti en train de naître : un besoin
transversal réécrit ailleurs que dans son module du socle, une dépendance à l'envers, un module
trop gros ou sans test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "qlab"
MODULES = sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")

# Couches : un paquet n'importe que des paquets de rang inférieur ou égal.
LAYERS = {
    "core": 0,
    "exchange": 1,
    "costs": 2,
    "data": 2,
    "research": 3,
    "sampling": 3,
    "features": 3,
    "sizing": 3,
    "longterm": 4,
    "backtest": 4,
    "events": 4,
    "live": 5,
}
MAX_LINES = 300
TOO_BIG_ALLOWED = {"core/config.py"}  # validé tel quel par le propriétaire (2026-09-25)
DATA_DIRS = {"meta", "logs", "reports", "raw", "bronze", "silver", "lt", "events"}


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_modules_found() -> None:
    assert len(MODULES) >= 10


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_layering(path: Path) -> None:
    """Pas de dépendance à l'envers (ex. ``core`` qui importerait ``exchange``)."""
    package = _rel(path).split("/")[0]
    assert package in LAYERS, f"paquet {package!r} absent de LAYERS : le classer"
    for name in _imports(_tree(path)):
        if name.startswith("qlab."):
            target = name.split(".")[1]
            assert LAYERS[target] <= LAYERS[package], (
                f"{_rel(path)} importe {name} : couche {target} au-dessus de {package}"
            )


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_network_only_in_http(path: Path) -> None:
    if _rel(path) == "core/http.py":
        return
    used = {n for n in _imports(_tree(path)) if n.startswith(("urllib", "requests", "http."))}
    assert not used, f"{_rel(path)} accède au réseau hors core/http.py : {used}"


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_parsers_only_in_cli(path: Path) -> None:
    if _rel(path) in {"core/cli.py", "core/config.py"}:
        return
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "ArgumentParser", (
                f"{_rel(path)} crée son parseur : passer par core/cli.run_command"
            )


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_data_paths_only_in_paths(path: Path) -> None:
    """Aucun ``chemin / "meta"`` hors core/paths.py : l'arbre des données a un seul endroit."""
    if _rel(path) == "core/paths.py":
        return
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
            if isinstance(right, ast.Constant) and right.value in DATA_DIRS:
                pytest.fail(
                    f"{_rel(path)}:{node.lineno} construit un chemin de données "
                    f"({right.value!r}) : passer par core/paths.DataPaths"
                )


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_no_blind_except(path: Path) -> None:
    """``except`` nu / ``except Exception`` : seulement dans run_cli ou pour nettoyer + relancer."""
    if _rel(path) == "core/errors.py":
        return
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
        )
        reraises = (
            bool(node.body) and isinstance(node.body[-1], ast.Raise) and (node.body[-1].exc is None)
        )
        assert not broad or reraises, (
            f"{_rel(path)}:{node.lineno} avale toutes les erreurs : attraper des types précis"
        )


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_no_sys_exit(path: Path) -> None:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Attribute) and node.attr == "exit":
            assert not (isinstance(node.value, ast.Name) and node.value.id == "sys"), (
                f"{_rel(path)} appelle sys.exit : renvoyer un code depuis main()"
            )


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_commands_use_run_command(path: Path) -> None:
    """Toute commande (``def main``) passe par core/cli.run_command (sauf config.py)."""
    if _rel(path) in {"core/config.py", "core/cli.py"}:
        return
    tree = _tree(path)
    if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "run_command" in calls, f"{_rel(path)} : main() sans run_command"


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_size(path: Path) -> None:
    lines = len(path.read_text(encoding="utf-8").splitlines())
    if _rel(path) not in TOO_BIG_ALLOWED:
        assert lines <= MAX_LINES, f"{_rel(path)} : {lines} lignes > {MAX_LINES}, découper"


@pytest.mark.parametrize("path", MODULES, ids=_rel)
def test_every_module_has_its_test(path: Path) -> None:
    """Règle du §11 : un fichier source + son test."""
    assert (ROOT / "tests" / f"test_{path.stem}.py").exists(), f"tests/test_{path.stem}.py absent"


def test_documented_in_arborescence() -> None:
    """Chaque module du code figure dans l'arbre de docs/ARBORESCENCE.md."""
    doc = (ROOT / "docs" / "ARBORESCENCE.md").read_text(encoding="utf-8")
    missing = [_rel(p) for p in MODULES if f"── {p.name}" not in doc]
    assert not missing, f"modules absents de ARBORESCENCE.md : {missing}"
