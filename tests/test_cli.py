"""Tests de qlab.core.cli : même mécanique pour toutes les commandes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from qlab.core.cli import Context, run_command
from qlab.core.errors import DataError
from qlab.core.jsonlog import read_log


def _add(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=0)


def _journal_kinds(root: Path) -> list[str]:
    return [
        str(r["kind"])
        for f in sorted((root / "logs" / "demo").glob("*.jsonl"))
        for r in read_log(f).records
    ]


def test_context_and_journal(config_dir: Path, tmp_path: Path) -> None:
    seen: list[Context] = []

    def action(ctx: Context) -> int:
        seen.append(ctx)
        ctx.journal.info("demo.done", {"n": ctx.args.n})
        return 0

    code = run_command(
        ["--config", str(config_dir), "--n", "3"],
        prog="demo",
        description="d",
        component="demo",
        add_arguments=_add,
        action=action,
    )
    assert code == 0
    ctx = seen[0]
    assert ctx.paths.root == tmp_path / "data"
    assert ctx.config.base.symbols.quote_asset == "USDT"
    assert _journal_kinds(tmp_path / "data") == ["run.start", "demo.done"]


def test_action_error_is_journaled(
    config_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def action(ctx: Context) -> int:
        raise DataError("trou")

    code = run_command(
        ["--config", str(config_dir)],
        prog="demo",
        description="d",
        component="demo",
        add_arguments=_add,
        action=action,
    )
    assert code == 1
    assert "ERREUR (DataError) : trou" in capsys.readouterr().err
    assert _journal_kinds(tmp_path / "data") == ["run.start", "run.failed"]


def test_bad_config_exits_1_without_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_command(
        ["--config", str(tmp_path / "absent")],
        prog="demo",
        description="d",
        component="demo",
        add_arguments=_add,
        action=lambda ctx: 0,
    )
    assert code == 1
    assert "ERREUR (ConfigError)" in capsys.readouterr().err
