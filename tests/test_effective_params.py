"""Tests de qlab.exchange.effective_params : valeurs calculées à la main.

Config de test : cotation USDT, paires tradées BTCUSDT / ETHUSDT / SOLUSDT, perte journalière
max 5 %, ordre max 100 %, paliers t0 = 50 (drawdown 30 %) et t1 = 200 (25 %), bandes
[10 %, 20 %], DCA 10. Snapshot : minNotional 5 sur chaque paire.

- 50 apportés, portefeuille inconnu ⇒ t0, portefeuille supposé 50 ; δ_min = 5 / 50 = 10 % ;
  perte journalière max 50 × 5 % = 2,5 ; ordre max 50.
- 200 apportés, portefeuille tombé à 150 ⇒ t1 (pas de retour en t0) ; δ_min = 5 / 150 ;
  perte journalière max 150 × 5 % = 7,5.
- Portefeuille 20 ⇒ δ_min = 25 % > bande la plus large 20 % ⇒ alerte par paire.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from fakes import FALLBACK_FEES, binance_symbol, exchange_info

from qlab.core.config import ConfigError, QlabConfig, load_config
from qlab.core.errors import DataError
from qlab.core.jsonlog import read_log
from qlab.core.ledger import Flow, Ledger
from qlab.core.paths import DataPaths
from qlab.exchange import effective_params as ep
from qlab.exchange.effective_params import compute, current
from qlab.exchange.snapshots import Snapshot, SnapshotStore

D = Decimal
T = 1_704_067_200_000  # 2024-01-01T00:00:00Z
FEES = FALLBACK_FEES


def _info(sol_status: str = "TRADING", drop: str | None = None) -> dict[str, Any]:
    symbols = [("BTCUSDT", "TRADING"), ("ETHUSDT", "TRADING"), ("SOLUSDT", sol_status)]
    return exchange_info([binance_symbol(s, "USDT", st) for s, st in symbols if s != drop])


def _snapshot(**kwargs: Any) -> Snapshot:
    return Snapshot(Path("20240101T000000Z.json.zst"), "binance", T, "h", FEES, _info(**kwargs))


@pytest.fixture
def cfg(config_dir: Path) -> QlabConfig:
    return load_config(config_dir)


def _edit(config_dir: Path, name: str, keys: list[str], value: Any) -> QlabConfig:
    path = config_dir / f"{name}.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = doc
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_config(config_dir)


# --- calculs à la main -------------------------------------------------------------------


def test_fifty_euros(cfg: QlabConfig) -> None:
    p = compute(cfg, _snapshot(), at_ms=T, net_deposits_quote=D("50"))
    assert p.tier.name == "t0"
    assert p.equity_quote == D("50") and p.equity_assumed
    assert [x.delta_min_frac for x in p.pairs] == [D("0.1")] * 3
    assert p.max_daily_loss_quote == D("2.50")
    assert p.max_order_quote == D("50.0")
    assert (p.fee_maker_frac, p.fee_taker_frac) == (D("0.001"), D("0.001"))
    assert p.warnings == ("valeur du portefeuille inconnue : capital apporté utilisé à la place",)


def test_losses_do_not_relax_tier(cfg: QlabConfig) -> None:
    p = compute(cfg, _snapshot(), at_ms=T, net_deposits_quote=D("200"), equity_quote=D("150"))
    assert p.tier.name == "t1" and p.tier.max_drawdown_frac == 0.25
    assert p.pairs[0].delta_min_frac == D("5") / D("150")
    assert p.max_daily_loss_quote == D("7.50")
    assert not p.equity_assumed and p.warnings == ()


def test_small_portfolio_cannot_rebalance(cfg: QlabConfig) -> None:
    p = compute(cfg, _snapshot(), at_ms=T, net_deposits_quote=D("50"), equity_quote=D("20"))
    assert p.pairs[0].delta_min_frac == D("0.25")
    assert len([w for w in p.warnings if "rééquilibrage impossible" in w]) == 3


def test_pair_not_trading(cfg: QlabConfig) -> None:
    p = compute(
        cfg,
        _snapshot(sol_status="BREAK"),
        at_ms=T,
        net_deposits_quote=D("50"),
        equity_quote=D("50"),
    )
    assert [x.tradable for x in p.pairs] == [True, True, False]
    assert p.warnings == ("SOLUSDT n'est pas en cotation (statut BREAK)",)


def test_extrapolated_snapshot_is_flagged(cfg: QlabConfig) -> None:
    p = compute(
        cfg,
        _snapshot(),
        at_ms=T - 1,
        net_deposits_quote=D("50"),
        equity_quote=D("50"),
        snapshot_extrapolated=True,
    )
    assert p.snapshot_extrapolated
    assert "règles actuelles appliquées au passé" in p.warnings[0]


def test_config_driven_warnings(config_dir: Path) -> None:
    cfg = _edit(config_dir, "longterm", ["dca", "amount_quote"], 3)
    cfg = _edit(config_dir, "base", ["risk", "max_order_frac"], 0.05)  # 50 × 5 % = 2,5 < 5
    p = compute(cfg, _snapshot(), at_ms=T, net_deposits_quote=D("50"), equity_quote=D("50"))
    assert "DCA de 3.0 sous le minNotional d'au moins une paire" in p.warnings
    assert len([w for w in p.warnings if "< minNotional" in w]) == 3


# --- erreurs -----------------------------------------------------------------------------


def test_traded_pair_missing_from_snapshot(cfg: QlabConfig) -> None:
    with pytest.raises(DataError, match=r"SOLUSDT .* absent du snapshot"):
        compute(cfg, _snapshot(drop="SOLUSDT"), at_ms=T, net_deposits_quote=D("50"))


@pytest.mark.parametrize("equity", [D("0"), D("-5"), D("NaN"), 50.0])
def test_bad_equity(cfg: QlabConfig, equity: Any) -> None:
    with pytest.raises(DataError, match="valeur du portefeuille"):
        compute(cfg, _snapshot(), at_ms=T, net_deposits_quote=D("50"), equity_quote=equity)


def test_no_deposit(cfg: QlabConfig) -> None:
    with pytest.raises(ConfigError, match="net_deposits_quote doit être > 0"):
        compute(cfg, _snapshot(), at_ms=T, net_deposits_quote=D("0"))


# --- lecture du registre et des snapshots ------------------------------------------------


def _prepare(cfg: QlabConfig) -> DataPaths:
    paths = DataPaths(cfg.base.data.root)
    Ledger(paths.ledger).append(Flow(T, "deposit", D("50")))
    SnapshotStore(paths, "binance").save(_info(), FEES, T, "https://x")
    return paths


def test_current_point_in_time(cfg: QlabConfig) -> None:
    paths = _prepare(cfg)
    Ledger(paths.ledger).append(Flow(T + 10_000, "deposit", D("150")))
    assert current(cfg, paths, at_ms=T + 5_000).tier.name == "t0"  # avant le 2e dépôt
    assert current(cfg, paths, at_ms=T + 10_000).tier.name == "t1"
    with pytest.raises(DataError, match="enregistrer un dépôt"):
        current(cfg, paths, at_ms=T - 1)


def test_current_without_snapshot(cfg: QlabConfig) -> None:
    paths = DataPaths(cfg.base.data.root)
    Ledger(paths.ledger).append(Flow(T, "deposit", D("50")))
    with pytest.raises(DataError, match="aucun snapshot"):
        current(cfg, paths, at_ms=T)


# --- commande ----------------------------------------------------------------------------


def test_cli(
    cfg: QlabConfig,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _prepare(cfg)
    monkeypatch.setattr(ep, "now_ms", lambda: T + 1)
    assert ep.main(["--config", str(config_dir), "--equity", "20"]) == 0
    out = capsys.readouterr().out
    assert "Capital apporté 50 USDT ⇒ palier t0 (drawdown max 30%)" in out
    assert "Portefeuille 20 USDT" in out and "supposé" not in out
    assert "BTCUSDT    TRADING  minNotional 5 ⇒ δ_min 25.0%" in out
    assert "ATTENTION : BTCUSDT : δ_min 25.0% > bande la plus large 20%" in out
    records = read_log(sorted((paths.logs / "effective_params").glob("*.jsonl"))[0]).records
    assert [r["kind"] for r in records] == ["run.start", "params.computed"]


def test_cli_errors(cfg: QlabConfig, config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prepare(cfg)
    assert ep.main(["--config", str(config_dir), "--date", "2024-13-01"]) == 1
    assert "YYYY-MM-DD" in capsys.readouterr().err
    assert ep.main(["--config", str(config_dir), "--equity", "abc"]) == 1
    assert "montant illisible" in capsys.readouterr().err
    assert ep.main(["--config", str(config_dir), "--date", "2023-12-31"]) == 1  # avant le dépôt
    assert "enregistrer un dépôt" in capsys.readouterr().err
