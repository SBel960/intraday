"""Tests de qlab.core.config : cas nominal, cas limites, cas d'erreur."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from qlab.core.config import (
    ConfigError,
    QlabConfig,
    load_base,
    load_config,
    load_intraday,
    load_longterm,
    main,
)

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"


def _edit(path: Path, keys: list[str | int], value: Any) -> None:
    """Remplace la valeur au chemin ``keys`` dans le YAML ; ``value=...`` supprime la clé."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = doc
    for k in keys[:-1]:
        node = node[k]
    if value is ...:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


# --- cas nominal -------------------------------------------------------------------------


def test_load_nominal_values(config_dir: Path, tmp_path: Path) -> None:
    cfg = load_config(config_dir)
    assert isinstance(cfg, QlabConfig)
    assert cfg.base.data.root == tmp_path / "data"
    assert cfg.base.data.disk_budget_gb == 150.0
    assert isinstance(cfg.base.data.disk_budget_gb, float)  # int YAML → float
    assert cfg.base.symbols.intraday == ("BTCUSDT", "ETHUSDT")
    fees = cfg.base.exchange("binance").fees
    assert fees.taker_frac == 0.001
    assert fees.pay_in_bnb is False
    assert fees.effective_taker_frac == 0.001  # pas de remise sans paiement en BNB
    assert [t.capital_quote for t in cfg.base.capital_tiers] == [50.0, 200.0]
    assert cfg.intraday.gate.horizons_s == (5, 30, 60, 300, 900)
    assert cfg.intraday.gate.min_move_cost_ratio == 3.0
    assert cfg.intraday.latency.order_ms == 100
    assert cfg.intraday.risk.freshness_max_ms == 2000
    assert cfg.longterm.dca.amount_quote == 10.0
    assert cfg.longterm.rebalance.band_fracs == (0.1, 0.2)


def test_repo_config_is_valid() -> None:
    """Les YAML du dépôt sont chargeables tels quels."""
    cfg = load_config(REPO_CONFIG)
    assert cfg.intraday.gate.horizons_s == (5, 30, 60, 300, 900)
    assert cfg.base.symbols.quote_asset == "EUR"  # compte EEE (MiCA)
    assert cfg.base.symbols.trade == ("BTCEUR", "ETHEUR", "SOLEUR")
    assert cfg.base.symbols.intraday == ("BTCEUR", "ETHEUR")
    assert cfg.base.observe.include_delisted is True  # vue globale sans biais du survivant
    assert cfg.base.observe.futures_metrics is True
    assert cfg.base.observe.kline_intervals == ("1d", "1h")
    assert cfg.longterm.costs.eur_conversion_cost_frac == 0.0


def test_deterministic(config_dir: Path) -> None:
    assert load_config(config_dir) == load_config(config_dir)


def test_frozen(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.base.data.disk_budget_gb = 1.0  # type: ignore[misc]


def test_main_ok(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(config_dir)]) == 0
    assert '"horizons_s"' in capsys.readouterr().out


# --- cas limites -------------------------------------------------------------------------


def test_single_symbol_and_single_tier(config_dir: Path) -> None:
    base = config_dir / "base.yaml"
    _edit(base, ["symbols", "trade"], ["BTCUSDT"])
    _edit(base, ["symbols", "intraday"], ["BTCUSDT"])
    _edit(base, ["observe", "kline_intervals"], ["1d"])
    _edit(base, ["capital_tiers"], [{"name": "t0", "capital_quote": 50, "max_drawdown_frac": 1}])
    cfg = load_base(base)
    assert cfg.symbols.trade == cfg.symbols.intraday == ("BTCUSDT",)
    assert cfg.observe.kline_intervals == ("1d",)
    assert len(cfg.capital_tiers) == 1


def test_zero_values_allowed_where_meaningful(config_dir: Path) -> None:
    _edit(config_dir / "intraday.yaml", ["latency", "data_ms"], 0)
    _edit(config_dir / "intraday.yaml", ["gate", "slippage_frac"], 0)
    _edit(config_dir / "longterm.yaml", ["universe", "warmup_days"], 0)
    cfg = load_config(config_dir)
    assert cfg.intraday.latency.data_ms == 0
    assert cfg.longterm.universe.warmup_days == 0


@pytest.mark.parametrize("path", [["gate", "horizons_s"], ["features", "flow_windows_s"]])
def test_empty_list_rejected(config_dir: Path, path: list[str | int]) -> None:
    _edit(config_dir / "intraday.yaml", path, [])
    with pytest.raises(ConfigError, match="vide"):
        load_intraday(config_dir / "intraday.yaml")


def test_empty_file_rejected(config_dir: Path) -> None:
    (config_dir / "longterm.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="table attendue, reçu NoneType"):
        load_longterm(config_dir / "longterm.yaml")


# --- cas d'erreur : structure ------------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="lecture impossible"):
        load_config(tmp_path / "absent")


def test_invalid_yaml(config_dir: Path) -> None:
    (config_dir / "base.yaml").write_text("data: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML invalide"):
        load_base(config_dir / "base.yaml")


def test_missing_key_reports_path(config_dir: Path) -> None:
    _edit(config_dir / "base.yaml", ["data", "disk_budget_gb"], ...)
    msg = r"base\.data : clé\(s\) manquante\(s\) \['disk_budget_gb'\]"
    with pytest.raises(ConfigError, match=msg):
        load_base(config_dir / "base.yaml")


def test_unknown_key_rejected(config_dir: Path) -> None:
    _edit(config_dir / "intraday.yaml", ["gate", "horizon_s"], [5])  # faute de frappe
    msg = r"intraday\.gate : clé\(s\) inconnue\(s\) \['horizon_s'\]"
    with pytest.raises(ConfigError, match=msg):
        load_intraday(config_dir / "intraday.yaml")


@pytest.mark.parametrize(
    ("keys", "value", "msg"),
    [
        (["latency", "data_ms"], True, "int attendu, reçu bool"),
        (["latency", "data_ms"], 1.5, "int attendu, reçu float"),
        (["gate", "min_move_cost_ratio"], "3", "float attendu, reçu str"),
        (["gate", "min_move_cost_ratio"], float("inf"), "nombre fini"),
        (["gate", "horizons_s"], 5, "liste attendue"),
        (["gate", "horizons_s"], [5, "30"], r"horizons_s\[1\] : int attendu"),
        (["gate"], [1], "table attendue"),
    ],
)
def test_wrong_types(config_dir: Path, keys: list[str | int], value: Any, msg: str) -> None:
    _edit(config_dir / "intraday.yaml", keys, value)
    with pytest.raises(ConfigError, match=msg):
        load_intraday(config_dir / "intraday.yaml")


def test_empty_string_rejected(config_dir: Path) -> None:
    _edit(config_dir / "base.yaml", ["exchanges", 0, "name"], "  ")
    with pytest.raises(ConfigError, match="chaîne vide"):
        load_base(config_dir / "base.yaml")


# --- cas d'erreur : domaine --------------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "msg"),
    [("/mnt/c/qlab-data", "sous /mnt"), ("relative/data", "chemin absolu")],
)
def test_data_root_rules(config_dir: Path, root: str, msg: str) -> None:
    _edit(config_dir / "base.yaml", ["data", "root"], root)
    with pytest.raises(ConfigError, match=msg):
        load_base(config_dir / "base.yaml")


def test_data_root_symlink_to_mnt_rejected(config_dir: Path, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to("/mnt/c/somewhere")
    _edit(config_dir / "base.yaml", ["data", "root"], str(link))
    with pytest.raises(ConfigError, match="sous /mnt"):
        load_base(config_dir / "base.yaml")


@pytest.mark.parametrize(
    ("keys", "value", "msg"),
    [
        (["data", "disk_budget_gb"], 0, "disk_budget_gb doit être > 0"),
        (["data", "disk_alert_fraction"], 1.5, r"disk_alert_fraction doit être dans \]0, 1\]"),
        (["exchanges", 0, "fees", "taker_frac"], -0.001, "taker_frac"),
        (["exchanges", 0, "fees", "bnb_discount_frac"], 1.0, "bnb_discount_frac"),
        (["exchanges", 0, "fees", "snapshot_refresh_hours"], 0, "snapshot_refresh_hours"),
        (["exchanges", 0, "fees", "pay_in_bnb"], 1, "bool attendu, reçu int"),
        (["exchanges", 0, "rest_url"], "http://api.binance.com", "https://"),
        (["exchanges", 0, "ws_url"], "https://x", "wss://"),
        (["symbols", "intraday"], ["BTCUSDT", "BTCUSDT"], "doublons"),
        (["symbols", "trade"], ["BTCEUR"], "pas coté en USDT"),
        (["symbols", "trade"], [], "trade ne doit pas être vide"),
        (
            ["symbols", "trade"],
            ["BTCUSDT"],
            r"intraday doit être inclus dans trade : \['ETHUSDT'\]",
        ),
        (["observe", "kline_intervals"], ["1h"], "doit contenir 1d"),
        (["observe", "kline_intervals"], ["1d", "2d"], r"inconnus de Binance : \['2d'\]"),
        (["observe", "kline_intervals"], ["1d", "1d"], "doublons"),
        (["observe", "include_delisted"], "yes", "bool attendu"),
        (["symbols", "intraday"], ["btcusdt"], "symbole invalide"),
        (["risk", "max_open_positions"], 0, "max_open_positions"),
        (["capital_tiers", 1, "capital_quote"], 50, "strictement croissant"),
        (["capital_tiers", 1, "name"], "t0", "doublons"),
        (["capital_tiers"], [], "vide"),
        (["exchanges"], [], "exchanges ne doit pas être vide"),
    ],
)
def test_base_domain(config_dir: Path, keys: list[str | int], value: Any, msg: str) -> None:
    _edit(config_dir / "base.yaml", keys, value)
    with pytest.raises(ConfigError, match=msg):
        load_base(config_dir / "base.yaml")


def test_duplicate_exchange_rejected(config_dir: Path) -> None:
    ex = yaml.safe_load((config_dir / "base.yaml").read_text(encoding="utf-8"))["exchanges"][0]
    _edit(config_dir / "base.yaml", ["exchanges"], [ex, ex])
    with pytest.raises(ConfigError, match=r"exchanges\.name contient des doublons"):
        load_base(config_dir / "base.yaml")


def test_unknown_exchange_lookup(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="non configuré : 'bybit'"):
        load_base(config_dir / "base.yaml").exchange("bybit")


@pytest.mark.parametrize(
    ("keys", "value", "msg"),
    [
        (["gate", "horizons_s"], [30, 5], "strictement croissant"),
        (["gate", "horizons_s"], [0, 5], r"horizons_s\[0\] doit être > 0"),
        (["gate", "slot_minutes"], 7, "diviser 1440"),
        (["gate", "min_move_cost_ratio"], 0, "> 0"),
        (["latency", "order_ms"], -1, "≥ 0"),
        (["features", "ewma_lambda"], 1.0, r"ewma_lambda doit être dans \]0, 1\["),
        (["features", "vpin_n_buckets"], 0, "vpin_n_buckets"),
        (["features", "seasonality_slot_minutes"], 0, "diviser 1440"),
        (["risk", "freshness_max_ms"], 0, "freshness_max_ms"),
    ],
)
def test_intraday_domain(config_dir: Path, keys: list[str | int], value: Any, msg: str) -> None:
    _edit(config_dir / "intraday.yaml", keys, value)
    with pytest.raises(ConfigError, match=msg):
        load_intraday(config_dir / "intraday.yaml")


@pytest.mark.parametrize(
    ("keys", "value", "msg"),
    [
        (["signals", "ma_fast_days"], [10, 150], "ma_fast_days doit être < chaque ma_slow_days"),
        (["signals", "xs_skip_days"], 30, "xs_skip_days doit être <"),
        (["signals", "xs_top_k"], 0, "xs_top_k"),
        (["signals", "vol_lookback_days"], 1, "≥ 2"),
        (["vol_target", "w_max"], 1.5, "w_max"),  # pas de levier
        (["rebalance", "band_fracs"], [0.1, 1.2], r"band_fracs\[-1\]"),
        (["dca", "amount_quote"], 0, "amount_quote"),
        (["costs", "fallback_spread_frac"], 0, "fallback_spread_frac"),
        (["costs", "max_drag_edge_fraction"], 0, "max_drag_edge_fraction"),
        (["universe", "warmup_days"], -1, "warmup_days"),
    ],
)
def test_longterm_domain(config_dir: Path, keys: list[str | int], value: Any, msg: str) -> None:
    _edit(config_dir / "longterm.yaml", keys, value)
    with pytest.raises(ConfigError, match=msg):
        load_longterm(config_dir / "longterm.yaml")


def test_main_reports_error(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit(config_dir / "base.yaml", ["data", "root"], "/mnt/c/x")
    assert main(["--config", str(config_dir)]) == 1  # erreur attendue, pas un bug (2)
    err = capsys.readouterr().err
    assert err.startswith("ERREUR (ConfigError) : base.data : root ne doit pas être sous /mnt")


# --- recalcul : frais effectifs et palier selon le capital -------------------------------


def test_bnb_discount_applied(config_dir: Path) -> None:
    _edit(config_dir / "base.yaml", ["exchanges", 0, "fees", "pay_in_bnb"], True)
    fees = load_base(config_dir / "base.yaml").exchange("binance").fees
    # 0,1 % × (1 − 0,25) = 0,075 %
    assert fees.effective_maker_frac == pytest.approx(0.00075)
    assert fees.effective_taker_frac == pytest.approx(0.00075)


def test_fee_change_is_picked_up(config_dir: Path) -> None:
    _edit(config_dir / "base.yaml", ["exchanges", 0, "fees", "taker_frac"], 0.0009)
    fees = load_base(config_dir / "base.yaml").exchange("binance").fees
    assert fees.effective_taker_frac == 0.0009


@pytest.mark.parametrize(
    ("net_deposits", "tier"),
    [
        ("30", "t0"),  # apport sous le premier seuil : premier palier
        ("50", "t0"),  # seuil exact
        ("199.99", "t0"),
        ("199.999999999999999999", "t0"),  # juste sous 200 : aucun arrondi flottant
        ("200", "t1"),
        ("10000", "t1"),  # au-delà du dernier seuil : dernier palier
    ],
)
def test_tier_for_net_deposits(config_dir: Path, net_deposits: str, tier: str) -> None:
    assert load_base(config_dir / "base.yaml").tier_for(Decimal(net_deposits)).name == tier


def test_tier_is_not_relaxed_by_losses(config_dir: Path) -> None:
    """200 déposés puis valeur tombée à 150 : on reste en t1 (drawdown max 25 %, pas 30 %).

    Le palier ne prend que l'apport en argument ; la valeur de marché ne peut pas l'influencer.
    """
    base = load_base(config_dir / "base.yaml")
    tier = base.tier_for(net_deposits_quote=Decimal("200"))
    assert tier.name == "t1"
    assert tier.max_drawdown_frac == 0.25


@pytest.mark.parametrize("net_deposits", ["0", "-5", "NaN", "Infinity"])
def test_tier_for_invalid_deposits(config_dir: Path, net_deposits: str) -> None:
    with pytest.raises(ConfigError, match="net_deposits_quote doit être > 0"):
        load_base(config_dir / "base.yaml").tier_for(Decimal(net_deposits))


@pytest.mark.parametrize("net_deposits", [200.0, 200])
def test_tier_for_rejects_non_decimal(config_dir: Path, net_deposits: object) -> None:
    """Pas de flottant pour de l'argent : on refuse au lieu de convertir en silence."""
    with pytest.raises(ConfigError, match="doit être un Decimal"):
        load_base(config_dir / "base.yaml").tier_for(net_deposits)  # type: ignore[arg-type]
