"""Unique fixture partagée : un dossier de configuration valide et complet dans ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

BASE: dict[str, object] = {
    "data": {"root": "DATA_ROOT", "disk_budget_gb": 150, "disk_alert_fraction": 0.8},
    "exchanges": [
        {
            "name": "binance",
            "rest_url": "https://api.binance.com",
            "ws_url": "wss://stream.binance.com:9443",
            "fees": {
                "maker_frac": 0.001,
                "taker_frac": 0.001,
                "bnb_discount_frac": 0.25,
                "pay_in_bnb": False,
            },
            "snapshot_refresh_hours": 24,
            "max_clock_offset_ms": 500,
            "trading_days_per_year": 365,
        }
    ],
    "archives": {
        "binance_vision_url": "https://data.binance.vision",
        "binance_vision_list_url": "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
        "tardis_url": "https://datasets.tardis.dev",
        "tardis_api_url": "https://api.tardis.dev",
        "download_workers": 4,
        "list_workers": 4,
        "futures_metrics_symbols": ["BTCUSDT"],
    },
    "symbols": {
        "quote_asset": "USDT",
        "trade": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "intraday": ["BTCUSDT", "ETHUSDT"],
    },
    "observe": {"include_delisted": True, "futures_metrics": True, "kline_intervals": ["1d", "1h"]},
    "risk": {"max_daily_loss_frac": 0.05, "max_open_positions": 1, "max_order_frac": 1.0},
    "capital_tiers": [
        {"name": "t0", "capital_quote": 50, "max_drawdown_frac": 0.3},
        {"name": "t1", "capital_quote": 200, "max_drawdown_frac": 0.25},
    ],
}

INTRADAY: dict[str, object] = {
    "gate": {
        "horizons_s": [5, 30, 60, 300, 900],
        "min_move_cost_ratio": 3.0,
        "slot_minutes": 60,
        "slippage_frac": 0.0,
        "sample_step_ms": 1000,
    },
    "latency": {"data_ms": 100, "order_ms": 100},
    "features": {
        "flow_windows_s": [1, 5, 30],
        "ofi_depth_window_s": 300,
        "vpin_buckets_per_day": 50,
        "vpin_n_buckets": 50,
        "realized_spread_delay_ms": 5000,
        "ewma_lambda": 0.94,
        "signature_freqs_s": [1, 60, 900],
        "seasonality_slot_minutes": 5,
    },
    "risk": {"freshness_max_ms": 2000},
}

LONGTERM: dict[str, object] = {
    "universe": {
        "warmup_days": 30,
        "reference_quotes": ["USDT", "BUSD"],
        "excluded_bases": ["USDC", "EUR"],
        "leveraged_suffixes": ["UP", "DOWN"],
        "volume_lookback_days": 30,
        "min_volume_quote": 1_000_000.0,
    },
    "signals": {
        "momentum_lookbacks_days": [30, 90],
        "ma_fast_days": [10, 20],
        "ma_slow_days": [100, 200],
        "xs_skip_days": 7,
        "xs_top_k": 3,
        "vol_lookback_days": 30,
    },
    "vol_target": {"target_vol_annual": 0.4, "w_max": 1.0},
    "rebalance": {"calendar_days": [7, 30], "band_fracs": [0.1, 0.2]},
    "dca": {"amount_quote": 10, "period_days": 7},
    "costs": {
        "fallback_spread_frac": 0.001,
        "slippage_frac": 0.0005,
        "eur_conversion_cost_frac": 0.002,
        "max_drag_edge_fraction": 0.5,
    },
}


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Écrit base/intraday/longterm.yaml valides ; la racine des données est ``tmp_path/data``."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    base = {
        **BASE,
        "data": {**BASE["data"], "root": str(tmp_path / "data")},  # type: ignore[dict-item]
        "secrets_file": str(tmp_path / "secrets.env"),  # absent par défaut : frais de repli
    }
    for name, content in (("base", base), ("intraday", INTRADAY), ("longterm", LONGTERM)):
        (cfg / f"{name}.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")
    return cfg
