"""Chargement de la configuration YAML en dataclasses figées et validées.

Trois fichiers dans un dossier : ``base.yaml``, ``intraday.yaml``, ``longterm.yaml``.
Toute clé manquante ou inconnue, tout type incorrect et toute valeur hors domaine lève
``ConfigError`` avec le chemin de la clé (ex. ``base.data.root``). Aucune valeur par défaut :
ce qui n'est pas dans le YAML n'existe pas.

Conventions d'unité dans les noms : ``_s`` secondes, ``_ms`` millisecondes, ``_days`` jours,
``_frac`` fraction sans unité (0.001 = 0,1 %), ``_quote`` montant en devise de cotation,
``_gb`` gigaoctets.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from qlab.core.errors import QlabError

BASE_FILE = "base.yaml"
INTRADAY_FILE = "intraday.yaml"
LONGTERM_FILE = "longterm.yaml"
MINUTES_PER_DAY = 1440


class ConfigError(QlabError, ValueError):
    """Configuration absente, incomplète, mal typée ou hors domaine."""


# --- validateurs -------------------------------------------------------------------------


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _positive(name: str, x: float) -> None:
    _check(x > 0, f"{name} doit être > 0 (reçu {x})")


def _non_negative(name: str, x: float) -> None:
    _check(x >= 0, f"{name} doit être ≥ 0 (reçu {x})")


def _fraction(name: str, x: float) -> None:
    """Fraction dans ]0, 1]."""
    _check(0 < x <= 1, f"{name} doit être dans ]0, 1] (reçu {x})")


def _increasing(name: str, xs: Sequence[float]) -> None:
    """Liste non vide, strictement croissante, de valeurs > 0."""
    _check(len(xs) > 0, f"{name} ne doit pas être vide")
    _positive(f"{name}[0]", xs[0])
    _check(
        all(a < b for a, b in itertools.pairwise(xs)),
        f"{name} doit être strictement croissant (reçu {list(xs)})",
    )


def _unique(name: str, xs: Sequence[str]) -> None:
    _check(len(set(xs)) == len(xs), f"{name} contient des doublons (reçu {list(xs)})")


def _url(name: str, url: str, scheme: str) -> None:
    _check(url.startswith(f"{scheme}://"), f"{name} doit commencer par {scheme}:// (reçu {url})")


def _under_windows_mount(path: Path) -> bool:
    return any(p.parts[:2] == ("/", "mnt") for p in (path, path.resolve()))


# --- base.yaml ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: Path
    disk_budget_gb: float
    disk_alert_fraction: float

    def __post_init__(self) -> None:
        _check(self.root.is_absolute(), f"root doit être un chemin absolu (reçu {self.root})")
        _check(
            not _under_windows_mount(self.root),
            f"root ne doit pas être sous /mnt (disque Windows) : {self.root}",
        )
        _positive("disk_budget_gb", self.disk_budget_gb)
        _fraction("disk_alert_fraction", self.disk_alert_fraction)


@dataclass(frozen=True, slots=True)
class FeesConfig:
    """Barème de repli, appliqué tant qu'aucun snapshot de frais réel n'est disponible.

    Frais en fraction du notional (0.001 = 0,1 %). ``bnb_discount_frac`` : remise appliquée
    si les frais sont payés en BNB. ``snapshot_refresh_hours`` : période de relecture des frais
    réels par ``exchange_info.py`` ; tout changement déclenche le recalcul des coûts.
    """

    maker_frac: float
    taker_frac: float
    bnb_discount_frac: float
    pay_in_bnb: bool
    snapshot_refresh_hours: int

    def __post_init__(self) -> None:
        for name in ("maker_frac", "taker_frac", "bnb_discount_frac"):
            x: float = getattr(self, name)
            _check(0 <= x < 1, f"{name} doit être dans [0, 1[ (reçu {x})")
        _positive("snapshot_refresh_hours", self.snapshot_refresh_hours)

    @property
    def effective_maker_frac(self) -> float:
        """Frais maker après remise BNB éventuelle, en fraction du notional."""
        return (
            self.maker_frac * (1 - self.bnb_discount_frac) if self.pay_in_bnb else self.maker_frac
        )

    @property
    def effective_taker_frac(self) -> float:
        """Frais taker après remise BNB éventuelle, en fraction du notional."""
        return (
            self.taker_frac * (1 - self.bnb_discount_frac) if self.pay_in_bnb else self.taker_frac
        )


@dataclass(frozen=True, slots=True)
class ExchangeConfig:
    name: str
    rest_url: str
    ws_url: str
    fees: FeesConfig

    def __post_init__(self) -> None:
        _url("rest_url", self.rest_url, "https")
        _url("ws_url", self.ws_url, "wss")


@dataclass(frozen=True, slots=True)
class ArchivesConfig:
    binance_vision_url: str
    tardis_url: str

    def __post_init__(self) -> None:
        _url("binance_vision_url", self.binance_vision_url, "https")
        _url("tardis_url", self.tardis_url, "https")


@dataclass(frozen=True, slots=True)
class SymbolsConfig:
    quote_asset: str
    intraday: tuple[str, ...]

    def __post_init__(self) -> None:
        _check(len(self.intraday) > 0, "intraday ne doit pas être vide")
        _unique("intraday", self.intraday)
        for sym in (self.quote_asset, *self.intraday):
            _check(sym.isalnum() and sym.isupper(), f"symbole invalide : {sym!r}")
        for sym in self.intraday:
            _check(
                sym.endswith(self.quote_asset) and sym != self.quote_asset,
                f"{sym} n'est pas coté en {self.quote_asset}",
            )


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_daily_loss_frac: float
    max_open_positions: int
    max_order_frac: float

    def __post_init__(self) -> None:
        _fraction("max_daily_loss_frac", self.max_daily_loss_frac)
        _check(self.max_open_positions >= 1, "max_open_positions doit être ≥ 1")
        _fraction("max_order_frac", self.max_order_frac)


@dataclass(frozen=True, slots=True)
class CapitalTier:
    name: str
    capital_quote: float
    max_drawdown_frac: float

    def __post_init__(self) -> None:
        _positive("capital_quote", self.capital_quote)
        _fraction("max_drawdown_frac", self.max_drawdown_frac)


@dataclass(frozen=True, slots=True)
class BaseConfig:
    data: DataConfig
    exchanges: tuple[ExchangeConfig, ...]
    archives: ArchivesConfig
    symbols: SymbolsConfig
    risk: RiskConfig
    capital_tiers: tuple[CapitalTier, ...]

    def __post_init__(self) -> None:
        _check(len(self.exchanges) > 0, "exchanges ne doit pas être vide")
        _unique("exchanges.name", [e.name for e in self.exchanges])
        _unique("capital_tiers.name", [t.name for t in self.capital_tiers])
        _increasing("capital_tiers.capital_quote", [t.capital_quote for t in self.capital_tiers])

    def exchange(self, name: str) -> ExchangeConfig:
        """Configuration de l'échange ``name`` ; ``ConfigError`` s'il n'est pas déclaré."""
        for ex in self.exchanges:
            if ex.name == name:
                return ex
        raise ConfigError(f"échange non configuré : {name!r}")

    def tier_for(self, net_deposits_quote: float) -> CapitalTier:
        """Palier applicable au capital apporté (dépôts − retraits, en devise de cotation).

        Le palier ne dépend jamais de la valeur de marché du portefeuille : sinon une perte
        ferait redescendre de palier et relâcherait la limite de drawdown, et un capital
        oscillant autour d'un seuil changerait de palier à chaque mouvement de prix. On monte
        de palier en déposant, jamais grâce aux gains ; les pertes sont gérées par le kill
        switch (drawdown mesuré depuis le plus haut), pas par un changement de palier.

        Retourne le plus haut palier dont le seuil est ≤ ``net_deposits_quote`` ; sous le
        premier seuil, le premier palier.
        """
        _check(
            math.isfinite(net_deposits_quote) and net_deposits_quote > 0,
            f"net_deposits_quote doit être > 0 et fini (reçu {net_deposits_quote})",
        )
        eligible = [t for t in self.capital_tiers if t.capital_quote <= net_deposits_quote]
        return eligible[-1] if eligible else self.capital_tiers[0]


# --- intraday.yaml -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateConfig:
    horizons_s: tuple[int, ...]
    min_move_cost_ratio: float
    slot_minutes: int
    slippage_frac: float

    def __post_init__(self) -> None:
        _increasing("horizons_s", self.horizons_s)
        _positive("min_move_cost_ratio", self.min_move_cost_ratio)
        _check(
            self.slot_minutes > 0 and MINUTES_PER_DAY % self.slot_minutes == 0,
            f"slot_minutes doit diviser 1440 (reçu {self.slot_minutes})",
        )
        _non_negative("slippage_frac", self.slippage_frac)


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    data_ms: int
    order_ms: int

    def __post_init__(self) -> None:
        _non_negative("data_ms", self.data_ms)
        _non_negative("order_ms", self.order_ms)


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    flow_windows_s: tuple[int, ...]
    ofi_depth_window_s: int
    vpin_buckets_per_day: int
    vpin_n_buckets: int
    realized_spread_delay_ms: int
    ewma_lambda: float
    signature_freqs_s: tuple[int, ...]
    seasonality_slot_minutes: int

    def __post_init__(self) -> None:
        _increasing("flow_windows_s", self.flow_windows_s)
        _increasing("signature_freqs_s", self.signature_freqs_s)
        for name in (
            "ofi_depth_window_s",
            "vpin_buckets_per_day",
            "vpin_n_buckets",
            "realized_spread_delay_ms",
        ):
            _positive(name, getattr(self, name))
        _check(0 < self.ewma_lambda < 1, f"ewma_lambda doit être dans ]0, 1[ ({self.ewma_lambda})")
        _check(
            self.seasonality_slot_minutes > 0
            and MINUTES_PER_DAY % self.seasonality_slot_minutes == 0,
            f"seasonality_slot_minutes doit diviser 1440 ({self.seasonality_slot_minutes})",
        )


@dataclass(frozen=True, slots=True)
class IntradayRiskConfig:
    freshness_max_ms: int

    def __post_init__(self) -> None:
        _positive("freshness_max_ms", self.freshness_max_ms)


@dataclass(frozen=True, slots=True)
class IntradayConfig:
    gate: GateConfig
    latency: LatencyConfig
    features: FeaturesConfig
    risk: IntradayRiskConfig


# --- longterm.yaml -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    warmup_days: int

    def __post_init__(self) -> None:
        _non_negative("warmup_days", self.warmup_days)


@dataclass(frozen=True, slots=True)
class SignalsConfig:
    momentum_lookbacks_days: tuple[int, ...]
    ma_fast_days: tuple[int, ...]
    ma_slow_days: tuple[int, ...]
    xs_skip_days: int
    xs_top_k: int
    vol_lookback_days: int

    def __post_init__(self) -> None:
        _increasing("momentum_lookbacks_days", self.momentum_lookbacks_days)
        _increasing("ma_fast_days", self.ma_fast_days)
        _increasing("ma_slow_days", self.ma_slow_days)
        _check(
            self.ma_fast_days[-1] < self.ma_slow_days[0],
            "chaque ma_fast_days doit être < chaque ma_slow_days",
        )
        _non_negative("xs_skip_days", self.xs_skip_days)
        _check(
            self.xs_skip_days < self.momentum_lookbacks_days[0],
            "xs_skip_days doit être < au plus petit momentum_lookbacks_days",
        )
        _positive("xs_top_k", self.xs_top_k)
        _check(self.vol_lookback_days >= 2, "vol_lookback_days doit être ≥ 2")


@dataclass(frozen=True, slots=True)
class VolTargetConfig:
    target_vol_annual: float
    w_max: float

    def __post_init__(self) -> None:
        _positive("target_vol_annual", self.target_vol_annual)
        _fraction("w_max", self.w_max)


@dataclass(frozen=True, slots=True)
class RebalanceConfig:
    calendar_days: tuple[int, ...]
    band_fracs: tuple[float, ...]

    def __post_init__(self) -> None:
        _increasing("calendar_days", self.calendar_days)
        _increasing("band_fracs", self.band_fracs)
        _fraction("band_fracs[-1]", self.band_fracs[-1])


@dataclass(frozen=True, slots=True)
class DcaConfig:
    amount_quote: float
    period_days: int

    def __post_init__(self) -> None:
        _positive("amount_quote", self.amount_quote)
        _positive("period_days", self.period_days)


@dataclass(frozen=True, slots=True)
class LtCostsConfig:
    fallback_spread_frac: float
    slippage_frac: float
    eur_conversion_cost_frac: float
    max_drag_edge_fraction: float

    def __post_init__(self) -> None:
        _positive("fallback_spread_frac", self.fallback_spread_frac)
        _non_negative("slippage_frac", self.slippage_frac)
        _non_negative("eur_conversion_cost_frac", self.eur_conversion_cost_frac)
        _fraction("max_drag_edge_fraction", self.max_drag_edge_fraction)


@dataclass(frozen=True, slots=True)
class LongtermConfig:
    universe: UniverseConfig
    signals: SignalsConfig
    vol_target: VolTargetConfig
    rebalance: RebalanceConfig
    dca: DcaConfig
    costs: LtCostsConfig


@dataclass(frozen=True, slots=True)
class QlabConfig:
    base: BaseConfig
    intraday: IntradayConfig
    longterm: LongtermConfig


# --- conversion YAML → dataclasses -------------------------------------------------------


def _convert(tp: Any, value: object, path: str) -> object:
    """Convertit ``value`` vers le type annoté ``tp`` ; strict (pas de bool pour un int, etc.)."""
    if get_origin(tp) is tuple:
        item_tp = get_args(tp)[0]
        if not isinstance(value, list):
            raise ConfigError(f"{path} : liste attendue, reçu {type(value).__name__}")
        return tuple(_convert(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value))
    if isinstance(tp, type) and is_dataclass(tp):
        return _build(tp, value, path)
    if tp is bool and isinstance(value, bool):
        return value
    if tp is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if tp is float and isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ConfigError(f"{path} : nombre fini attendu, reçu {value}")
        return float(value)
    if tp in (str, Path) and isinstance(value, str):
        if not value.strip():
            raise ConfigError(f"{path} : chaîne vide")
        return Path(value).expanduser() if tp is Path else value
    if tp not in (bool, int, float, str, Path):
        raise TypeError(f"{path} : type d'annotation non géré {tp!r}")
    raise ConfigError(f"{path} : {tp.__name__} attendu, reçu {type(value).__name__} ({value!r})")


def _build[T](cls: type[T], raw: object, path: str) -> T:
    """Construit la dataclass ``cls`` depuis un dict YAML ; clés manquantes/inconnues interdites."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} : table attendue, reçu {type(raw).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    keys = {str(k) for k in raw}
    missing = [n for n in names if n not in keys]
    unknown = sorted(keys - set(names))
    if missing:
        raise ConfigError(f"{path} : clé(s) manquante(s) {missing}")
    if unknown:
        raise ConfigError(f"{path} : clé(s) inconnue(s) {unknown}")
    kwargs = {n: _convert(hints[n], raw[n], f"{path}.{n}") for n in names}
    try:
        return cls(**kwargs)
    except ConfigError as exc:
        raise ConfigError(f"{path} : {exc}") from exc


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"lecture impossible de {path} : {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {path} : {exc}") from exc


def load_base(path: Path) -> BaseConfig:
    return _build(BaseConfig, _read_yaml(path), "base")


def load_intraday(path: Path) -> IntradayConfig:
    return _build(IntradayConfig, _read_yaml(path), "intraday")


def load_longterm(path: Path) -> LongtermConfig:
    return _build(LongtermConfig, _read_yaml(path), "longterm")


def load_config(config_dir: Path) -> QlabConfig:
    """Charge et valide ``base.yaml``, ``intraday.yaml`` et ``longterm.yaml`` de ``config_dir``."""
    return QlabConfig(
        base=load_base(config_dir / BASE_FILE),
        intraday=load_intraday(config_dir / INTRADAY_FILE),
        longterm=load_longterm(config_dir / LONGTERM_FILE),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Valide la config et l'affiche en JSON : ``python -m qlab.core.config --config config``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, required=True, help="dossier des YAML")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"ERREUR de configuration : {exc}")
        return 1
    print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
