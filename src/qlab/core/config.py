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
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from qlab.core.errors import run_cli
from qlab.core.yamlschema import SchemaError, build, read_yaml

BASE_FILE = "base.yaml"
INTRADAY_FILE = "intraday.yaml"
LONGTERM_FILE = "longterm.yaml"
MINUTES_PER_DAY = 1440


class ConfigError(SchemaError):
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
    si les frais sont payés en BNB. Ils sont enregistrés dans chaque snapshot ``exchangeInfo`` ;
    un changement de barème crée une nouvelle version.
    """

    maker_frac: float
    taker_frac: float
    bnb_discount_frac: float
    pay_in_bnb: bool

    def __post_init__(self) -> None:
        for name in ("maker_frac", "taker_frac", "bnb_discount_frac"):
            x: float = getattr(self, name)
            _check(0 <= x < 1, f"{name} doit être dans [0, 1[ (reçu {x})")

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
    """Un échange. ``snapshot_refresh_hours`` : âge au-delà duquel ``exchange_info fetch
    --if-due`` reprend un snapshot. ``max_clock_offset_ms`` : écart toléré entre l'horloge
    locale et celle du serveur ; au-delà, alerte (journal, puis refus des modules sensibles).
    ``trading_days_per_year`` : calendrier du marché (crypto 24/7 : tous les jours ; actions :
    environ 252 séances) — annualisations et fenêtres « 1 an », jamais écrites dans le code."""

    name: str
    rest_url: str
    ws_url: str
    fees: FeesConfig
    snapshot_refresh_hours: int
    max_clock_offset_ms: int
    trading_days_per_year: int

    def __post_init__(self) -> None:
        _check(
            bool(re.match(r"^[a-z0-9_]+$", self.name)),
            f"name doit être en [a-z0-9_]+ (sert de dossier) : {self.name!r}",
        )
        _url("rest_url", self.rest_url, "https")
        _url("ws_url", self.ws_url, "wss")
        _positive("snapshot_refresh_hours", self.snapshot_refresh_hours)
        _positive("max_clock_offset_ms", self.max_clock_offset_ms)
        _positive("trading_days_per_year", self.trading_days_per_year)


@dataclass(frozen=True, slots=True)
class ArchivesConfig:
    """Archives publiques. ``binance_vision_list_url`` : listage S3 (pagination XML) des
    fichiers publiés sous ``binance_vision_url``. ``tardis_api_url`` : métadonnées Tardis (dates
    de disponibilité par paire). ``download_workers`` : téléchargements
    simultanés ; ``list_workers`` : listages simultanés (légers : réponses de quelques Ko).
    ``futures_metrics_symbols`` : paires USDⓈ-M dont on prend les métriques
    quotidiennes (un fichier par jour : limité à une liste ; le financement, mensuel, est pris
    pour toutes les paires)."""

    binance_vision_url: str
    binance_vision_list_url: str
    tardis_url: str
    tardis_api_url: str
    download_workers: int
    list_workers: int
    futures_metrics_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        _url("binance_vision_url", self.binance_vision_url, "https")
        _url("binance_vision_list_url", self.binance_vision_list_url, "https")
        _url("tardis_url", self.tardis_url, "https")
        _url("tardis_api_url", self.tardis_api_url, "https")
        _check(1 <= self.download_workers <= 32, "download_workers doit être dans [1, 32]")
        _check(1 <= self.list_workers <= 64, "list_workers doit être dans [1, 64]")
        _unique("futures_metrics_symbols", self.futures_metrics_symbols)
        for sym in self.futures_metrics_symbols:
            _check(sym.isalnum() and sym.isupper(), f"symbole invalide : {sym!r}")


@dataclass(frozen=True, slots=True)
class SymbolsConfig:
    """Univers **tradé** : paires cotées en ``quote_asset``. ``intraday`` ⊆ ``trade``."""

    quote_asset: str
    trade: tuple[str, ...]
    intraday: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("trade", "intraday"):
            pairs: tuple[str, ...] = getattr(self, name)
            _check(len(pairs) > 0, f"{name} ne doit pas être vide")
            _unique(name, pairs)
        for sym in (self.quote_asset, *self.trade, *self.intraday):
            _check(sym.isalnum() and sym.isupper(), f"symbole invalide : {sym!r}")
        for sym in self.trade:
            _check(
                sym.endswith(self.quote_asset) and sym != self.quote_asset,
                f"{sym} n'est pas coté en {self.quote_asset}",
            )
        extra = [s for s in self.intraday if s not in self.trade]
        _check(not extra, f"intraday doit être inclus dans trade : {extra} absents de trade")


BINANCE_KLINE_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)


@dataclass(frozen=True, slots=True)
class ObserveConfig:
    """Univers **observé** (vue globale) : toutes les paires spot Binance, toutes devises.

    ``include_delisted`` : garder les paires retirées (pas de biais du survivant).
    ``futures_metrics`` : taux de financement et positions ouvertes des contrats USDⓈ-M.
    ``kline_intervals`` : intervalles des bougies téléchargées ; ``1d`` obligatoire.
    """

    include_delisted: bool
    futures_metrics: bool
    kline_intervals: tuple[str, ...]

    def __post_init__(self) -> None:
        _unique("kline_intervals", self.kline_intervals)
        bad = [i for i in self.kline_intervals if i not in BINANCE_KLINE_INTERVALS]
        _check(not bad, f"kline_intervals inconnus de Binance : {bad}")
        _check("1d" in self.kline_intervals, "kline_intervals doit contenir 1d (volet long terme)")


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
    observe: ObserveConfig
    risk: RiskConfig
    capital_tiers: tuple[CapitalTier, ...]
    secrets_file: Path

    def __post_init__(self) -> None:
        _check(
            self.secrets_file.is_absolute(),
            f"secrets_file doit être un chemin absolu (reçu {self.secrets_file})",
        )
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

    def tier_for(self, net_deposits_quote: Decimal) -> CapitalTier:
        """Palier applicable au capital apporté (dépôts − retraits, en devise de cotation).

        Le palier ne dépend jamais de la valeur de marché du portefeuille : sinon une perte
        ferait redescendre de palier et relâcherait la limite de drawdown, et un capital
        oscillant autour d'un seuil changerait de palier à chaque mouvement de prix. On monte
        de palier en déposant, jamais grâce aux gains ; les pertes sont gérées par le kill
        switch (drawdown mesuré depuis le plus haut), pas par un changement de palier.

        Retourne le plus haut palier dont le seuil est ≤ ``net_deposits_quote`` ; sous le
        premier seuil, le premier palier. Montant en ``Decimal`` (argent, cf. ``ledger.py``) ;
        la comparaison ``Decimal`` / ``float`` de Python est exacte.
        """
        _check(
            isinstance(net_deposits_quote, Decimal),
            f"net_deposits_quote doit être un Decimal (reçu {net_deposits_quote!r})",
        )
        _check(
            net_deposits_quote.is_finite() and net_deposits_quote > 0,
            f"net_deposits_quote doit être > 0 et fini (reçu {net_deposits_quote})",
        )
        eligible = [t for t in self.capital_tiers if t.capital_quote <= net_deposits_quote]
        return eligible[-1] if eligible else self.capital_tiers[0]


# --- intraday.yaml -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Cost gate (§5). ``sample_step_ms`` : pas de la grille régulière d'échantillonnage (sinon
    les périodes agitées, riches en cotations, pèseraient plus que les calmes)."""

    horizons_s: tuple[int, ...]
    min_move_cost_ratio: float
    slot_minutes: int
    slippage_frac: float
    sample_step_ms: int

    def __post_init__(self) -> None:
        _increasing("horizons_s", self.horizons_s)
        _positive("min_move_cost_ratio", self.min_move_cost_ratio)
        _check(
            self.slot_minutes > 0 and MINUTES_PER_DAY % self.slot_minutes == 0,
            f"slot_minutes doit diviser 1440 (reçu {self.slot_minutes})",
        )
        _non_negative("slippage_frac", self.slippage_frac)
        _positive("sample_step_ms", self.sample_step_ms)
        _check(
            self.sample_step_ms <= self.horizons_s[0] * 1000,
            f"sample_step_ms ({self.sample_step_ms}) doit être ≤ au plus petit horizon",
        )


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
    """Univers point-in-time (``longterm/universe.py``).

    ``warmup_days`` : chauffe après la première bougie. Univers observé : un actif = une paire,
    cotée dans la première devise de ``reference_quotes`` disponible ce jour-là ; bases de
    ``excluded_bases`` (stablecoins, devises) et tokens à levier (base = autre base +
    ``leveraged_suffixes``) exclus ; liquidité : volume médian sur ``volume_lookback_days``
    ≥ ``min_volume_quote`` (en devise de cotation, donc ≈ dollars). Univers tradé élargi
    (``trade_eur``) : paires cotées dans la devise du compte, volume médian ≥
    ``quote_min_volume`` (en cette devise).
    """

    warmup_days: int
    reference_quotes: tuple[str, ...]
    excluded_bases: tuple[str, ...]
    leveraged_suffixes: tuple[str, ...]
    volume_lookback_days: int
    min_volume_quote: float
    quote_min_volume: float

    def __post_init__(self) -> None:
        _non_negative("warmup_days", self.warmup_days)
        _check(len(self.reference_quotes) > 0, "reference_quotes ne doit pas être vide")
        for name in ("reference_quotes", "excluded_bases", "leveraged_suffixes"):
            _unique(name, getattr(self, name))
        _positive("volume_lookback_days", self.volume_lookback_days)
        _positive("min_volume_quote", self.min_volume_quote)
        _positive("quote_min_volume", self.quote_min_volume)


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
    max_rejected_share: float
    spread_min_samples: int

    def __post_init__(self) -> None:
        _positive("fallback_spread_frac", self.fallback_spread_frac)
        _non_negative("slippage_frac", self.slippage_frac)
        _non_negative("eur_conversion_cost_frac", self.eur_conversion_cost_frac)
        _fraction("max_drag_edge_fraction", self.max_drag_edge_fraction)
        _fraction("max_rejected_share", self.max_rejected_share)
        _positive("spread_min_samples", self.spread_min_samples)


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    """Critères d'acceptation (SPEC_LONG_TERME LT.6) et réglages du bootstrap stationnaire
    (``research/report.py``). ``mean_block_days`` : longueur moyenne des blocs tirés."""

    dsr_min: float
    confidence: float
    min_subperiods: int
    min_assets: int
    require_bear: bool
    n_boot: int
    mean_block_days: float
    seed: int

    def __post_init__(self) -> None:
        _fraction("dsr_min", self.dsr_min)
        _fraction("confidence", self.confidence)
        _check(self.confidence < 1, "confidence doit être < 1")
        _positive("min_subperiods", self.min_subperiods)
        _positive("min_assets", self.min_assets)
        _positive("n_boot", self.n_boot)
        _check(self.mean_block_days >= 1, "mean_block_days doit être ≥ 1")
        _non_negative("seed", self.seed)


@dataclass(frozen=True, slots=True)
class LongtermConfig:
    universe: UniverseConfig
    signals: SignalsConfig
    vol_target: VolTargetConfig
    rebalance: RebalanceConfig
    dca: DcaConfig
    costs: LtCostsConfig
    acceptance: AcceptanceConfig


@dataclass(frozen=True, slots=True)
class QlabConfig:
    base: BaseConfig
    intraday: IntradayConfig
    longterm: LongtermConfig


def load_base(path: Path) -> BaseConfig:
    return build(BaseConfig, read_yaml(path, error=ConfigError), "base", error=ConfigError)


def load_intraday(path: Path) -> IntradayConfig:
    return build(IntradayConfig, read_yaml(path, error=ConfigError), "intraday", error=ConfigError)


def load_longterm(path: Path) -> LongtermConfig:
    return build(LongtermConfig, read_yaml(path, error=ConfigError), "longterm", error=ConfigError)


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

    def run() -> int:
        cfg = load_config(args.config)
        print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str, ensure_ascii=False))
        return 0

    return run_cli(run)


if __name__ == "__main__":
    raise SystemExit(main())
