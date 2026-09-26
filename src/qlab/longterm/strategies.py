"""Table des stratégies long terme : fiche → signal, données, règle de rééquilibrage.

Une seule table pour le gate de coûts (``lt_costs``) et le backtest : chaque fiche de
``hypotheses/`` (volet long terme) y a une entrée ; une fiche sans entrée est une erreur.

- **Données** (``Market``) : clôtures des paires tradées (``symbols.trade``) sur grille
  journalière, financement moyen de leurs contrats perpétuels (``{base}USDT``), largeur du
  marché calculée sur l'univers observé (``market_state``), calendrier du marché
  (``trading_days_per_year`` : la fenêtre « 1 an » de la fiche financement).
- **Rééquilibrage** : calendaire, tous les ``horizon_s`` de la fiche (en jours), sauf quand le
  signal fixe lui-même ses dates (fin de mois : chaque jour).
- Momentum transversal : la fiche le teste sur tout le marché (``research/ic.py``) et
  l'applique aux paires tradées ; c'est cette application qui est rejouée ici.

Commandes (gate de coûts de tous les paliers, verdict d'une vague) : ``longterm/lt_wave.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from qlab.core.config import QlabConfig, SignalsConfig
from qlab.core.errors import DataError
from qlab.core.paths import DataPaths
from qlab.core.timeutils import MS_PER_DAY, MS_PER_S
from qlab.exchange.snapshots import Snapshot
from qlab.longterm import funding, klines, market_state, universe
from qlab.longterm import signals as sg
from qlab.longterm.allocation import Policy
from qlab.research.hypothesis import Hypothesis


@dataclass(frozen=True, slots=True)
class Panel:
    """Un univers sur grille journalière : clôtures, ouvertures et, pour un univers qui change
    (``trade_eur``), ``members`` (booléens, même grille : paire éligible ce jour-là)."""

    closes: pl.DataFrame
    opens: pl.DataFrame
    members: pl.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class Market:
    closes: pl.DataFrame  # paires tradées, grille journalière (signals.wide)
    volumes: pl.DataFrame  # volumes en devise de cotation, même grille
    funding: pl.DataFrame  # date_ms, funding_1d (moyenne des contrats des paires tradées)
    breadth: Mapping[int, pl.DataFrame]  # ma_days → date_ms, breadth
    days_per_year: int
    settings: SignalsConfig
    bases: Mapping[str, str]  # paire → actif de base (exchangeInfo)
    quoted: Panel | None = None  # univers trade_eur (paires cotées dans la devise du compte)


Params = Mapping[str, float]


def _days(params: Params, name: str) -> int:
    value = params[name]
    if value != int(value):
        raise DataError(f"{name} = {value} : nombre de jours entier attendu")
    return int(value)


def _ts_momentum(m: Market, p: Params) -> pl.DataFrame:
    return sg.ts_momentum(m.closes, _days(p, "lookback_days"))


def _xs_momentum(m: Market, p: Params) -> pl.DataFrame:
    present = m.closes.select(sg.DATE, *(pl.col(c).is_not_null() for c in m.closes.columns[1:]))
    skip = m.settings.xs_skip_days
    return sg.xs_momentum(m.closes, present, _days(p, "lookback_days"), skip, _days(p, "top_k"))


def _low_volatility(m: Market, p: Params) -> pl.DataFrame:
    return sg.low_volatility(m.closes, _days(p, "vol_lookback_days"))


def _short_reversal(m: Market, p: Params) -> pl.DataFrame:
    return sg.short_reversal(m.closes, _days(p, "drop_lookback_days"))


def _funding_leverage(m: Market, p: Params) -> pl.DataFrame:
    return sg.funding_leverage(m.closes, m.funding, p["funding_quantile"], m.days_per_year)


def _market_breadth(m: Market, p: Params) -> pl.DataFrame:
    return sg.market_breadth(m.closes, m.breadth[_days(p, "ma_days")], p["min_breadth"])


def _turn_of_month(m: Market, p: Params) -> pl.DataFrame:
    return sg.turn_of_month(m.closes, _days(p, "pre_days"), _days(p, "post_days"))


# Constantes écrites dans les fiches de la vague 2 (hypotheses/lt_volume_shock.yaml).
VOLUME_BASELINE_DAYS = 30  # « médiane des 30 jours précédents »
VOLUME_HOLD_DAYS = 21  # « détention de 3 semaines » (horizon_s de la fiche)


def _btc_alt_rotation(m: Market, p: Params) -> pl.DataFrame:
    anchor = next((s for s, b in m.bases.items() if b == market_state.BTC), None)
    if anchor is None:
        raise DataError("rotation : aucune paire tradée sur BTC")
    return sg.relative_rotation(m.closes, anchor, _days(p, "lookback_days"))


def _breakout(m: Market, p: Params) -> pl.DataFrame:
    return sg.breakout(m.closes, _days(p, "entry_days"))


def _volume_shock(m: Market, p: Params) -> pl.DataFrame:
    return sg.volume_shock(
        m.closes,
        m.volumes,
        ratio=p["volume_ratio"],
        baseline_days=VOLUME_BASELINE_DAYS,
        hold_days=VOLUME_HOLD_DAYS,
    )


def _near_high(m: Market, p: Params) -> pl.DataFrame:
    return sg.near_high(m.closes, p["min_ratio"], m.days_per_year)


def _xs_momentum_eur(m: Market, p: Params) -> pl.DataFrame:
    q = m.quoted
    if q is None or q.members is None:
        raise DataError("univers trade_eur non chargé (load_market(..., quoted=True))")
    lookback, k = _days(p, "lookback_days"), _days(p, "top_k")
    return sg.xs_momentum(q.closes, q.members, lookback, m.settings.xs_skip_days, k)


@dataclass(frozen=True, slots=True)
class Strategy:
    """``multi_asset`` : la fiche répartit entre plusieurs actifs (sur un seul, elle se réduit
    au buy & hold) ; sa stabilité se vérifie sur des sous-paniers, pas actif par actif."""

    build: Callable[[Market, Params], pl.DataFrame]
    rebalance_days: int | None = None  # None : horizon de la fiche
    multi_asset: bool = False


REGISTRY: dict[str, Strategy] = {
    "lt_ts_momentum": Strategy(_ts_momentum),
    "lt_xs_momentum": Strategy(_xs_momentum, multi_asset=True),
    "lt_low_volatility": Strategy(_low_volatility, multi_asset=True),
    "lt_short_reversal": Strategy(_short_reversal),
    "lt_funding_leverage": Strategy(_funding_leverage),
    "lt_market_breadth": Strategy(_market_breadth),
    "lt_turn_of_month": Strategy(_turn_of_month, rebalance_days=1),
    "lt_btc_alt_rotation": Strategy(_btc_alt_rotation, multi_asset=True),
    "lt_breakout": Strategy(_breakout),
    "lt_volume_shock": Strategy(_volume_shock, rebalance_days=1),  # le choc fixe l'entrée
    "lt_near_high": Strategy(_near_high),
    "lt_xs_momentum_eur": Strategy(_xs_momentum_eur, multi_asset=True),
}


def policy_for(hypothesis: Hypothesis) -> Policy:
    """Calendaire : période = horizon de la fiche en jours (ou celle imposée par le signal)."""
    strategy = strategy_for(hypothesis)
    if strategy.rebalance_days is not None:
        return Policy("calendar", period_days=strategy.rebalance_days)
    days, rest = divmod(hypothesis.horizon_s * MS_PER_S, MS_PER_DAY)
    if rest or days < 1:
        raise DataError(f"{hypothesis.id} : horizon_s doit être un nombre entier de jours")
    return Policy("calendar", period_days=days)


def strategy_for(hypothesis: Hypothesis) -> Strategy:
    if hypothesis.id not in REGISTRY:
        raise DataError(f"{hypothesis.id} : aucune stratégie dans longterm/strategies.py")
    return REGISTRY[hypothesis.id]


def load_market(
    paths: DataPaths,
    config: QlabConfig,
    snapshot: Snapshot,
    ma_days: Sequence[int],
    exchange: str,
    *,
    quoted: bool = False,
) -> Market:
    """Charge les données de toutes les fiches (largeur seulement pour ``ma_days`` ; univers
    ``trade_eur`` seulement si ``quoted``)."""
    trade = config.base.symbols.trade
    bars = {s: klines.load(paths, universe.INTERVAL, s) for s in trade}
    closes, volumes = sg.wide(bars), sg.wide(bars, "volume_quote")
    known = snapshot.by_symbol()
    bases = {s: str(known[s]["baseAsset"]) for s in trade if s in known}
    contracts = [f"{known[s]['baseAsset']}USDT" for s in trade if s in known]
    rates = pl.concat([funding.daily(funding.load(paths, c)) for c in contracts])
    mean_rate = rates.group_by("date_ms").agg(pl.col("funding_1d").mean()).sort("date_ms")
    breadth: dict[int, pl.DataFrame] = {}
    if ma_days:
        members = universe.observed(paths, known, config.longterm.universe).members
        state = market_state.market_state(paths, members, ma_days)
        breadth = {n: state.select("date_ms", breadth=f"breadth_{n}") for n in ma_days}
    days = config.base.exchange(exchange).trading_days_per_year
    panel = quoted_panel(paths, config, known) if quoted else None
    return Market(closes, volumes, mean_rate, breadth, days, config.longterm.signals, bases, panel)


def quoted_panel(
    paths: DataPaths, config: QlabConfig, known: Mapping[str, Mapping[str, object]]
) -> Panel:
    """Univers ``trade_eur`` : prix de toutes les paires membres un jour ou l'autre, et
    matrice des membres jour par jour (``universe.quoted``)."""
    cfg, quote = config.longterm.universe, config.base.symbols.quote_asset
    members = universe.quoted(paths, known, cfg, quote).members
    if members.is_empty():
        raise DataError(f"univers {quote} vide : aucune paire assez liquide")
    symbols = members["symbol"].unique().sort().to_list()
    bars = {s: klines.load(paths, universe.INTERVAL, s) for s in symbols}
    closes = sg.wide(bars)
    flags = members.with_columns(member=pl.lit(value=True)).pivot(
        on="symbol", index="date_ms", values="member"
    )
    grid = closes.select(sg.DATE).join(flags, on=sg.DATE, how="left").fill_null(value=False)
    return Panel(closes, sg.wide(bars, "open"), grid.select(sg.DATE, *symbols))


def breadth_lengths(hypotheses: Sequence[Hypothesis]) -> list[int]:
    """Longueurs de moyenne (``ma_days``) dont les fiches ont besoin pour la largeur."""
    return sorted(
        {int(v) for h in hypotheses for p in h.parameters if p.name == "ma_days" for v in p.values}
    )


def trial_name(hypothesis: Hypothesis, params: Params) -> str:
    return f"{hypothesis.id} · " + ", ".join(f"{k}={v:g}" for k, v in params.items())
