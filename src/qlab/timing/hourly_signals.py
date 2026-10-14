"""Signaux horaires du volet timing : un nombre par (contrat, heure), **orienté** (plus grand ⇒
hausse attendue plus forte), calculé avec les seules bougies closes jusqu'à ``t`` inclus.

Chaque signal correspond à une fiche ``hypotheses/timing/*.yaml`` (mécanisme écrit avant les
tests) ; ``lookback_h`` = k heures. Les expressions s'évaluent sur un panel trié par
(``symbol``, ``t_ms``) à grille complète (``timing/panel.py``) : ``shift(k)`` = k heures. Un trou
dans la fenêtre donne un signal nul (contrat ignoré à cette heure).

| nom | formule (par contrat) |
|---|---|
| ``momentum`` | ln(close_t / close_{t−k}) |
| ``reversal`` | − momentum |
| ``flow`` | Σ_k (2 · achats agressifs − volume) / Σ_k volume (déséquilibre ∈ [−1, 1]) |
| ``flow_reversal`` | − flow |
| ``volume_shock`` | ln(volume des k h / (k · moyenne horaire des 30 j d'avant)) · signe(momentum) |
| ``funding_contrarian`` | − Σ_k financement (foule acheteuse à levier ⇒ baisse) |
| ``btc_catchup`` | momentum_k(référence) − momentum_k(contrat) (retard sur BTC) |
| ``breakout`` | close_t / max(high des k h précédentes) − 1 |
| ``low_volatility`` | − écart-type des rendements log horaires sur k h |
| ``channel`` | position de close_t dans [min low, max high] des k h (canal de Donchian) − ½ |
| ``seasonality`` | rendement moyen à l'horizon h, mêmes heures de la semaine, semaines passées |

Fiches qui gardent une formule et changent le **portefeuille** (signé, long seul, horizons
longs) : ``ALIASES`` donne le signal qu'elles utilisent (``trend`` ⇒ ``momentum``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import polars as pl

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, MS_PER_HOUR
from qlab.timing.panel import forward_return

HOURS_PER_DAY = MS_PER_DAY // MS_PER_HOUR
HOURS_PER_WEEK = 7 * HOURS_PER_DAY
BASELINE_H = 30 * HOURS_PER_DAY  # volume « normal » : moyenne horaire des 30 jours précédents
REFERENCE_COLUMN = "ref_close"


def _per(expr: pl.Expr) -> pl.Expr:
    return expr.over("symbol")


def _momentum(k: int) -> pl.Expr:
    return _per((pl.col("close") / pl.col("close").shift(k)).log())


def _flow(k: int) -> pl.Expr:
    imbalance = 2 * pl.col("taker_buy_quote") - pl.col("volume_quote")
    total = pl.col("volume_quote").rolling_sum(k)
    return _per(pl.when(total > 0).then(imbalance.rolling_sum(k) / total))


def _volume_shock(k: int) -> pl.Expr:
    recent = pl.col("volume_quote").rolling_sum(k)
    normal = pl.col("volume_quote").rolling_mean(BASELINE_H).shift(k) * k
    ratio = _per(pl.when(normal > 0).then(recent / normal))
    return ratio.log() * _momentum(k).sign()


def _catchup(k: int) -> pl.Expr:
    ref = _per((pl.col(REFERENCE_COLUMN) / pl.col(REFERENCE_COLUMN).shift(k)).log())
    return ref - _momentum(k)


def _breakout(k: int) -> pl.Expr:
    return _per(pl.col("close") / pl.col("high").rolling_max(k).shift(1) - 1)


def _low_volatility(k: int) -> pl.Expr:
    return -_per(pl.col("close").log().diff().rolling_std(k))


def _channel(k: int) -> pl.Expr:
    lo, hi = pl.col("low").rolling_min(k), pl.col("high").rolling_max(k)
    return _per(pl.when(hi > lo).then((pl.col("close") - lo) / (hi - lo) - 0.5))


def _funding_contrarian(k: int) -> pl.Expr:
    return -_per(pl.col("funding").rolling_sum(k))


_BY_LOOKBACK: dict[str, Callable[[int], pl.Expr]] = {
    "momentum": _momentum,
    "reversal": lambda k: -_momentum(k),
    "flow": _flow,
    "flow_reversal": lambda k: -_flow(k),
    "volume_shock": _volume_shock,
    "funding_contrarian": _funding_contrarian,
    "btc_catchup": _catchup,
    "breakout": _breakout,
    "low_volatility": _low_volatility,
    "channel": _channel,
}
ALIASES = {"trend": "momentum", "momentum_slow": "momentum", "momentum_long_only": "momentum"}
NAMES = (*_BY_LOOKBACK, "seasonality", *ALIASES)


def resolve(name: str) -> str:
    """Signal réellement calculé pour ``name`` (lui-même, ou la formule de son alias)."""
    return ALIASES.get(name, name)


def seasonality(panel: pl.DataFrame, horizon_h: int) -> pl.Series:
    """Moyenne passée du rendement à ``horizon_h`` aux mêmes heures de la semaine. Le dernier
    rendement utilisé date d'une semaine : connu si l'horizon (+ 1 h d'exécution) < 1 semaine.
    Le rendement est matérialisé d'abord : une fenêtre par contrat ne s'imbrique pas dans une
    fenêtre par (contrat, heure de la semaine)."""
    if horizon_h + 1 >= HOURS_PER_WEEK:
        raise DataError(f"seasonality : horizon {horizon_h} h trop long (< 1 semaine requis)")
    slot = (pl.col("t_ms") // MS_PER_HOUR) % HOURS_PER_WEEK
    r = pl.col("_r")
    total = r.fill_null(0.0).cum_sum().shift(1).over("symbol", slot)
    count = r.is_not_null().cast(pl.Int64).cum_sum().shift(1).over("symbol", slot)
    with_r = panel.select("symbol", "t_ms", forward_return(horizon_h).alias("_r"))
    return with_r.select(pl.when(count > 0).then(total / count).alias("signal"))["signal"]


def compute(
    panel: pl.DataFrame, name: str, params: Mapping[str, float], horizon_h: int
) -> pl.Series:
    """Valeurs du signal ``name`` (ou de son alias) sur ``panel`` (paramètre ``lookback_h``,
    sauf ``seasonality`` qui dépend de l'horizon), dans l'ordre des lignes."""
    name = resolve(name)
    if name == "seasonality":
        return seasonality(panel, horizon_h)
    if name not in _BY_LOOKBACK:
        raise DataError(f"signal inconnu : {name!r} (connus : {', '.join(NAMES)})")
    k = int(params["lookback_h"])
    if k < 1 or k != params["lookback_h"]:
        raise DataError(f"{name} : lookback_h entier ≥ 1 attendu (reçu {params['lookback_h']})")
    return panel.select(_BY_LOOKBACK[name](k).alias("signal"))["signal"]


def with_reference(panel: pl.DataFrame, reference: str) -> pl.DataFrame:
    """Ajoute la clôture du contrat de référence (ex. BTCUSDT) à chaque heure (``btc_catchup``)."""
    ref = panel.filter(pl.col("symbol") == reference).select(
        "t_ms", pl.col("close").alias(REFERENCE_COLUMN)
    )
    if ref.is_empty():
        raise DataError(f"contrat de référence absent du panel : {reference}")
    return panel.join(ref, on="t_ms", how="left", maintain_order="left")
