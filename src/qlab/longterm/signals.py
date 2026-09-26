"""Signaux des fiches long terme (vague 1) → poids cibles par actif, spot sans vente à découvert.

Convention unique : un tableau **large** ``date_ms`` + une colonne par actif, sur une grille
journalière **complète** (``wide``) : décaler de ``L`` lignes = remonter de ``L`` jours, et un
jour absent reste nul (jamais comblé). Le poids de la ligne ``t`` est décidé après la clôture de
``t`` avec les seules données ≤ ``t`` ; il est détenu à partir de ``t + 1`` (le décalage
d'exécution est celui du backtest). Poids ≥ 0, somme ≤ 1, le reste en cash ; un actif absent
ce jour-là reçoit 0.

Paramètres = ceux des fiches (``hypotheses/lt_*.yaml``), chaque combinaison est un essai.
Aucun calendrier supposé : une fenêtre « 1 an » est passée en jours par l'appelant.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

from qlab.core.errors import DataError
from qlab.core.timeutils import MS_PER_DAY, date_str

DATE = "date_ms"


def wide(series: Mapping[str, pl.DataFrame], value: str = "close") -> pl.DataFrame:
    """``{actif: DataFrame(open_time_ms, value)}`` → grille journalière complète, nuls = absents."""
    if not series:
        raise DataError("aucune série")
    times = np.concatenate([df["open_time_ms"].to_numpy() for df in series.values()])
    if times.size == 0:
        raise DataError("séries toutes vides")
    days = pl.int_range(int(times.min()), int(times.max()) + 1, MS_PER_DAY, eager=True)
    grid = pl.DataFrame({DATE: days})
    for name, df in series.items():
        col = df.select(pl.col("open_time_ms").alias(DATE), pl.col(value).alias(name))
        grid = grid.join(col, on=DATE, how="left")
    return grid


def _assets(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c != DATE]


def _check_days(**days: int) -> None:
    bad = {k: v for k, v in days.items() if v < 1}
    if bad:
        raise DataError(f"longueurs en jours ≥ 1 attendues : {bad}")


def _equal_sleeves(closes: pl.DataFrame, on: pl.DataFrame) -> pl.DataFrame:
    """Chaque actif présent reçoit 1/N du capital (N = présents du jour), investi si ``on``."""
    assets = _assets(closes)
    present = pl.sum_horizontal(pl.col(a).is_not_null().cast(pl.Float64) for a in assets)
    joined = closes.join(on.rename({a: f"on_{a}" for a in assets}), on=DATE)
    return joined.select(
        DATE,
        *(
            pl.when(pl.col(a).is_not_null() & pl.col(f"on_{a}").fill_null(False))
            .then(1.0 / present)
            .otherwise(0.0)
            .alias(a)
            for a in assets
        ),
    )


def equal_weight(closes: pl.DataFrame, members: pl.DataFrame | None = None) -> pl.DataFrame:
    """Référence : panier équipondéré des actifs présents chaque jour (1/N chacun) ;
    ``members`` (booléens, même grille) : seulement les membres de l'univers du jour."""
    if members is not None:
        closes = closes.with_columns(
            pl.when(pl.lit(members[a].fill_null(False))).then(pl.col(a)).alias(a)
            for a in _assets(closes)
        )
    return _equal_sleeves(
        closes, closes.select(DATE, *(pl.lit(True).alias(a) for a in _assets(closes)))
    )


def log_return(closes: pl.DataFrame, days: int, skip: int = 0) -> pl.DataFrame:
    """ln(C_{t−skip} / C_{t−days}) par actif ; nul si une des deux clôtures manque."""
    _check_days(days=days)
    if not 0 <= skip < days:
        raise DataError("skip doit être dans [0, days[")
    ratios = [(pl.col(a).shift(skip) / pl.col(a).shift(days)).log() for a in _assets(closes)]
    return closes.select(DATE, *ratios)


def ts_momentum(closes: pl.DataFrame, lookback_days: int) -> pl.DataFrame:
    """``lt_ts_momentum`` : poche de 1/N investie si ln(C_t / C_{t−L}) > 0, sinon cash."""
    ret = log_return(closes, lookback_days)
    return _equal_sleeves(closes, ret.select(DATE, *(pl.col(a) > 0 for a in _assets(ret))))


def market_breadth(closes: pl.DataFrame, breadth: pl.DataFrame, min_breadth: float) -> pl.DataFrame:
    """``lt_market_breadth`` : panier équipondéré des actifs, investi si largeur ≥ seuil.
    ``breadth`` : ``date_ms``, ``breadth`` (``market_state``)."""
    if not 0 <= min_breadth <= 1:
        raise DataError("min_breadth dans [0, 1] attendu")
    on = closes.select(DATE).join(breadth.select(DATE, "breadth"), on=DATE, how="left")
    flag = pl.col("breadth") >= min_breadth
    return _equal_sleeves(closes, on.select(DATE, *(flag.alias(a) for a in _assets(closes))))


def funding_leverage(
    closes: pl.DataFrame, funding: pl.DataFrame, quantile: float, window_days: int
) -> pl.DataFrame:
    """``lt_funding_leverage`` : panier équipondéré, en cash si le financement moyen du jour
    (``funding`` : ``date_ms``, ``funding_1d``, moyenne des contrats des actifs) dépasse son
    quantile ``quantile`` sur les ``window_days`` derniers jours (jour compris ; fenêtre
    complète exigée). Financement absent ⇒ cash (prudence)."""
    _check_days(window_days=window_days)
    if not 0 < quantile < 1:
        raise DataError("quantile dans ]0, 1[ attendu")
    f = closes.select(DATE).join(funding.select(DATE, "funding_1d"), on=DATE, how="left")
    limit = pl.col("funding_1d").rolling_quantile(
        quantile, window_size=window_days, min_samples=window_days
    )
    calm = (pl.col("funding_1d") <= limit).fill_null(False)
    return _equal_sleeves(closes, f.select(DATE, *(calm.alias(a) for a in _assets(closes))))


def low_volatility(closes: pl.DataFrame, vol_lookback_days: int) -> pl.DataFrame:
    """``lt_low_volatility`` : w_i ∝ 1 / σ_i (écart-type des rendements journaliers sur la
    fenêtre, complète exigée), somme 1 sur les actifs mesurables du jour."""
    _check_days(vol_lookback_days=vol_lookback_days)
    if vol_lookback_days < 2:
        raise DataError("vol_lookback_days ≥ 2 attendu")
    n, assets = vol_lookback_days, _assets(closes)
    sigma = [pl.col(a).log().diff().rolling_std(n, min_samples=n) for a in assets]
    inv = closes.select(DATE, *sigma).select(DATE, *((1 / pl.col(a)).alias(a) for a in assets))
    total = pl.sum_horizontal(pl.col(a).fill_null(0.0) for a in assets)
    return inv.select(DATE, *((pl.col(a) / total).fill_null(0.0).fill_nan(0.0) for a in assets))


def top_k(scores: pl.DataFrame, k: int, members: pl.DataFrame | None = None) -> pl.DataFrame:
    """Les ``k`` meilleurs scores du jour à poids égaux 1/k (moins d'actifs ⇒ cash pour le
    reste). ``members`` (même grille, booléens) restreint à l'univers du jour."""
    if k < 1:
        raise DataError("top_k ≥ 1 attendu")
    assets = _assets(scores)
    values = scores.select(assets).to_numpy().astype(float)
    if members is not None:
        allowed = members.select(assets).to_numpy()
        values = np.where(allowed == True, values, np.nan)  # noqa: E712 (nuls → faux)
    ranks = np.argsort(np.argsort(-np.nan_to_num(values, nan=-np.inf), axis=1), axis=1)
    chosen = (ranks < k) & np.isfinite(values)
    return pl.DataFrame({DATE: scores[DATE], **{a: chosen[:, i] / k for i, a in enumerate(assets)}})


def xs_momentum(
    closes: pl.DataFrame, members: pl.DataFrame, lookback_days: int, skip_days: int, k: int
) -> pl.DataFrame:
    """``lt_xs_momentum`` : classement sur ln(C_{t−S} / C_{t−L}) parmi les membres du jour."""
    return top_k(log_return(closes, lookback_days, skip_days), k, members)


def short_reversal(closes: pl.DataFrame, drop_lookback_days: int) -> pl.DataFrame:
    """``lt_short_reversal`` : tout sur l'actif à la plus forte baisse sur la fenêtre, si
    baisse il y a (rendement < 0) ; sinon cash. Détention : horizon de la fiche (backtest)."""
    ret = log_return(closes, drop_lookback_days)
    drops = [pl.when(pl.col(a) < 0).then(-pl.col(a)).alias(a) for a in _assets(ret)]
    falling = ret.select(DATE, *drops)
    return top_k(falling, 1)


def turn_of_month(closes: pl.DataFrame, pre_days: int, post_days: int) -> pl.DataFrame:
    """``lt_turn_of_month`` : panier équipondéré détenu le **lendemain** ``t + 1`` s'il est
    parmi les ``pre_days`` derniers jours du mois ou les ``post_days`` premiers (calendrier
    connu d'avance : aucune fuite)."""
    _check_days(pre_days=pre_days, post_days=post_days)
    tomorrow = pl.from_epoch(pl.col(DATE) + MS_PER_DAY, time_unit="ms")
    day, last = tomorrow.dt.day(), tomorrow.dt.month_end().dt.day()
    window = (day <= post_days) | (day > last - pre_days)
    return _equal_sleeves(closes, closes.select(DATE, *(window.alias(a) for a in _assets(closes))))


def describe(weights: pl.DataFrame) -> str:
    """Résumé lisible : période, exposition moyenne, jours investis."""
    exposure = weights.select(pl.sum_horizontal(pl.exclude(DATE)))[:, 0].to_numpy()
    dates = weights[DATE].to_numpy()
    return (
        f"{date_str(int(dates.min()))} → {date_str(int(dates.max()))} : exposition moyenne "
        f"{exposure.mean():.2f}, investi {(exposure > 0).mean():.0%} des jours"
    )


# --- vague 2 -----------------------------------------------------------------------------


def relative_rotation(closes: pl.DataFrame, anchor: str, lookback_days: int) -> pl.DataFrame:
    """``lt_btc_alt_rotation`` : les autres actifs (poids égaux entre présents) si leur
    rendement log moyen sur L jours dépasse celui de ``anchor`` (BTC), sinon tout sur
    ``anchor``. Rendement inconnu (début de série) ⇒ cash."""
    ret = log_return(closes, lookback_days)
    others = [a for a in _assets(closes) if a != anchor]
    if anchor not in closes.columns or not others:
        raise DataError(f"rotation : {anchor} et au moins un autre actif attendus")
    mean_others = pl.mean_horizontal(*others)
    rotate = mean_others > pl.col(anchor)
    known = mean_others.is_not_null() & pl.col(anchor).is_not_null()
    present = pl.sum_horizontal(pl.col(a).is_not_null().cast(pl.Float64) for a in others)
    alt_weight = [
        pl.when(known & rotate & pl.col(a).is_not_null()).then(1 / present).otherwise(0.0).alias(a)
        for a in others
    ]
    btc_weight = pl.when(known & ~rotate).then(1.0).otherwise(0.0).alias(anchor)
    return ret.select(DATE, btc_weight, *alt_weight).select(DATE, *_assets(closes))


def _hold_flags(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """État « investi » d'un actif : entre à ``entry``, sort à ``exit_`` (sortie prioritaire)."""
    held, state = np.zeros(entry.size, dtype=bool), False
    for t in range(entry.size):
        state = False if exit_[t] else state or bool(entry[t])
        held[t] = state
    return held


def breakout(closes: pl.DataFrame, entry_days: int) -> pl.DataFrame:
    """``lt_breakout`` : poche de 1/N investie après une clôture au-dessus du plus haut des
    ``entry_days`` jours **précédents**, jusqu'à une clôture sous le plus bas des
    ``entry_days // 2`` jours précédents (fenêtres complètes exigées)."""
    _check_days(entry_days=entry_days)
    if entry_days < 2:
        raise DataError("entry_days ≥ 2 attendu (sortie sur la moitié de la fenêtre)")
    exit_days = entry_days // 2
    flags = {}
    for a in _assets(closes):
        c = pl.col(a)
        high = c.shift(1).rolling_max(entry_days, min_samples=entry_days)
        low = c.shift(1).rolling_min(exit_days, min_samples=exit_days)
        sig = closes.select(
            (c > high).fill_null(False).alias("entry"), (c < low).fill_null(False).alias("exit")
        )
        flags[a] = _hold_flags(sig[:, 0].to_numpy(), sig[:, 1].to_numpy())
    return _equal_sleeves(closes, pl.DataFrame({DATE: closes[DATE], **flags}))


def volume_shock(
    closes: pl.DataFrame,
    volumes: pl.DataFrame,
    *,
    ratio: float,
    baseline_days: int,
    hold_days: int,
) -> pl.DataFrame:
    """``lt_volume_shock`` : poche de 1/N investie ``hold_days`` jours après un jour haussier
    dont le volume ≥ ``ratio`` × médiane des ``baseline_days`` jours précédents ; un nouveau
    choc prolonge la détention."""
    _check_days(baseline_days=baseline_days, hold_days=hold_days)
    if ratio <= 1 or volumes[DATE].to_list() != closes[DATE].to_list():
        raise DataError("ratio > 1 et volumes sur la même grille que les prix attendus")
    flags = {}
    for a in _assets(closes):
        base = pl.col(a).shift(1).rolling_median(baseline_days, min_samples=baseline_days)
        shock = volumes.select((pl.col(a) >= ratio * base).fill_null(False))[:, 0].to_numpy()
        up = closes.select((pl.col(a) > pl.col(a).shift(1)).fill_null(False))[:, 0].to_numpy()
        started = np.flatnonzero(shock & up)
        held = np.zeros(closes.height, dtype=bool)
        for t in started:
            held[t : t + hold_days] = True
        flags[a] = held
    return _equal_sleeves(closes, pl.DataFrame({DATE: closes[DATE], **flags}))


def near_high(closes: pl.DataFrame, min_ratio: float, window_days: int) -> pl.DataFrame:
    """``lt_near_high`` : poche de 1/N investie si clôture ≥ ``min_ratio`` × plus haut des
    ``window_days`` derniers jours, jour compris (fenêtre complète exigée ; « 1 an » : le
    calendrier du marché, fourni par l'appelant)."""
    _check_days(window_days=window_days)
    if not 0 < min_ratio <= 1:
        raise DataError("min_ratio dans ]0, 1] attendu")
    near = [
        (pl.col(a) >= min_ratio * pl.col(a).rolling_max(window_days, min_samples=window_days))
        .fill_null(False)
        .alias(a)
        for a in _assets(closes)
    ]
    return _equal_sleeves(closes, closes.select(DATE, *near))
