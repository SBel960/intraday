"""Bandes d'entrée et de sortie : ne trader que les signaux forts, puis tenir la position.

Un signal brut (``timing/hourly_signals.py``) est mis à l'échelle de sa propre variabilité :
``z = s / σ(s)``, σ = écart-type des ``window`` valeurs **précédentes** du signal sur le même
actif (aucun regard sur la barre courante ni sur l'avenir). Puis, actif par actif, barre après
barre (hystérésis) :

- à plat : achat si ``z ≥ enter``, vente si ``z ≤ −enter`` ;
- acheteur : on reste tant que ``z > exit`` ; sinon vente si ``z ≤ −enter``, à plat sinon ;
- vendeur : symétrique (on reste tant que ``z < −exit``).

L'état ne change **que sur une barre tradable** : éligible et dont la barre suivante a une
ouverture (sinon l'entrée se ferait à un prix figé). Pendant les heures fermées (nuit d'un
indice, week-end, bascule exclue), la position est tenue telle quelle : pas de fermeture
forcée, donc pas de frais payés deux fois. Sortie : −1 / 0 / +1 par ligne du panel, à donner à
un portefeuille ``Book(persistent=True)`` (``timing/portfolio.py``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qlab.core.errors import DataError


def zscore(panel: pl.DataFrame, raw: pl.Series, window: int) -> pl.Series:
    """Signal divisé par l'écart-type de ses ``window`` valeurs précédentes (par actif)."""
    if window < 2:
        raise DataError("bandes : fenêtre d'au moins 2 barres")
    s = pl.col("_raw")
    scale = s.rolling_std(window, min_samples=window // 2).shift(1).over("symbol")
    frame = panel.select("symbol", raw.alias("_raw"))
    return frame.select(pl.when(scale > 0).then(s / scale).alias("z"))["z"]


def tradable(panel: pl.DataFrame) -> pl.Series:
    """Barre où l'on peut changer de position : éligible et ouverture de la barre suivante."""
    next_open = pl.col("open").shift(-1).over("symbol").is_not_null()
    return panel.select((pl.col("eligible") & next_open).alias("t"))["t"]


def hysteresis(
    symbols: pl.Series, z: pl.Series, can_trade: pl.Series, enter: float, exit_: float
) -> pl.Series:
    """États −1 / 0 / +1 (voir le module) ; panel trié par (actif, temps)."""
    if not 0 <= exit_ < enter:
        raise DataError(f"bandes : 0 ≤ sortie < entrée attendu (reçu {exit_}, {enter})")
    codes = symbols.cast(pl.Categorical).to_physical().to_numpy()
    values = z.fill_null(np.nan).to_numpy()
    ok = can_trade.fill_null(False).to_numpy()
    out = np.zeros(len(values))
    state, previous = 0.0, -1
    for i in range(len(values)):
        if codes[i] != previous:
            state, previous = 0.0, codes[i]
        v = values[i]
        if ok[i] and not np.isnan(v):
            if (state > 0 and v <= exit_) or (state < 0 and v >= -exit_):
                state = 0.0
            if state == 0:
                state = 1.0 if v >= enter else -1.0 if v <= -enter else 0.0
        out[i] = state
    return pl.Series("signal", out)


def band_signal(
    panel: pl.DataFrame, raw: pl.Series, enter: float, exit_: float, window: int
) -> pl.Series:
    """Signal brut → positions tenues −1 / 0 / +1."""
    return hysteresis(panel["symbol"], zscore(panel, raw, window), tradable(panel), enter, exit_)
