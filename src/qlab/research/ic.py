"""Information Coefficient : corrélation de rang (Spearman) entre un signal et le rendement futur.

- **IC temporel** d'un actif, par horizon ``h`` : ρ(signal_t, r_{t→t+h}) ; la **décroissance**
  de l'IC avec ``h`` dit combien de temps le signal garde de l'information (à comparer à
  l'horizon minimal du cost gate).
- **IC transversal** : à chaque date, on classe les actifs entre eux (signal contre rendement
  futur) ; la moyenne de ces IC dans le temps, avec un t-stat de Newey–West, mesure si le
  signal trie les gagnants des perdants. Des centaines d'actifs ⇒ bien plus de puissance
  qu'un seul actif.

Conventions : séries alignées sur la date de décision ``t`` (le signal n'utilise que
l'information ≤ t ; ``forward_returns`` fournit r_{t→t+h}). ``NaN`` = absent (actif pas encore
coté ou retiré, fin de série) : la paire est ignorée, jamais comblée. Aucune hypothèse de
marché : les horizons sont en nombre de périodes, fournis par l'appelant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from qlab.core.errors import DataError
from qlab.research.stats import newey_west_lags, newey_west_tstat

Floats = npt.NDArray[np.float64]


def _as_floats(x: npt.ArrayLike, ndim: int, name: str) -> Floats:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim:
        raise DataError(f"{name} : tableau à {ndim} dimension(s) attendu")
    if np.any(np.isinf(a)):
        raise DataError(f"{name} : valeurs infinies (NaN seul marque une absence)")
    return a


def spearman(signal: npt.ArrayLike, forward: npt.ArrayLike, *, min_obs: int = 3) -> float:
    """ρ de Spearman sur les paires où les deux valeurs sont présentes (rangs moyens en cas
    d'égalité). ``NaN`` si moins de ``min_obs`` paires ou si un côté est constant."""
    x, y = _as_floats(signal, 1, "signal"), _as_floats(forward, 1, "rendements futurs")
    if x.shape != y.shape:
        raise DataError("signal et rendements futurs : même longueur attendue")
    both = np.isfinite(x) & np.isfinite(y)
    x, y = x[both], y[both]
    if x.size < max(min_obs, 2) or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def forward_returns(log_prices: npt.ArrayLike, horizon: int) -> Floats:
    """r_{t→t+h} = ln P_{t+h} − ln P_t le long du premier axe (dates) ; ``NaN`` à la fin."""
    p = np.asarray(log_prices, dtype=np.float64)
    if p.ndim not in (1, 2) or p.shape[0] == 0:
        raise DataError("log-prix : tableau 1D ou 2D (dates × actifs) non vide attendu")
    if horizon < 1:
        raise DataError("horizon ≥ 1 période attendu")
    out = np.full_like(p, np.nan)
    if horizon < p.shape[0]:
        out[:-horizon] = p[horizon:] - p[:-horizon]
    return out


@dataclass(frozen=True, slots=True)
class DecayPoint:
    """IC temporel à un horizon (en périodes) et nombre de paires utilisées."""

    horizon: int
    ic: float
    n: int


def ic_decay(
    signal: npt.ArrayLike, log_prices: npt.ArrayLike, horizons: Sequence[int]
) -> list[DecayPoint]:
    """IC temporel d'un actif pour chaque horizon (courbe de décroissance)."""
    s = _as_floats(signal, 1, "signal")
    p = _as_floats(log_prices, 1, "log-prix")
    if s.shape != p.shape:
        raise DataError("signal et log-prix : même longueur attendue")
    if not horizons or len(set(horizons)) != len(horizons):
        raise DataError("horizons : liste non vide sans doublon attendue")
    points = []
    for h in sorted(horizons):
        fwd = forward_returns(p, h)
        n = int(np.count_nonzero(np.isfinite(s) & np.isfinite(fwd)))
        points.append(DecayPoint(h, spearman(s, fwd), n))
    return points


@dataclass(frozen=True, slots=True)
class CrossSectionalIC:
    """IC transversal : ``per_date`` (NaN si date inutilisable), moyenne, t-stat Newey–West
    (``lags`` retards), ``icir`` = moyenne / écart-type, part des dates où IC > 0."""

    per_date: Floats
    n_dates: int
    mean: float
    tstat: float
    lags: int
    icir: float
    hit_rate: float


def cross_sectional_ic(
    signal: npt.ArrayLike,
    forward: npt.ArrayLike,
    *,
    horizon: int,
    min_assets: int,
    lags: int | None = None,
) -> CrossSectionalIC:
    """IC transversal sur des matrices dates × actifs.

    Une date compte si au moins ``min_assets`` actifs ont signal et rendement présents.
    Des rendements futurs sur ``horizon`` périodes échantillonnés à chaque période se
    chevauchent : les IC successifs sont corrélés sur ``horizon − 1`` retards, d'où
    ``lags`` ≥ ``horizon − 1`` (par défaut : max de cette borne et de la règle de Newey–West).
    """
    s = _as_floats(signal, 2, "signal")
    f = _as_floats(forward, 2, "rendements futurs")
    if s.shape != f.shape:
        raise DataError("signal et rendements futurs : mêmes dimensions (dates × actifs)")
    if horizon < 1 or min_assets < 3:
        raise DataError("horizon ≥ 1 et min_assets ≥ 3 attendus")
    per_date = np.array([spearman(a, b, min_obs=min_assets) for a, b in zip(s, f, strict=True)])
    valid = per_date[np.isfinite(per_date)]
    need = max(horizon - 1, newey_west_lags(valid.size)) if lags is None else lags
    if lags is not None and lags < horizon - 1:
        raise DataError(f"lags ≥ horizon − 1 = {horizon - 1} (rendements qui se chevauchent)")
    if valid.size < need + 2:
        raise DataError(f"IC transversal : {valid.size} dates utilisables, {need + 2} requises")
    std = float(np.std(valid, ddof=1))
    return CrossSectionalIC(
        per_date=per_date,
        n_dates=int(valid.size),
        mean=float(valid.mean()),
        tstat=newey_west_tstat(valid, need),
        lags=need,
        icir=float(valid.mean()) / std if std > 0 else float("nan"),
        hit_rate=float(np.mean(valid > 0)),
    )
