"""Bootstrap stationnaire (Politis & Romano 1994) : intervalles de confiance qui respectent
l'autocorrélation et les grappes de volatilité des rendements.

Un bootstrap classique tire les jours un par un, comme s'ils étaient indépendants : il détruit
les tendances et les périodes agitées, et donne des intervalles trop étroits. Ici on tire des
**blocs** de jours consécutifs, de longueur aléatoire (loi géométrique, moyenne
``mean_block``), en bouclant en fin de série : la série rééchantillonnée reste stationnaire.

- Plusieurs séries (ex. stratégie et buy & hold) sont rééchantillonnées **ensemble**, aux mêmes
  dates : on compare toujours des périodes identiques.
- Déterministe : graine explicite (``seed``), même entrée ⇒ mêmes intervalles.
- Intervalle de confiance par percentiles des statistiques rééchantillonnées.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlab.core.errors import DataError
from qlab.research.stats import moments

Floats = npt.NDArray[np.float64]
Statistic = Callable[..., float]


def stationary_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """``n`` indices : blocs consécutifs (bouclés) de longueurs géométriques de moyenne
    ``mean_block`` ; chaque bloc démarre à une position uniforme."""
    if n < 1:
        raise DataError("bootstrap : série vide")
    if mean_block < 1:
        raise DataError("bootstrap : mean_block doit être ≥ 1")
    new_block = rng.random(n) < 1 / mean_block
    new_block[0] = True
    block_id = np.cumsum(new_block) - 1
    positions = np.arange(n)
    block_start_pos = positions[new_block]
    starts = rng.integers(0, n, size=int(new_block.sum()))
    offset = positions - block_start_pos[block_id]
    return ((starts[block_id] + offset) % n).astype(np.int64)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """``estimate`` : statistique sur les données réelles ; ``samples`` : sur chaque tirage."""

    estimate: float
    samples: Floats

    def interval(self, level: float) -> tuple[float, float]:
        """Intervalle de confiance bilatéral à ``level`` (ex. 0,95), par percentiles."""
        if not 0 < level < 1:
            raise DataError("niveau de confiance dans ]0, 1[ attendu")
        tail = (1 - level) / 2 * 100
        low, high = np.percentile(self.samples, [tail, 100 - tail])
        return float(low), float(high)

    def strictly_positive(self, level: float) -> bool:
        """Critère d'acceptation : borne basse de l'intervalle strictement > 0."""
        return self.interval(level)[0] > 0


def bootstrap(
    statistic: Statistic, *series: npt.ArrayLike, n_boot: int, mean_block: float, seed: int
) -> BootstrapResult:
    """Applique ``statistic`` aux séries réelles puis à ``n_boot`` rééchantillonnages communs.

    Les séries doivent avoir la même longueur (mêmes dates) et des valeurs finies.
    """
    if n_boot < 1:
        raise DataError("bootstrap : n_boot doit être ≥ 1")
    arrays = [np.asarray(s, dtype=np.float64) for s in series]
    if not arrays:
        raise DataError("bootstrap : au moins une série")
    n = arrays[0].size
    if any(a.ndim != 1 or a.size != n for a in arrays):
        raise DataError("bootstrap : séries 1D de même longueur (mêmes dates) attendues")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise DataError("bootstrap : valeurs non finies (trous à exclure avant)")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = stationary_indices(n, mean_block, rng)
        samples[b] = statistic(*(a[idx] for a in arrays))
    return BootstrapResult(float(statistic(*arrays)), samples)


def sharpe(returns: Floats) -> float:
    """Sharpe par période (statistique usuelle à bootstrapper)."""
    return moments(returns).sharpe


def sharpe_difference(strategy: Floats, benchmark: Floats) -> float:
    """Sharpe de la stratégie − Sharpe de la référence, sur les mêmes dates."""
    return moments(strategy).sharpe - moments(benchmark).sharpe
