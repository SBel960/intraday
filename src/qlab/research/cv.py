"""Validation croisée sans fuite : K plis purgés avec embargo, et walk-forward.

Chaque observation i porte l'intervalle d'information dont dépend son résultat,
``[starts[i], ends[i]]`` (ex. décision en t, rendement mesuré jusqu'à t + horizon). Unités
libres mais identiques (index de bougie, ms…), ``starts`` triés.

- **Purge** : une observation d'entraînement dont l'intervalle chevauche celui du bloc de test
  est retirée (elle « connaît » une partie du test).
- **Embargo** : on retire aussi celles qui commencent moins de ``embargo`` après la fin du test
  (autocorrélation). SPEC_LONG_TERME LT.6 : embargo ≥ lookback maximal + horizon de détention.
- **Walk-forward** : on n'entraîne que sur le passé (purgé), test sur le bloc suivant ; fenêtre
  expansive (tout le passé) ou glissante (les ``train_size`` dernières observations).

``leaks`` vérifie un découpage : utilisable par tout backtest comme garde-fou.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlab.core.errors import DataError

Ints = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Split:
    """Indices d'entraînement et de test (triés)."""

    train: Ints
    test: Ints


def _intervals(starts: npt.ArrayLike, ends: npt.ArrayLike) -> tuple[Ints, Ints]:
    s, e = np.asarray(starts), np.asarray(ends)
    if s.ndim != 1 or s.shape != e.shape or s.size == 0:
        raise DataError("starts et ends : séries 1D non vides de même longueur")
    if not (np.issubdtype(s.dtype, np.integer) and np.issubdtype(e.dtype, np.integer)):
        raise DataError("starts et ends : entiers attendus (index ou ms)")
    if np.any(e < s):
        raise DataError("chaque intervalle doit vérifier start ≤ end")
    if np.any(np.diff(s) < 0):
        raise DataError("starts doit être trié par ordre croissant")
    return s.astype(np.int64), e.astype(np.int64)


def _purged_train(candidates: Ints, s: Ints, e: Ints, test: Ints, embargo: int) -> Ints:
    """Candidats sans chevauchement avec le test ni départ dans l'embargo qui le suit."""
    test_start, test_end = int(s[test].min()), int(e[test].max())
    keep = (e[candidates] < test_start) | (s[candidates] > test_end + embargo)
    return candidates[keep]


def purged_kfold(
    starts: npt.ArrayLike, ends: npt.ArrayLike, *, n_splits: int, embargo: int
) -> list[Split]:
    """K plis contigus ; l'entraînement de chaque pli est purgé et embargoé."""
    s, e = _intervals(starts, ends)
    if not 2 <= n_splits <= s.size:
        raise DataError(f"n_splits doit être dans [2, {s.size}]")
    if embargo < 0:
        raise DataError("embargo doit être ≥ 0")
    all_idx = np.arange(s.size, dtype=np.int64)
    splits = []
    for test in np.array_split(all_idx, n_splits):
        others = np.setdiff1d(all_idx, test)
        splits.append(Split(_purged_train(others, s, e, test, embargo), test))
    return splits


def walk_forward(
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    *,
    test_size: int,
    min_train: int,
    train_size: int | None = None,
) -> list[Split]:
    """Blocs de test successifs de ``test_size`` observations ; entraînement = passé purgé
    (tout, ou les ``train_size`` derniers) ; le premier bloc attend ``min_train`` observations
    d'entraînement. Aucune donnée postérieure au début du test n'est jamais utilisée."""
    s, e = _intervals(starts, ends)
    if test_size < 1 or min_train < 1 or (train_size is not None and train_size < min_train):
        raise DataError("test_size ≥ 1, min_train ≥ 1 et train_size ≥ min_train attendus")
    splits, first = [], 0
    while first < s.size:
        train = np.nonzero(e < s[first])[0].astype(np.int64)  # passé entièrement connu
        if train.size >= min_train:
            if train_size is not None:
                train = train[-train_size:]
            test = np.arange(first, min(first + test_size, s.size), dtype=np.int64)
            splits.append(Split(train, test))
            first += test_size
        else:
            first += 1
    return splits


def leaks(split: Split, starts: npt.ArrayLike, ends: npt.ArrayLike, embargo: int) -> list[int]:
    """Indices d'entraînement qui chevauchent le test ou tombent dans son embargo (vide = sain)."""
    s, e = _intervals(starts, ends)
    if split.test.size == 0:
        return []
    clean = set(_purged_train(split.train, s, e, split.test, embargo).tolist())
    return [int(i) for i in split.train if int(i) not in clean]
