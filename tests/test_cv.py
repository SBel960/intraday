"""Tests de qlab.research.cv : découpages calculés à la main, absence de fuite, erreurs.

Cas de base : 10 observations, intervalle [i, i + 2] (résultat mesuré sur 2 pas), embargo 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.core.errors import DataError
from qlab.research.cv import Split, leaks, purged_kfold, walk_forward

S = np.arange(10)
E = S + 2


def _lists(splits: list[Split]) -> list[tuple[list[int], list[int]]]:
    return [(sp.test.tolist(), sp.train.tolist()) for sp in splits]


def test_purged_kfold_by_hand() -> None:
    """Test [2, 3] couvre 2 → 5 : purge des observations dont la fin ≥ 2 et le début ≤ 5,
    embargo jusqu'à 6 ⇒ entraînement [7, 8, 9]."""
    assert _lists(purged_kfold(S, E, n_splits=5, embargo=1)) == [
        ([0, 1], [5, 6, 7, 8, 9]),
        ([2, 3], [7, 8, 9]),
        ([4, 5], [0, 1, 9]),
        ([6, 7], [0, 1, 2, 3]),
        ([8, 9], [0, 1, 2, 3, 4, 5]),
    ]


def test_without_label_overlap_or_embargo_nothing_is_purged() -> None:
    splits = purged_kfold(S, S, n_splits=5, embargo=0)  # intervalles ponctuels
    assert all(len(sp.train) == 8 for sp in splits)


def test_walk_forward_expanding_by_hand() -> None:
    """Passé entièrement connu avant t : ends < t. Il faut 4 observations ⇒ premier test en 6."""
    assert _lists(walk_forward(S, E, test_size=2, min_train=4)) == [
        ([6, 7], [0, 1, 2, 3]),
        ([8, 9], [0, 1, 2, 3, 4, 5]),
    ]


def test_walk_forward_rolling_by_hand() -> None:
    assert _lists(walk_forward(S, E, test_size=2, min_train=3, train_size=3)) == [
        ([5, 6], [0, 1, 2]),
        ([7, 8], [2, 3, 4]),
        ([9], [4, 5, 6]),
    ]


def test_random_intervals_never_leak() -> None:
    """500 jeux aléatoires (horizons variables, trous) : aucun découpage ne fuit ; les plis de
    test couvrent chaque observation exactement une fois ; le walk-forward ne regarde jamais
    l'avenir."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(20, 120))
        s = np.sort(rng.integers(0, 1000, n))
        e = s + rng.integers(0, 40, n)
        emb = int(rng.integers(0, 30))
        folds = purged_kfold(s, e, n_splits=int(rng.integers(2, 8)), embargo=emb)
        assert sorted(np.concatenate([f.test for f in folds]).tolist()) == list(range(n))
        for f in folds:
            assert leaks(f, s, e, emb) == []
            assert not set(f.train.tolist()) & set(f.test.tolist())
        for f in walk_forward(s, e, test_size=5, min_train=5):
            assert int(e[f.train].max()) < int(s[f.test].min())


def test_leaks_detects_a_contaminated_split() -> None:
    bad = Split(train=np.array([1, 3, 8]), test=np.array([4, 5]))  # 3 chevauche, 8 dans l'embargo
    assert leaks(bad, S, E, embargo=2) == [3, 8]
    assert leaks(Split(np.array([0]), np.array([], dtype=np.int64)), S, E, 0) == []


@pytest.mark.parametrize(
    ("call", "msg"),
    [
        (lambda: purged_kfold(S, E, n_splits=1, embargo=0), "n_splits"),
        (lambda: purged_kfold(S, E, n_splits=11, embargo=0), "n_splits"),
        (lambda: purged_kfold(S, E, n_splits=2, embargo=-1), "embargo"),
        (lambda: purged_kfold(S, S - 1, n_splits=2, embargo=0), "start ≤ end"),
        (lambda: purged_kfold(S[::-1], E[::-1], n_splits=2, embargo=0), "trié"),
        (lambda: purged_kfold(S.astype(float), E, n_splits=2, embargo=0), "entiers"),
        (lambda: purged_kfold([], [], n_splits=2, embargo=0), "non vides"),
        (lambda: walk_forward(S, E, test_size=0, min_train=1), "test_size"),
        (lambda: walk_forward(S, E, test_size=1, min_train=3, train_size=2), "train_size"),
    ],
)
def test_errors(call: object, msg: str) -> None:
    with pytest.raises(DataError, match=msg):
        call()  # type: ignore[operator]


def test_not_enough_history_gives_no_split() -> None:
    assert walk_forward(S, E, test_size=2, min_train=50) == []
