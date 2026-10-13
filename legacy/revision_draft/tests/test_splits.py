import pytest

from revision.make_splits import make_folds, verify_folds


def test_stratified_folds_cover_pool_once_and_keep_test_out():
    pool = [f"img_{i:03d}.jpg" for i in range(50)]
    strata = [1] * 40 + [0] * 10
    folds = make_folds(pool, strata, n_folds=5, seed=0)
    verify_folds(folds, set(pool), {"test_001.jpg"})
    assert [len(f["val"]) for f in folds] == [10] * 5
    negatives_per_val = [sum(1 for n in f["val"] if strata[pool.index(n)] == 0) for f in folds]
    assert negatives_per_val == [2] * 5


def test_folds_are_deterministic_per_seed():
    pool = [f"img_{i:03d}.jpg" for i in range(30)]
    strata = [1] * 24 + [0] * 6
    assert make_folds(pool, strata, 5, 1) == make_folds(pool, strata, 5, 1)
    assert make_folds(pool, strata, 5, 1) != make_folds(pool, strata, 5, 2)


def test_verify_detects_leak():
    pool = [f"img_{i:03d}.jpg" for i in range(20)]
    folds = make_folds(pool, [1] * 16 + [0] * 4, 4, 0)
    with pytest.raises(AssertionError):
        verify_folds(folds, set(pool), {pool[0]})
