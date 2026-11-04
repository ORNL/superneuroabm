"""
_unique_rows must give exactly np.unique(mat, axis=0, return_inverse=True): the same distinct
rows in the same order and the same inverse, for few or many distinct rows, any number of
columns (including one whose combined key overflows int64), and the empty edge cases.
"""
import numpy as np
import pytest

from superneuroabm.model import _unique_rows


def _check(mat):
    ref_rows, ref_inv = np.unique(mat, axis=0, return_inverse=True)
    rows, inv = _unique_rows(mat)
    assert rows.dtype == mat.dtype
    np.testing.assert_array_equal(rows, ref_rows)
    np.testing.assert_array_equal(inv, ref_inv.ravel())


@pytest.mark.parametrize("seed", range(20))
def test_matches_numpy_with_duplicates(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 2000)), int(rng.integers(1, 6))
    levels = rng.normal(size=(int(rng.integers(1, 8)), k))     # few distinct values per column
    _check(levels[rng.integers(0, len(levels), size=n)] * rng.integers(1, 3, size=k))


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.int32])
def test_matches_numpy_all_distinct(dtype):
    rng = np.random.default_rng(1)
    _check((rng.normal(size=(5000, 3)) * 1000).astype(dtype))


def test_matches_numpy_when_the_key_overflows():
    # 5 columns of 20,000 distinct values: 20,000^5 > 2^63, so the key is re-compressed
    rng = np.random.default_rng(2)
    mat = rng.random((20_000, 5))
    mat[1::7] = mat[0::7][: len(mat[1::7])]                     # some repeated rows too
    _check(mat)


def test_matches_numpy_when_rows_become_distinct_after_a_recompression():
    # first column has 10 values, the rest are distinct: the key overflows at the sixth
    # column, is re-compressed, and every row is already distinct there
    rng = np.random.default_rng(3)
    mat = rng.random((20_000, 6))
    mat[:, 0] = rng.integers(0, 10, size=len(mat))
    _check(mat)


def test_empty_edge_cases():
    for shape in [(0, 2), (3, 0), (0, 0), (1, 1)]:
        mat = np.zeros(shape)
        rows, inv = _unique_rows(mat)
        if shape[1]:
            _check(mat)
        else:
            assert rows.shape == ((1 if shape[0] else 0), 0) and inv.shape == (shape[0],)
