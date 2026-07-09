"""Maximum-weight (Hungarian) assignment."""

import pytest

from buttcrack.assignment import hungarian_max


def test_square_optimum():
    total, assign = hungarian_max([[1, 2, 3], [6, 5, 4], [1, 1, 1]])
    assert assign == [2, 0, 1] and total == 10


def test_rectangular_more_columns_than_rows():
    total, assign = hungarian_max([[10, 1, 1], [1, 1, 10]])
    assert assign == [0, 2] and total == 20


def test_beats_greedy_when_two_rows_want_one_column():
    # greedy would take column 0 for both rows (9, 8); the optimum splits them.
    total, assign = hungarian_max([[9, 1], [8, 7]])
    assert total == 16 and assign == [0, 1]  # 9 + 7 beats 1 + 8


def test_more_rows_than_columns_raises():
    with pytest.raises(ValueError):
        hungarian_max([[1, 2], [3, 4], [5, 6]])
