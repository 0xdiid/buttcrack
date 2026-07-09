"""Polybius square foundation."""

import pytest

from buttcrack.ciphers.squares import PolybiusSquare


def test_zebras_square_rowwise():
    sq = PolybiusSquare("ZEBRAS")
    assert "".join(sq.grid) == "ZEBRASCDFGHIKLMNOPQTUVWXY"
    # Wikipedia Nihilist example: D is at row 2, col 3 (1-indexed).
    assert sq.rc("D") == (1, 2)
    assert sq.at(1, 2) == "D"


def test_ji_merge():
    sq = PolybiusSquare("KEYWORD")
    assert sq.rc("J") == sq.rc("I")  # J folds onto I
    assert "J" not in sq.grid
    assert len(sq.grid) == 25


def test_coordinate_round_trip():
    sq = PolybiusSquare("PLAYFAIR")
    for ch in sq.grid:
        r, c = sq.rc(ch)
        assert sq.at(r, c) == ch


def test_six_by_six_keeps_digits_and_j():
    sq = PolybiusSquare("SECRET", size=6)
    assert len(sq.grid) == 36
    assert "J" in sq.grid and "5" in sq.grid
    assert sq.rc("J") != sq.rc("I")  # no merge in 6x6


def test_bad_size_raises():
    with pytest.raises(ValueError):
        PolybiusSquare("KEY", size=5, alphabet="ABC")  # too few cells
