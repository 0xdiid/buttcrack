"""Tests for the keyword / key-square finder (inverse of K1 keyed construction).

Keyed alphabets and squares are built here with the SAME forward construction
the project uses (``dedupe(keyword) + remaining alphabet`` for the alphabet, and
``PolybiusSquare`` for the grids), then the finder is asked to recover the
keyword. Recovery is considered correct when a returned keyword reconstructs the
identical alphabet/square — the construction is many-to-one, so the *exact*
original string need not come back, only an equivalent one.
"""

from __future__ import annotations

import pytest

from buttcrack.ciphers.squares import ALPHABET_5, ALPHABET_6, PolybiusSquare
from buttcrack.keyfinder import (
    find_keyword_in,
    keysquare_candidates,
    keyword_from_alphabet,
)

_FULL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _build_keyed(keyword: str, alphabet: str) -> str:
    """Forward K1 construction, identical to the production builders."""
    seq: list[str] = []
    for ch in keyword.upper() + alphabet:
        if ch in alphabet and ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _dedupe(keyword: str, alphabet: str) -> str:
    """The deduped, alphabet-filtered keyword (its leading portion in ``keyed``)."""
    seq: list[str] = []
    for ch in keyword.upper():
        if ch in alphabet and ch not in seq:
            seq.append(ch)
    return "".join(seq)


# --- 26-letter keyed alphabet ------------------------------------------------

ALPHABET_KEYWORDS = [
    "CIPHER",
    "KEYWORD",
    "ZEBRA",
    "PLAYFAIR",
    "SUBSTITUTION",
    "CRYPTOGRAM",
    "MACHINE",
    "QWERTY",
]


@pytest.mark.parametrize("keyword", ALPHABET_KEYWORDS)
def test_keyword_from_alphabet_recovers(keyword: str):
    keyed = _build_keyed(keyword, _FULL)
    assert len(keyed) == 26 and len(set(keyed)) == 26

    candidates = keyword_from_alphabet(keyed)
    assert candidates, f"no keyword recovered for {keyword!r}"

    # The genuine (shortest) keyword is a prefix of the deduped original; the
    # deduped original itself reconstructs the identical alphabet and is a valid
    # candidate (it may share the alphabet with a shorter equivalent keyword).
    deduped = _dedupe(keyword, _FULL)
    assert deduped in candidates
    assert deduped.startswith(candidates[0])

    # Every candidate reconstructs the identical alphabet.
    for cand in candidates:
        assert _build_keyed(cand, _FULL) == keyed

    # Candidates come back shortest first.
    lengths = [len(c) for c in candidates]
    assert lengths == sorted(lengths)


def test_straight_alphabet_returns_empty():
    assert keyword_from_alphabet(_FULL) == []


def test_alphabet_accepts_lowercase_and_punctuation():
    keyed = _build_keyed("CIPHER", _FULL)
    spaced = " ".join(keyed.lower())  # lowercase, spaces between letters
    assert keyword_from_alphabet(spaced) == keyword_from_alphabet(keyed)


def test_alphabet_rejects_wrong_length():
    with pytest.raises(ValueError):
        keyword_from_alphabet("ABCDEF")


def test_alphabet_rejects_non_permutation():
    with pytest.raises(ValueError):
        keyword_from_alphabet("A" * 26)


# --- 5x5 and 6x6 Polybius squares -------------------------------------------

SQUARE_KEYWORDS = ["CIPHER", "PLAYFAIR", "ZEBRAS", "KEYWORD", "MONARCHY"]


@pytest.mark.parametrize("keyword", SQUARE_KEYWORDS)
def test_keysquare_5x5_recovers(keyword: str):
    sq = PolybiusSquare(keyword, size=5)
    square = "".join(sq.grid)
    assert len(square) == 25

    candidates = keysquare_candidates(square, size=5)
    assert candidates, f"no keyword recovered for 5x5 {keyword!r}"

    # The deduped keyword (J->I, alphabet-filtered) is recovered, and the
    # shortest candidate is a prefix of it.
    deduped = _dedupe(keyword.replace("J", "I"), ALPHABET_5)
    assert deduped in candidates
    assert deduped.startswith(candidates[0])

    for cand in candidates:
        rebuilt = "".join(PolybiusSquare(cand, size=5).grid)
        assert rebuilt == square


@pytest.mark.parametrize("keyword", SQUARE_KEYWORDS)
def test_keysquare_6x6_recovers(keyword: str):
    sq = PolybiusSquare(keyword, size=6)
    square = "".join(sq.grid)
    assert len(square) == 36

    candidates = keysquare_candidates(square, size=6)
    assert candidates, f"no keyword recovered for 6x6 {keyword!r}"

    deduped = _dedupe(keyword, ALPHABET_6)
    assert deduped in candidates
    assert deduped.startswith(candidates[0])

    for cand in candidates:
        rebuilt = "".join(PolybiusSquare(cand, size=6).grid)
        assert rebuilt == square


def test_keysquare_6x6_with_digit_keyword():
    # 6x6 alphabet includes 0-9; a keyword using digits round-trips too.
    sq = PolybiusSquare("AGENT47", size=6)
    square = "".join(sq.grid)
    candidates = keysquare_candidates(square, size=6)
    assert candidates
    for cand in candidates:
        assert "".join(PolybiusSquare(cand, size=6).grid) == square


def test_straight_5x5_square_returns_empty():
    assert keysquare_candidates(ALPHABET_5, size=5) == []


def test_straight_6x6_square_returns_empty():
    assert keysquare_candidates(ALPHABET_6, size=6) == []


def test_keysquare_rejects_bad_size():
    with pytest.raises(ValueError):
        keysquare_candidates(ALPHABET_5, size=7)


def test_keysquare_rejects_wrong_length():
    with pytest.raises(ValueError):
        keysquare_candidates("ABCDE", size=5)


# --- find_keyword_in dispatcher ---------------------------------------------


def test_find_keyword_in_routes_by_length():
    alpha = _build_keyed("CIPHER", _FULL)
    assert find_keyword_in(alpha) == keyword_from_alphabet(alpha)

    sq5 = "".join(PolybiusSquare("ZEBRAS", size=5).grid)
    assert find_keyword_in(sq5) == keysquare_candidates(sq5, size=5)

    sq6 = "".join(PolybiusSquare("KEYWORD", size=6).grid)
    assert find_keyword_in(sq6) == keysquare_candidates(sq6, size=6)


def test_find_keyword_in_size_override():
    sq5 = "".join(PolybiusSquare("CIPHER", size=5).grid)
    assert find_keyword_in(sq5, size=5) == keysquare_candidates(sq5, size=5)


def test_find_keyword_in_rejects_unknown_shape():
    with pytest.raises(ValueError):
        find_keyword_in("ABCDEFG")
