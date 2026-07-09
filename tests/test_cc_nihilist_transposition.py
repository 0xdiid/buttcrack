"""Tests for the Nihilist Transposition cipher."""

from __future__ import annotations

import pytest

from buttcrack.ciphers.nihilist_transposition import NihilistTransposition
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def test_published_vector():
    """ACA 'The ACA and You' vector (Elcy ch. IV p.18), take-off by columns."""
    cipher = NihilistTransposition()
    plaintext = "SQUARENEEDEDHERE"
    key = "2134"
    # "EQDER SEHNU EREAD E" with spaces removed.
    expected = "EQDERSEHNUEREADE"
    assert cipher.encode(plaintext, key) == expected


def test_round_trip():
    cipher = NihilistTransposition()
    key = "315264"  # n=6 grid (36 cells)
    msg = "Attack at dawn, the enemy sleeps now!"  # 29 letters, padded to 36
    encoded = cipher.encode(msg, key)
    decoded = cipher.decode(encoded, key)
    # Decode returns the padded square; the original letters are a prefix.
    plain = only_letters(msg).upper()
    assert decoded.startswith(plain)
    assert len(decoded) == 36


def test_round_trip_exact_square():
    cipher = NihilistTransposition()
    key = "2134"
    msg = "SQUARENEEDEDHERE"  # exactly 16 letters -> 4x4, no padding
    encoded = cipher.encode(msg, key)
    decoded = cipher.decode(encoded, key)
    assert decoded == only_letters(msg).upper()


@pytest.mark.slow
def test_crack_recovers_plaintext():
    cipher = NihilistTransposition()
    scorer = get_scorer()
    plaintext = (
        "WE HOLD THESE TRUTHS TO BE SELF EVIDENT THAT ALL MEN ARE CREATED "
        "EQUAL THAT THEY ARE ENDOWED"
    )  # 75 letters -> n=9 grid (81 cells)
    key = "417253869"
    ciphertext = cipher.encode(plaintext, key)
    candidates = cipher.crack(ciphertext, scorer, top=5)
    assert candidates, "crack returned no candidates"
    expected = only_letters(plaintext).upper()
    best_letters = only_letters(candidates[0].plaintext).upper()
    # The recovered plaintext (sans trailing X padding) should match.
    assert best_letters.rstrip("X").startswith(expected) or expected.startswith(
        best_letters.rstrip("X")
    )
