"""Tests for the Nihilist substitution cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.nihilist_substitution import NihilistSubstitution
from buttcrack.scoring import get_scorer


def test_vector_wikipedia():
    """Published worked example.

    Source: Wikipedia, "Nihilist cipher" -- square keyword ZEBRAS, additive key
    RUSSIAN, plaintext "DYNAMITE WINTER PALACE".
    PT coords 23 55 41 15 35 32 45 12 53 32 41 45 12 14 43 15 34 15 22 12
    KEY coords (RUSSIAN, repeating) 14 51 21 21 32 15 41 ...
    CT = 37 106 62 36 67 47 86 26 104 53 62 77 27 55 57 66 55 36 54 27
    """
    cipher = NihilistSubstitution()
    ct = cipher.encode("DYNAMITE WINTER PALACE", "ZEBRAS/RUSSIAN")
    expected = "37 106 62 36 67 47 86 26 104 53 62 77 27 55 57 66 55 36 54 27"
    assert ct == expected
    # Compare on the stripped digit string too, to be robust to spacing.
    assert ct.replace(" ", "") == expected.replace(" ", "")


def test_round_trip():
    cipher = NihilistSubstitution()
    key = "ZEBRAS/RUSSIAN"
    # The prepared plaintext merges J->I and drops non-letters.
    prepared = "DYNAMITEWINTERPALACE"
    ct = cipher.encode("DYNAMITE WINTER PALACE", key)
    assert cipher.decode(ct, key) == prepared


def test_round_trip_other_keys():
    cipher = NihilistSubstitution()
    for square_kw, add_kw in (("KEYWORD", "SECRET"), ("", "VODKA"), ("PALMER", "STONE")):
        key = f"{square_kw}/{add_kw}"
        prepared = cipher.decode(cipher.encode("ATTACKATDAWN", key), key)
        assert prepared == "ATTACKATDAWN".replace("J", "I")


@pytest.mark.slow
def test_crack_standard_square():
    """The crack assumes the standard A..Z (J->I) square; verify it recovers
    plaintext + period for such a ciphertext of reasonable length."""
    cipher = NihilistSubstitution()
    scorer = get_scorer()
    key = "/CIPHER"  # standard square, additive keyword CIPHER (period 6)
    plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENRUNSAWAYQUICKLYBEFORENIGHTFALLSUPONUSALL"
    ct = cipher.encode(plaintext, key)
    expected = cipher.decode(ct, key)  # prepared plaintext (J->I)
    results = cipher.crack(ct, scorer, top=3, rng=random.Random(1), timeout=30)
    assert results, "crack returned no candidates"
    assert results[0].plaintext == expected
    assert results[0].meta["period"] == 6
