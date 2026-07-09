"""Tests for the Redefence cipher (rail fence with permuted read-order + offset)."""

from __future__ import annotations

import pytest

from buttcrack.ciphers.redefence import Redefence
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def test_vector_published():
    """Published Black Chamber vector: 3 rails, offset 0, read order rail2,rail3,rail1."""
    cipher = Redefence()
    # key "3:0:3,1,2" => rail1 read 3rd, rail2 1st, rail3 2nd.
    out = cipher.encode("THIS IS A TEST", "3:0:3,1,2")
    assert out == "HSSTSIATTIE"  # spec ciphertext "HSSTS IATTI E", letters only


def test_roundtrip():
    cipher = Redefence()
    msg = "The quick brown fox jumps over the lazy dog near the river bank tonight"
    expected = only_letters(msg).upper()
    for key in ["4:2:3,1,4,2", "3:0:3,1,2", "5:7:2,4,1,5,3", "6:0:1,2,3,4,5,6"]:
        assert cipher.decode(cipher.encode(msg, key), key) == expected


def test_bare_permutation_key_is_plain_railfence_offset0():
    cipher = Redefence()
    msg = "ATTACK AT DAWN FROM THE NORTH"
    # bare permutation assumes offset 0; identity order == plain rail fence.
    assert cipher.decode(cipher.encode(msg, "1,2,3"), "1,2,3") == only_letters(msg).upper()


@pytest.mark.slow
def test_crack_recovers_plaintext():
    cipher = Redefence()
    scorer = get_scorer()
    plain = (
        "the project gutenberg ebook of the adventures of sherlock holmes by "
        "arthur conan doyle this ebook is for the use of anyone anywhere in the "
        "united states and most other parts of the world at no cost"
    )
    key = "5:3:4,1,5,2,3"
    ciphertext = cipher.encode(plain, key)
    candidates = cipher.crack(ciphertext, scorer, top=5, timeout=30)
    assert candidates, "crack returned no candidates"
    assert candidates[0].plaintext == only_letters(plain).upper()
