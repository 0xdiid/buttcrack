"""Tests for the Morbit cipher (ACA digraph fractionation of Morse).

Vector source: dCode "Morbit Cipher" (https://www.dcode.fr/morbit-cipher),
verified by recomputation in docs/cipher-specs.json: plaintext "MORE BITS" with
keyword "MORSECODE" (pair columns -> digits 5 6 8 9 3 1 7 2 4) encrypts to
"32379749578158".
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.morbit import Morbit
from buttcrack.scoring import get_scorer


def test_vector_dcode_more_bits():
    # dCode published vector, recomputed exactly.
    cipher = Morbit()
    assert cipher.encode("MORE BITS", "MORSECODE") == "32379749578158"


def test_vector_digit_permutation_key():
    # The keyword MORSECODE is equivalent to the direct digit key 5 6 8 9 3 1 7 2 4.
    cipher = Morbit()
    assert cipher.encode("MORE BITS", "568931724") == "32379749578158"


def test_decode_vector():
    cipher = Morbit()
    assert cipher.decode("32379749578158", "MORSECODE") == "MORE BITS"


def test_round_trip():
    cipher = Morbit()
    key = "MORSECODE"
    msg = "DEFEND THE EAST WALL OF THE CASTLE"
    prepared = "DEFEND THE EAST WALL OF THE CASTLE"
    assert cipher.decode(cipher.encode(msg, key), key) == prepared


def test_round_trip_grouped_ciphertext_ignores_spaces():
    # Decoding should tolerate the conventional 5-digit grouping (spaces).
    cipher = Morbit()
    key = "MORSECODE"
    ct = cipher.encode("WE ARE DISCOVERED FLEE AT ONCE", key)
    grouped = " ".join(ct[i : i + 5] for i in range(0, len(ct), 5))
    assert cipher.decode(grouped, key) == "WE ARE DISCOVERED FLEE AT ONCE"


@pytest.mark.slow
def test_crack_recovers_plaintext():
    cipher = Morbit()
    scorer = get_scorer()
    plaintext = "MEET ME AT THE OLD MILL HOUSE TONIGHT AT MIDNIGHT"
    ciphertext = cipher.encode(plaintext, "MORSECODE")
    candidates = cipher.crack(ciphertext, scorer, top=5, rng=random.Random(7), timeout=60)
    assert candidates
    assert candidates[0].plaintext.replace(" ", "") == plaintext.replace(" ", "")
