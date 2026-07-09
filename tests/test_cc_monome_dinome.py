"""Tests for the Monome-Dinome cipher.

Vector source: dCode "Monome-Dinome Cipher" (https://www.dcode.fr/monome-dinome-cipher).
Default grid (row prefixes 3 and 7), with the top (monome) row skipping the two
prefix columns and the dinome rows filled row-major across all ten columns::

        0  1  2  3  4  5  6  7  8  9
        A  B  C     D  E  F     G  H      (monome: A=0 B=1 C=2 D=4 E=5 F=6 G=8 H=9)
    3   I  J  K  L  M  N  O  P  Q  R      (dinome 30..39)
    7   S  T  U  V  W  X  Y  Z            (dinome 70..77)

dCode's worked example: MONOME -> 34,36,35,36,34,5 ; the digits 4,30,35,36,34,5
decode to DINOME. The published key for this board is row prefixes "37" with a
straight A-H top row and A-Z fill, i.e. "37/ABCDEFGH/IJKLMNOPQRSTUVWXYZ".
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.monome_dinome import MonomeDinome
from buttcrack.scoring import get_scorer

# The dCode default board, expressed in this cipher's key format.
DCODE_KEY = "37/ABCDEFGH/IJKLMNOPQRSTUVWXYZ"


def test_vector_dcode_monome():
    # dCode worked example: MONOME -> 34 36 35 36 34 5 (concatenated 34363536345).
    cipher = MonomeDinome()
    assert cipher.encode("MONOME", DCODE_KEY) == "34363536345"


def test_vector_dcode_dinome_decode():
    # dCode worked example: 4 30 35 36 34 5 -> DINOME.
    cipher = MonomeDinome()
    assert cipher.decode("4303536345", DCODE_KEY) == "DINOME"


def test_roundtrip_pangram():
    cipher = MonomeDinome()
    msg = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    assert cipher.decode(cipher.encode(msg, DCODE_KEY), DCODE_KEY) == msg


def test_roundtrip_second_digit_equals_prefix():
    # L=33, O=37, V=73, Z=77 put a prefix digit in the dinome's SECOND position;
    # decoding must still consume exactly two digits after a leading prefix.
    cipher = MonomeDinome()
    msg = "LOVEZONEHELLOWORLD"
    assert cipher.decode(cipher.encode(msg, DCODE_KEY), DCODE_KEY) == msg


def test_roundtrip_default_top_row():
    # Bare two-digit key: standard high-frequency top row (ETAOINSR), A-Z fill.
    cipher = MonomeDinome()
    msg = "ATTACKATDAWNXYZQJV"
    assert cipher.decode(cipher.encode(msg, "37"), "37") == msg


def test_roundtrip_keyed_top_and_fill():
    cipher = MonomeDinome()
    key = "26/SENORITA/CRYPTOGRAM"
    msg = "DEFENDTHEEASTWALLOFTHECASTLE"
    assert cipher.decode(cipher.encode(msg, key), key) == msg


def test_decode_ignores_grouping_spaces():
    cipher = MonomeDinome()
    ct = cipher.encode("MEETMEATDAWN", DCODE_KEY)
    grouped = " ".join(ct[i : i + 5] for i in range(0, len(ct), 5))
    assert cipher.decode(grouped, DCODE_KEY) == "MEETMEATDAWN"


@pytest.mark.slow
def test_crack_recovers_plaintext():
    # The crack is a wall-clock-timeout-bounded simulated annealing, so a single
    # (seed, timeout) is not reproducible across machines; it is also seed-flaky.
    # Validate that the solver *can* recover: at least one of a few seeds converges.
    cipher = MonomeDinome()
    scorer = get_scorer()
    plaintext = (
        "THEFROZENGROUNDCRUNCHEDUNDERFOOTASTHESCOUTSMOVEDQUIETLYTOWARD"
        "THERIVERWATCHINGFORANYSIGNOFTHEENEMYPATROLNEARTHEOLDSTONEBRIDGE"
    )
    ciphertext = cipher.encode(plaintext, DCODE_KEY)
    best = 0.0
    for seed in (12345, 1, 2, 3, 7):
        candidates = cipher.crack(ciphertext, scorer, top=5, rng=random.Random(seed), timeout=45)
        if candidates:
            best = max(
                best,
                max(
                    sum(a == b for a, b in zip(c.plaintext, plaintext, strict=False))
                    / len(plaintext)
                    for c in candidates
                ),
            )
        if best >= 0.95:
            break
    assert best >= 0.95
