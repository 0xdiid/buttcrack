"""Tests for the turning (Fleissner) grille transposition cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.grille import Grille
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

# Published vector: ACA cipher sheet, aca.info/ciphers/Grille.pdf.
# A 4x4 turning grille with holes reported (ACA convention) as "1 8 10 12";
# turning 90 degrees clockwise each quarter and reading the filled grid
# horizontally. Plaintext "THE TURNING GRILLE" (16 letters) encodes to
# "TILUN RGHGE LTENI R" -> "TILUNRGHGELTENIR".
VECTOR_PLAINTEXT = "THE TURNING GRILLE"
VECTOR_KEY = "1 8 10 12"
VECTOR_CIPHERTEXT = "TILUNRGHGELTENIR"


def test_vector():
    cipher = Grille()
    assert cipher.encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_vector_comma_key():
    # comma-separated and 'width N;' prefixed forms are equivalent
    cipher = Grille()
    assert cipher.encode(VECTOR_PLAINTEXT, "1,8,10,12") == VECTOR_CIPHERTEXT
    assert cipher.encode(VECTOR_PLAINTEXT, "width 4; 1 8 10 12") == VECTOR_CIPHERTEXT


def test_vector_decode():
    cipher = Grille()
    assert cipher.decode(VECTOR_CIPHERTEXT, VECTOR_KEY) == only_letters(VECTOR_PLAINTEXT)


def test_roundtrip_exact_block():
    cipher = Grille()
    msg = "Defend the east"  # 13 letters -> one 4x4 block with padding
    enc = cipher.encode(msg, VECTOR_KEY)
    dec = cipher.decode(enc, VECTOR_KEY)
    # padding completes the block; the original letters are the prefix
    assert dec.startswith(only_letters(msg))


def test_roundtrip_multiblock_6x6():
    cipher = Grille()
    # 6x6 grille block = 36 letters; use 72 letters for two clean blocks.
    msg = "ACTIONWILLDESTROYYOURPROCRASTINATIONANDLEADYOUSTRAIGHTTOWARDFAILUREXX"
    msg = (msg + "Z" * (72 - len(msg)))[:72]
    key = "1 2 5 11 21 24 27 28 34"  # nine holes -> valid 6x6 grille
    enc = cipher.encode(msg, key)
    dec = cipher.decode(enc, key)
    assert dec == only_letters(msg)


@pytest.mark.slow
def test_crack_recovers_plaintext():
    cipher = Grille()
    scorer = get_scorer()
    rng = random.Random(7)
    plain = (
        "the turning grille is a transposition cipher that hides a message "
        "inside a square grid by writing letters through holes and turning"
    )
    letters = only_letters(plain).upper()
    # pad to a whole number of 4x4 blocks so there are no padding artefacts
    block = 16
    padded = letters + "X" * ((-len(letters)) % block)
    key = "1 8 10 12"
    ciphertext = cipher.encode(padded, key)

    candidates = cipher.crack(ciphertext, scorer, top=5, rng=rng)
    assert candidates, "crack returned no candidates"
    # crack strips trailing pad letters, so compare against the unpadded letters
    assert candidates[0].plaintext == letters
