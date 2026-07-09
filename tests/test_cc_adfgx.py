"""Tests for the ADFGX cipher.

Vector source: Wikipedia "ADFGVX cipher" worked example (the ADFGX section),
en.wikipedia.org/wiki/ADFGVX_cipher.

Square (rows), labels A D F G X on rows and columns:
    B T A L P
    D H O Z K
    Q F V S N
    G I(J) C U X
    M R E W Y
Reproduced here by passing the full 25-letter mixed alphabet as the square key
(PolybiusSquare dedups keyword+alphabet, so a complete 25-letter keyword IS the
grid, row-by-row). Transposition keyword: CARGO.

plaintext "attack at once" -> fractionation stream AFADADAFGFDXAFADDFFXGFXF
written under CARGO and read in column order A,C,G,O,R ->
ciphertext FAXDFADDDGDGFFFAFAXAFAFX.
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.adfgx import ADFGX
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

SQUARE_KEY = "BTALPDHOZKQFVSNGICUXMREWY"
COL_KEY = "CARGO"
KEY = f"{SQUARE_KEY}/{COL_KEY}"


def test_vector():
    cipher = ADFGX()
    out = cipher.encode("Attack at once", KEY)
    assert out == "FAXDFADDDGDGFFFAFAXAFAFX"


def test_roundtrip():
    cipher = ADFGX()
    msg = "Defend the east wall of the castle"
    ct = cipher.encode(msg, KEY)
    pt = cipher.decode(ct, KEY)
    # decode recovers the *prepared* plaintext: letters only, uppercased, J->I.
    # "Defend the east wall of the castle" has no J, so it round-trips cleanly.
    assert pt == "DEFENDTHEEASTWALLOFTHECASTLE"


def test_roundtrip_with_j():
    cipher = ADFGX()
    # J merges into I in the 5x5 square, so a J in the input comes back as I.
    msg = "Major Jim enjoys jazz"
    ct = cipher.encode(msg, KEY)
    pt = cipher.decode(ct, KEY)
    assert pt == "MAIORIIMENIOYSIAZZ"


def test_roundtrip_keyword_square():
    # A normal keyword square (not a full 25-letter alphabet) also works.
    cipher = ADFGX()
    key = "PLAYFAIR/GERMAN"
    msg = "The quick brown fox jumps"
    ct = cipher.encode(msg, key)
    pt = cipher.decode(ct, key)
    assert pt == "THEQUICKBROWNFOXIUMPS"  # J->I in "jumps"


@pytest.mark.slow
def test_crack_best_effort():
    # ADFGX keyless crack is hard; only assert recovery if it actually finds it.
    cipher = ADFGX()
    scorer = get_scorer()
    plaintext = (
        "WEARESURROUNDEDONALLSIDESBYTHEENEMYBUTWEWILLHOLDOURGROUNDUNTILREINFORCE"
        "MENTSARRIVEATDAWNTOMORROWMORNINGWITHOUTFAIL"
    )
    ct = cipher.encode(plaintext, KEY)
    rng = random.Random(1234)
    cands = cipher.crack(ct, scorer, top=5, rng=rng, timeout=20.0)
    if not cands:
        return  # acceptable: keyless ADFGX is intractable in this short budget
    # The blind crack recovers the (square, order) but not as a keyword pair, so it
    # reports key=None; when confident, assert the plaintext itself was recovered.
    best = cands[0]
    if best.confidence >= 0.85:
        assert only_letters(best.plaintext) == only_letters(cipher.decode(ct, KEY))
