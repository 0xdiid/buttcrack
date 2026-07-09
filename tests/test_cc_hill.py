"""Tests for the Hill cipher (2x2 and 3x3 matrix cipher mod 26).

Vector sources: Wikipedia "Hill cipher" (en.wikipedia.org/wiki/Hill_cipher) and
the Crypto Corner "Hill Cipher" worked example.

2x2: K = [[3,3],[2,5]], plaintext "HELP".
  HE=(7,4)^T -> [[3,3],[2,5]](7,4) = (33,38) = (7,12) mod 26 = "HI"
  LP=(11,15)^T -> (78,97) = (0,19) mod 26 = "AT"   => ciphertext "HIAT".

3x3: K = [[6,24,1],[13,16,10],[20,17,15]], plaintext "ACT".
  ACT=(0,2,19)^T -> K*(0,2,19) = (67,222,319) = (15,14,7) mod 26 = "POH".
  The same key sends "CAT" -> "FIN".
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.hill import Hill
from buttcrack.scoring import get_scorer

# 2x2 published vector (Wikipedia / Crypto Corner).
KEY_2X2 = "3,3,2,5"
PT_2X2 = "HELP"
CT_2X2 = "HIAT"

# 3x3 published vector (Wikipedia).
KEY_3X3 = "6,24,1,13,16,10,20,17,15"
PT_3X3 = "ACT"
CT_3X3 = "POH"


def test_vector_2x2():
    """Encode reproduces the published 2x2 Wikipedia ciphertext exactly."""
    assert Hill().encode(PT_2X2, KEY_2X2) == CT_2X2


def test_vector_3x3():
    """Encode reproduces the published 3x3 Wikipedia ciphertext exactly."""
    assert Hill().encode(PT_3X3, KEY_3X3) == CT_3X3


def test_vector_3x3_second():
    """Same 3x3 key sends CAT -> FIN (Wikipedia)."""
    assert Hill().encode("CAT", KEY_3X3) == "FIN"


def test_keyword_key_2x2():
    """A 4-letter keyword (DDCF = 3,3,2,5) builds the same 2x2 matrix."""
    assert Hill().encode(PT_2X2, "DDCF") == CT_2X2


def test_roundtrip_2x2():
    """decode(encode(...)) recovers the prepared plaintext (2x2)."""
    c = Hill()
    msg = "Defend the east wall of the castle now"
    prepared = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z")
    if len(prepared) % 2:
        prepared += "X"
    assert c.decode(c.encode(msg, KEY_2X2), KEY_2X2) == prepared


def test_roundtrip_3x3():
    """decode(encode(...)) recovers the prepared plaintext (3x3)."""
    c = Hill()
    msg = "Meet me at dawn by the old oak tree"
    prepared = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z")
    if len(prepared) % 3:
        prepared += "X" * (3 - len(prepared) % 3)
    assert c.decode(c.encode(msg, KEY_3X3), KEY_3X3) == prepared


def test_roundtrip_keyword_3x3():
    """A 9-letter keyword builds a 3x3 matrix that round-trips."""
    c = Hill()
    key = "GYBNQKURP"  # invertible mod 26 (det 25)
    msg = "The quick brown fox jumps over the lazy dog"
    prepared = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z")
    if len(prepared) % 3:
        prepared += "X" * (3 - len(prepared) % 3)
    assert c.decode(c.encode(msg, key), key) == prepared


def test_non_invertible_matrix_raises():
    """A matrix with a determinant sharing a factor with 26 is rejected."""
    c = Hill()
    # det([[2,4],[6,8]]) = 16-24 = -8 = 18 mod 26, gcd(18,26)=2 -> not invertible.
    with pytest.raises(ValueError):
        c.decode("ABCD", "2,4,6,8")


@pytest.mark.slow
def test_crack_recovers_2x2():
    """Keyless 2x2 crack recovers the plaintext by hill-climbing."""
    c = Hill()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
    )
    prepared = "".join(ch for ch in pt.upper() if "A" <= ch <= "Z")
    if len(prepared) % 2:
        prepared += "X"
    ct = c.encode(pt, KEY_2X2)
    res = c.crack(ct, scorer, rng=random.Random(7), timeout=180)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == prepared
