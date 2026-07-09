"""Tests for the Progressive Key cipher.

VECTOR source: CryptoCrack user guide, Progressive Key page
(https://sites.google.com/site/cryptocrackprogram/user-guide/cipher-types/substitution/progressive-key).
Worked example: plaintext "History will be kind to me for I intend to write it",
keyword POLITICS, progression 3, Vigenere base ->
ciphertext "WWDBHZAO ACZMAVNI YNFADTWP GFHKGEOU PWOCYYWX".

The progression applies per group (group size = keyword length = 8): group 0 gets
an additional shift of A (0), group 1 of D (3), group 2 of G (6), group 3 of J
(9), group 4 of M (12). Verified self-consistent with a from-scratch reference
implementation before this test was written.
"""

import random

import pytest

from buttcrack.ciphers.progressive_key import ProgressiveKey
from buttcrack.scoring import get_scorer


def test_vector_cryptocrack_politics():
    p = ProgressiveKey()
    ct = p.encode(
        "History will be kind to me for I intend to write it",
        "POLITICS/3",
    )
    assert ct == "WWDBHZAOACZMAVNIYNFADTWPGFHKGEOUPWOCYYWX"


def test_vector_default_base_is_vigenere():
    # Omitting the base must behave identically to an explicit vigenere base.
    p = ProgressiveKey()
    msg = "History will be kind to me for I intend to write it"
    assert p.encode(msg, "POLITICS/3") == p.encode(msg, "POLITICS/3/vigenere")


def test_round_trip_vigenere():
    p = ProgressiveKey()
    key = "POLITICS/3"
    msg = "History will be kind to me for I intend to write it"
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    assert p.decode(p.encode(msg, key), key) == prepared


def test_round_trip_all_bases():
    p = ProgressiveKey()
    msg = (
        "the quick brown fox jumps over the lazy dog while the night watchman "
        "sleeps soundly again and again"
    )
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    for base in ("vigenere", "beaufort", "variant", "porta"):
        for progression in (0, 1, 5, 7):
            key = f"SECRET/{progression}/{base}"
            assert p.decode(p.encode(msg, key), key) == prepared, key


def test_progression_zero_equals_plain_periodic():
    # With progression 0 the second layer is the identity, so it reduces to a
    # plain Vigenere with the keyword.
    p = ProgressiveKey()
    msg = "ATTACKATDAWNFROMTHEEASTERNRIDGE"
    ct = p.encode(msg, "LEMON/0")
    # Plain Vigenere of the same with LEMON.
    keyletters = [ord(c) - 65 for c in "LEMON"]
    expect = "".join(chr((ord(ch) - 65 + keyletters[i % 5]) % 26 + 65) for i, ch in enumerate(msg))
    assert ct == expect


@pytest.mark.slow
def test_crack_recovers_with_keyword_hint():
    p = ProgressiveKey()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
    )
    prepared = "".join(c for c in pt.upper() if "A" <= c <= "Z")
    ct = p.encode(pt, "POLITICS/3/vigenere")
    res = p.crack(
        ct,
        scorer,
        rng=random.Random(7),
        timeout=60,
        keyword="POLITICS",
    )
    assert res, "crack returned no candidates"
    assert res[0].plaintext == prepared


def test_crack_without_keyword_returns_empty():
    p = ProgressiveKey()
    scorer = get_scorer()
    ct = p.encode("ATTACKATDAWNFROMTHEEASTERNRIDGEUNDERHEAVYFOG", "POLITICS/3")
    assert p.crack(ct, scorer, timeout=5) == []
