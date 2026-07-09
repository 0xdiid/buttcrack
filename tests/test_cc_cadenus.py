"""Tests for the Cadenus cipher.

VECTOR source: ACA worked example reproduced by The Black Chamber
(https://theblackchamber552383191.wordpress.com/2020/11/16/cadenus/), keyword
``SET``.  The article gives the plaintext "to all men there is a season, and in
that season you turn turn turn and if you stop you burn burn", states the
per-column rotations explicitly ("rotating the 'E' column up by 4, 'S' up by 18
and 'T' up by 19" -- exactly the index of each letter in the 25-letter row
alphabet ABCDEFGHIJKLMNOPQRSTUVXYZ), and publishes the ciphertext
"IDUSY OSSOA PUIUB HRNSU ASTMY LTTER NHSRE EUAOA ANINN ODATT EYTOB AONNU
RUTOR NLURN TNENF".  A from-scratch reimplementation reproduces both the final
ciphertext and the article's intermediate rotated grid exactly.

(The CryptoCrack/Black-Chamber "CALM" example is internally inconsistent -- its
published ciphertext contains a literal W despite the V/W-combined rule -- so it
is NOT used as the authoritative vector.)
"""

import random

import pytest

from buttcrack.ciphers.cadenus import Cadenus
from buttcrack.scoring import get_scorer

# ACA / Black Chamber worked example (keyword SET).
SET_PLAINTEXT = (
    "to all men there is a season, and in that season "
    "you turn turn turn and if you stop you burn burn"
)
SET_CIPHERTEXT = "IDUSYOSSOAPUIUBHRNSUASTMYLTTERNHSREEUAOAANINNODATTEYTOBAONNURUTORNLURNTNENF"


def _prepared(text: str) -> str:
    """The clean stream the cipher actually operates on: A-Z only, W -> V."""
    return "".join(("V" if ch == "W" else ch) for ch in text.upper() if "A" <= ch <= "Z")


def test_vector_aca_set():
    # encode(published_plaintext, key) == published_ciphertext, exactly.
    c = Cadenus()
    assert c.encode(SET_PLAINTEXT, "SET") == SET_CIPHERTEXT


def test_round_trip():
    # decode(encode(msg, key), key) recovers the prepared plaintext (W folded to V).
    c = Cadenus()
    key = "DARKEN"
    # 25 * 6 = 150 letters of clean stream.
    msg = (
        "When in the course of human events it becomes necessary for one people "
        "to dissolve the political bands which have connected them with another "
        "and assume the powers of the earth the separate"
    )
    prepared = _prepared(msg)[: 25 * len(key)]
    ct = c.encode(prepared, key)
    assert c.decode(ct, key) == prepared


def test_w_folds_to_v():
    # W is folded into V on a 25-row grid; round-trip preserves the folded stream.
    c = Cadenus()
    key = "WOLF"
    msg = "WWWW" + "ABCDEFGHIJKLMNOPQRSTUVXYZ" * 4
    prepared = _prepared(msg)[: 25 * len(key)]
    assert "W" not in c.encode(prepared, key)
    assert c.decode(c.encode(prepared, key), key) == prepared


@pytest.mark.slow
def test_crack_recovers():
    c = Cadenus()
    scorer = get_scorer()
    rng = random.Random(11)
    # 100-letter clean stream -> keyword length 4.
    pt = (
        "IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES IT WAS THE AGE OF "
        "WISDOM IT WAS THE AGE OF FOOLISHNESS IT WAS THE EPOCH OF BELIEF"
    )
    prepared = _prepared(pt)[:100]
    ct = c.encode(prepared, "DARK")

    # Given the keyword as a hint, the dictionary search recovers it exactly.
    hinted = c.crack(ct, scorer, rng=rng, keyword="DARK")
    assert hinted, "hinted crack returned no candidates"
    assert hinted[0].key == "DARK"
    assert _prepared(hinted[0].plaintext) == prepared

    # Keyless dictionary sweep: the true keyword surfaces among the top
    # candidates.  (Cadenus admits near-English cyclic alternatives, so the #1
    # slot is not guaranteed -- recovery means the truth ranks at the top.)
    res = c.crack(ct, scorer, rng=rng, timeout=120, top=5)
    assert res, "keyless crack returned no candidates"
    assert any(r.key == "DARK" and _prepared(r.plaintext) == prepared for r in res)
