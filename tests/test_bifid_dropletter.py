"""Tests for the bifid drop-letter parameter (Task 2).

Standard 5x5 bifid drops J (merging J->I); a differently-dropped square omits some
other letter. These cover the new ``drop_letter`` threading through the Bifid cipher's
key format, the module-level ``bifid_encode``/``bifid_decode`` helpers, and the
``drop_letters`` sweep in ``Bifid.crack`` — plus the structural failure of assuming J
when the real square drops something else.
"""

from __future__ import annotations

import random

from buttcrack.ciphers.bifid import (
    Bifid,
    bifid_decode,
    bifid_encode,
    square_alphabet,
)
from buttcrack.scoring import get_scorer

# A Q-free (and J-free after merge) passage, so dropping Q is lossless.
QFREE = "".join(
    c
    for c in (
        "THEOLDLIGHTHOUSESTOODALONEONTHEROCKYHEADLANDWHERETHEWINDANDWAVES"
        "HADWORNTHESTONEFORMANYYEARSANDTHEKEEPERWALKEDTHESPIRALSTAIRS"
    ).upper()
    if "A" <= c <= "Z"
).replace("J", "I")


def test_square_alphabet_drop():
    assert square_alphabet("J") == "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # classic no-J
    aq = square_alphabet("Q")
    assert "Q" not in aq and "J" in aq and len(aq) == 25


def test_roundtrip_default_j():
    b = Bifid()
    ct = b.encode(QFREE, "SECRET/5")
    assert b.decode(ct, "SECRET/5") == QFREE


def test_roundtrip_custom_drop_via_key_and_helpers():
    b = Bifid()
    # 3-part key names the dropped letter.
    ct = b.encode(QFREE, "SECRET/5/Q")
    assert b.decode(ct, "SECRET/5/Q") == QFREE
    # module-level helpers agree with the class.
    ct2 = bifid_encode(QFREE, "SECRET", 5, drop_letter="Q")
    assert ct2 == ct
    assert bifid_decode(ct2, "SECRET", 5, drop_letter="Q") == QFREE


def test_j_assumption_fails_on_q_dropped_cipher():
    """Decoding a Q-dropped bifid under the J-drop assumption corrupts the plaintext."""
    b = Bifid()
    ct = bifid_encode(QFREE, "SECRET", 5, drop_letter="Q")
    wrong = b.decode(ct, "SECRET/5")  # assumes J
    assert wrong != QFREE
    right = b.decode(ct, "SECRET/5/Q")
    assert right == QFREE


def test_crack_honours_drop_letter_opt():
    """Bifid.crack threads the drop letter into its candidate keys/meta."""
    b = Bifid()
    ct = bifid_encode(QFREE, "CHARLES", 5, drop_letter="Q")
    res = b.crack(
        ct,
        get_scorer(),
        rng=random.Random(0),
        timeout=3,
        periods=[5],
        drop_letters="Q",
        restarts=1,
        iters=60,
    )
    assert res, "crack returned no candidates"
    # every candidate came from the requested drop-letter and encodes it in the key.
    assert all(c.meta.get("drop_letter") == "Q" for c in res)
    assert res[0].key.endswith("/Q")
