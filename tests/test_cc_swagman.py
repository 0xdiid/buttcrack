"""Tests for the Swagman cipher.

VECTOR source: American Cryptogram Association, *The ACA and You*,
``Swagman.pdf`` (cryptogram.org/downloads/aca.info/ciphers/Swagman.pdf).
Worked example, extracted verbatim via pdftotext:

  pt:  Don't be afraid to take a big leap if one is indicated. You cannot
       cross a river or a chasm in two small jumps.
  Key square (5x5 Latin square of 1..5):
       3 2 1 4 5 / 1 5 3 2 4 / 2 4 5 3 1 / 5 3 4 1 2 / 4 1 2 5 3
  ct:  ENDSC MORDA NIBOI SICTN ASTGB LTEWA OAREE FSAID VPYRM OEAIA FUILR
       LDOCO TJNRA AENOU NCMIT SOAPH SKATI

The published plaintext grid and ciphertext grid in the PDF were reproduced
exactly by an independent scratch implementation (the vector is self-consistent).
"""

import math
import random

import pytest

from buttcrack.ciphers.swagman import Swagman
from buttcrack.scoring import get_scorer


def test_vector_aca_swagman():
    s = Swagman()
    pt = (
        "Don't be afraid to take a big leap if one is indicated. "
        "You cannot cross a river or a chasm in two small jumps."
    )
    key = "32145/15324/24531/53412/41253"
    expected = (
        "ENDSCMORDANIBOISICTNASTGBLTEWAOAREEFSAIDVPYRMOEAIAFUILRLDOCOTJNRAAENOUNCMITSOAPHSKATI"
    )
    assert s.encode(pt, key) == expected


def test_flat_key_form():
    # The same square given as one block of n*n digits must encode identically.
    s = Swagman()
    pt = (
        "Don't be afraid to take a big leap if one is indicated. "
        "You cannot cross a river or a chasm in two small jumps."
    )
    flat = "3214515324245315341241253"
    rows = "32145/15324/24531/53412/41253"
    assert s.encode(pt, flat) == s.encode(pt, rows)


def test_round_trip():
    # decode(encode(msg, key), key) recovers the prepared plaintext, which is
    # the letters-only uppercase stream padded with X to fill the n-row grid.
    s = Swagman()
    key = "1234/2143/3412/4321"
    n = 4
    msg = "Meet me at the old bridge at midnight, bring the documents and a torch."
    letters = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    width = math.ceil(len(letters) / n)
    prepared = letters.ljust(n * width, "X")
    ct = s.encode(msg, key)
    assert s.decode(ct, key) == prepared


@pytest.mark.slow
def test_crack_recovers():
    s = Swagman()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
    )
    key = "1234/2143/3412/4321"
    n = 4
    width = math.ceil(len(pt) / n)
    prepared = pt.ljust(n * width, "X")
    ct = s.encode(pt, key)
    res = s.crack(ct, scorer, rng=random.Random(7), timeout=120)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == prepared
    # the reported key must re-decode to the same plaintext
    assert s.decode(ct, res[0].key) == prepared
