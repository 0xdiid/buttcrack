"""Tests for the Bifid cipher.

VECTOR sources (row-by-row square fill, matching PolybiusSquare):
  * Wikipedia, "Bifid cipher" (en.wikipedia.org/wiki/Bifid_cipher):
    square rows BGWKZ/QPNDS/IOAXE/FCLUM/THYVR, period = whole message,
    FLEEATONCE -> UAEOLWRINS.
  * American Cryptogram Association cipher sheet "BIFID"
    (cryptogram.org/.../ciphers/Bifid.pdf): EXTRAORDINARY square written
    clockwise-spiral, rows EXTRA/KLMPO/HWZQD/GVUSI/FCBYN, period 7,
    ODDPERIODSAREPOPULAR -> MWEINGIMGEOYYRLVEYWY. The spiral square is supplied
    to the row-by-row PolybiusSquare by passing its letters read row-by-row.
"""

import random

import pytest

from buttcrack.ciphers.bifid import Bifid
from buttcrack.scoring import get_scorer


def test_vector_wikipedia():
    b = Bifid()
    # Full 25-letter square supplied as the keyword reproduces the published grid.
    key = "BGWKZQPNDSIOAXEFCLUMTHYVR/10"
    assert b.encode("FLEEATONCE", key) == "UAEOLWRINS"


def test_vector_aca_extraordinary():
    b = Bifid()
    # Spiral square written out row-by-row, period 7.
    key = "EXTRAKLMPOHWZQDGVUSIFCBYN/7"
    assert b.encode("Odd periods are popular", key) == "MWEINGIMGEOYYRLVEYWY"


def test_round_trip():
    # decode(encode(msg)) recovers the *prepared* plaintext: letters only,
    # uppercased, J->I. No padding is used (short final block kept at true length).
    b = Bifid()
    key = "SECRETKEYWORD/5"
    msg = "Attack the junction at dawn, jolly good!"
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z").replace("J", "I")
    ct = b.encode(msg, key)
    assert b.decode(ct, key) == prepared


def test_round_trip_short_final_block():
    # Period that does not divide the length exercises the short trailing block.
    b = Bifid()
    key = "MONARCHY/7"
    msg = "DEFENDTHEEASTWALLOFTHECASTLENOWXY"
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z").replace("J", "I")
    ct = b.encode(msg, key)
    assert b.decode(ct, key) == prepared


@pytest.mark.slow
def test_crack_recovers():
    b = Bifid()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
    )
    prepared = "".join(c for c in pt.upper() if "A" <= c <= "Z").replace("J", "I")
    ct = b.encode(pt, "CHARLES/5")
    res = b.crack(ct, scorer, rng=random.Random(11), timeout=180, max_period=7)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == prepared
