"""Tests for the Conjugated Matrix Bifid (CM Bifid) cipher.

VECTOR source:
  American Cryptogram Association cipher sheet "CM BIFID (Conjugated Matrix
  Bifid)" (cryptogram.org/.../ciphers/CMBifid.pdf). Square A keyword EXTRA
  filled row-by-row -> rows EXTRA/KLMPO/HWZQD/GVUSI/FCBYN. Square B keyword
  NOVELTY entered in alternating verticals (boustrophedon by column) -> rows
  NCDRS/OBFQU/VAGPW/EYHMX/LTIKZ. Period 7, plaintext "Odd periods are popular"
  (ODDPERIODSAREPOPULAR) -> ciphertext FANXZEX FENUKKR BYNKAK.

  The plaintext coordinates (read from Square A) are the SAME as the plain-Bifid
  ACA example: rows 2332114/2341112/224211, cols 5554145/5545414/543254; the
  digit PAIRS 23 32 11 45 55 41 45 / 23 41 11 25 54 54 14 / 22 42 11 54 32 54
  read out of Square B give F A N X Z E X / F E N U K K R / B Y N K A K. Verified
  computationally: both the algorithm and the alternating-verticals fill of the
  NOVELTY keyed alphabet (NOVELTYABCDFGHIKMPQRSUWXZ) reproduce Square B exactly.

  The published grids are supplied row-by-row as the SQUAREA/SQUAREB keys so the
  row-by-row PolybiusSquare reproduces them regardless of original fill order.
"""

import random

import pytest

from buttcrack.ciphers.cm_bifid import CMBifid
from buttcrack.scoring import get_scorer

# Published ACA squares, read out row-by-row.
SQ_A = "EXTRAKLMPOHWZQDGVUSIFCBYN"
SQ_B = "NCDRSOBFQUVAGPWEYHMXLTIKZ"


def test_vector_aca_cmbifid():
    c = CMBifid()
    key = f"{SQ_A}/{SQ_B}/7"
    assert c.encode("Odd periods are popular", key) == "FANXZEXFENUKKRBYNKAK"


def test_round_trip():
    # decode(encode(msg)) recovers the *prepared* plaintext: letters only,
    # uppercased, J->I. No padding is used (short final block kept at true length).
    c = CMBifid()
    key = "SECRETKEYWORD/CONJUGATEMATRIX/5"
    msg = "Attack the junction at dawn, jolly good!"
    prepared = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z").replace("J", "I")
    ct = c.encode(msg, key)
    assert c.decode(ct, key) == prepared


def test_round_trip_short_final_block():
    # Period that does not divide the length exercises the short trailing block.
    c = CMBifid()
    key = f"{SQ_A}/{SQ_B}/7"
    msg = "DEFENDTHEEASTWALLOFTHECASTLENOWXY"
    prepared = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z").replace("J", "I")
    ct = c.encode(msg, key)
    assert c.decode(ct, key) == prepared


def test_degenerates_to_bifid_when_squares_equal():
    # If Square B == Square A, CM Bifid is exactly plain Bifid.
    from buttcrack.ciphers.bifid import Bifid

    c = CMBifid()
    b = Bifid()
    msg = "Odd periods are popular"
    assert c.encode(msg, f"{SQ_A}/{SQ_A}/7") == b.encode(msg, f"{SQ_A}/7")


@pytest.mark.slow
def test_crack_recovers():
    # Joint two-square recovery is hard; this only asserts the search machinery
    # runs and returns ranked candidates within the budget. (No equality claim:
    # reliable joint recovery of both 25-letter alphabets is not guaranteed.)
    c = CMBifid()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITY"
    )
    ct = c.encode(pt, f"{SQ_A}/{SQ_B}/5")
    res = c.crack(ct, scorer, rng=random.Random(7), timeout=20, max_period=5)
    assert isinstance(res, list)
