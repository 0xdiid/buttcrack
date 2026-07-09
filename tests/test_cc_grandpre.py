"""Tests for the Grandpre homophonic cipher.

VECTOR source (verified by decode):
  * CryptoCrack user guide, "Grandpre" page (worked example). The 8x8 grid, with
    rows/columns numbered 1-8, is

          1 2 3 4 5 6 7 8
      1 | A D J A C E N T
      2 | N A Z A R E N E
      3 | A G G R I E V E
      4 | R E Q U I T E D
      5 | C H A T E A U X
      6 | H A L F B A C K
      7 | I M P U N I T Y
      8 | C R O S S B O W

    Row words ADJACENT/NAZARENE/AGGRIEVE/REQUITED/CHATEAUX/HALFBACK/IMPUNITY/
    CROSSBOW; the first column also spells a word (ANARCHIC) and all 26 letters
    appear. Plaintext "Happiness is good health and a bad memory" enciphers to
        61 24 73 73 35 75 26 84 85 45 84 33 87 87 48 52 16 11 63 18 52 62 17 48
        66 86 22 12 72 38 72 87 25 78
    Decoding that stream with this grid recovers HAPPINESSISGOODHEALTHANDABADMEMORY.
"""

import random

from buttcrack.ciphers.grandpre import Grandpre
from buttcrack.text import only_letters

GRID = "ADJACENT/NAZARENE/AGGRIEVE/REQUITED/CHATEAUX/HALFBACK/IMPUNITY/CROSSBOW"
VECTOR_CT = (
    "61 24 73 73 35 75 26 84 85 45 84 33 87 87 48 52 16 11 63 18 52 62 17 48 "
    "66 86 22 12 72 38 72 87 25 78"
)
VECTOR_PT = "HAPPINESSISGOODHEALTHANDABADMEMORY"


def test_vector_cryptocrack():
    # CryptoCrack Grandpre worked example; verified by decode (encryption is
    # one-to-many, so the vector is anchored on decode).
    c = Grandpre()
    assert c.decode(VECTOR_CT, GRID) == VECTOR_PT


def test_round_trip():
    c = Grandpre()
    msg = "Happiness is good health and a bad memory!"
    prepared = only_letters(msg)
    # Deterministic (no rng): first cell chosen for each letter.
    ct = c.encode(msg, GRID)
    assert c.decode(ct, GRID) == prepared
    # Randomized homophone choice still round-trips.
    ct_rng = c.encode(msg, GRID, rng=random.Random(12345))
    assert c.decode(ct_rng, GRID) == prepared


def test_output_is_homophonic_not_reciprocal():
    c = Grandpre()
    # Every plaintext letter becomes a two-digit code -> stream of N two-digit
    # numbers, so decode is not the inverse-shaped operation of encode.
    ct = c.encode("HAPPINESS", GRID)
    nums = ct.split()
    assert len(nums) == len("HAPPINESS")
    assert all(len(n) == 2 and n.isdigit() for n in nums)
    assert c.decode(ct, GRID) == "HAPPINESS"
