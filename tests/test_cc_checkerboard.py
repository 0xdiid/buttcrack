"""Tests for the ACA Checkerboard digraphic substitution cipher.

VECTOR source (row-by-row square fill, matching PolybiusSquare):
  * CryptoCrack user guide, "Checkerboard" page (worked example): square keyword
    BACKUP, row keyword BRAIN, column keyword WAVES; the grid is

          W A V E S
      B | B A C K U
      R | P D E F G
      A | H I L M N
      I | O Q R S T
      N | V W X Y Z

    Plaintext I HAVENT LOST (letters IHAVENTLOST) enciphers digraph-by-digraph
    to AA AW BA NW RV AS IS AV IW IE IS, i.e. AAAWBANWRVASISAVIWIEIS. (I/J merge:
    I -> cell(B,W) = AA, H -> cell(B,A) = AW.)
"""

import random

import pytest

from buttcrack.ciphers.checkerboard import Checkerboard
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def test_vector_cryptocrack():
    c = Checkerboard()
    # CryptoCrack Checkerboard worked example (square BACKUP, rows BRAIN, cols WAVES).
    assert c.encode("I HAVENT LOST", "BACKUP/BRAIN/WAVES") == "AAAWBANWRVASISAVIWIEIS"


def test_vector_is_digraphic_not_reciprocal():
    c = Checkerboard()
    ct = c.encode("IHAVENTLOST", "BACKUP/BRAIN/WAVES")
    # ciphertext is twice the plaintext length and decode recovers the plaintext
    assert len(ct) == 2 * len("IHAVENTLOST")
    assert c.decode(ct, "BACKUP/BRAIN/WAVES") == "IHAVENTLOST"
    # encode != decode (not reciprocal)
    assert c.decode(ct, "BACKUP/BRAIN/WAVES") != c.encode(ct, "BACKUP/BRAIN/WAVES")


def test_round_trip():
    # decode(encode(msg)) recovers the prepared plaintext: letters only, J->I.
    c = Checkerboard()
    key = "SECRETKEY/MONDY/CLAWS"
    msg = "Attack the junction at dawn, jolly good!"
    prepared = only_letters(msg).replace("J", "I")
    ct = c.encode(msg, key)
    assert c.decode(ct, key) == prepared


@pytest.mark.slow
def test_crack_recovers():
    c = Checkerboard()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
    )
    prepared = only_letters(pt).replace("J", "I")
    ct = c.encode(pt, "CHARLES/BRAIN/WAVES")
    res = c.crack(ct, scorer, rng=random.Random(7), timeout=120)
    assert res, "crack returned no candidates"
    matches = sum(a == b for a, b in zip(res[0].plaintext, prepared, strict=False))
    assert matches / len(prepared) >= 0.9
