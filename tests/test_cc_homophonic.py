"""ACA Homophonic cipher: four numeric codes per letter."""

import random

from buttcrack.ciphers.homophonic import Homophonic
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def test_homophonic_decode_vector():
    # CryptoCrack worked example, keyword SLOW (encrypt is random, decode unique).
    h = Homophonic()
    vec = "70 84 76 48 29 73 41 03 86 70 55 90 80 01 98 34 70 71 28 18 55 97 26 22 76 67 33 34"
    assert h.decode(vec, "SLOW") == "HEWHOLAUGHSLASTTHINKSSLOWEST"


def test_homophonic_round_trip():
    h = Homophonic()
    msg = "defend the east wall of the castle at dawn"
    assert h.decode(h.encode(msg, "MILK"), "MILK") == only_letters(msg).replace("J", "I")


def test_homophonic_crack():
    h = Homophonic()
    pt = (
        "we shall fight on the beaches we shall fight on the landing grounds "
        "we shall never surrender whatever the cost may be"
    )
    ct = h.encode(pt, "DUSK")
    best = h.crack(ct, get_scorer(), rng=random.Random(1), timeout=30)[0]
    assert best.key == "DUSK"
    assert best.plaintext == only_letters(pt).replace("J", "I")
