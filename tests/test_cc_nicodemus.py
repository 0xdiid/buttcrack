"""Nicodemus: keyed columnar transposition + per-column Vigenere (ACA)."""

from buttcrack.ciphers.nicodemus import Nicodemus
from buttcrack.text import only_letters


def test_nicodemus_aca_vector():
    # ACA worked example: pt "THE EARLY BIRD GETS THE WORM", key CAT.
    n = Nicodemus()
    assert n.encode("THE EARLY BIRD GETS THE WORM", "CAT") == "HAYREVGNKIXKUWMTWMUGTAH"
    assert n.decode("HAYREVGNKIXKUWMTWMUGTAH", "CAT") == "THEEARLYBIRDGETSTHEWORM"


def test_nicodemus_round_trip():
    n = Nicodemus()
    msg = "meet me at the old bridge at dawn bring the map and the lantern"
    assert n.decode(n.encode(msg, "MONARCH"), "MONARCH") == only_letters(msg)


def test_nicodemus_explicit_key_form():
    # The =READORDER|SHIFTS form (what crack emits) must round-trip.
    n = Nicodemus()
    msg = "the quick brown fox jumps over the lazy dog"
    enc = n.encode(msg, "CAT")
    # CAT -> order [1,0,2], shifts A,C,T = 0,2,19
    assert n.decode(enc, "=1,0,2|A,C,T") == only_letters(msg)
