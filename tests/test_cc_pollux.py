"""Tests for the Pollux cipher (homophonic morse fractionation).

VECTOR source:
  The Black Chamber, "Pollux"
  (https://theblackchamber552383191.wordpress.com/2020/11/19/pollux/).
  Plaintext "THIS IS A TEST" with key 1=. 2=. 3=- 4=x 5=- 6=- 7=. 8=x 9=x 0=-
  enciphers to the morse stream -x....x..x...xx..x...xx.-xx-x.x...x- and the
  published ciphertext "58172 14779 21284 71411 79913 94081 47218 0".

Pollux encipherment is homophonic (each morse symbol maps to several digits, so
the encipherer freely picks one), hence non-deterministic: the published digit
string is one of many valid realizations. The deterministic, well-defined
contract is DECRYPTION, so the vector is verified in that direction — the
published ciphertext must decrypt to the published plaintext, and the morse
stream produced from the published digits must match the source exactly.
"""

import random

import pytest

from buttcrack.ciphers.morse import text_to_morse
from buttcrack.ciphers.pollux import Pollux
from buttcrack.scoring import get_scorer

# 1=. 2=. 3=- 4=x 5=- 6=- 7=. 8=x 9=x 0=-  (indexed by digits 1..9,0)
KEY = "..-x--.xx-"
PUBLISHED_CT = "58172 14779 21284 71411 79913 94081 47218 0"
PUBLISHED_PT = "THIS IS A TEST"
PUBLISHED_STREAM = "-x....x..x...xx..x...xx.-xx-x.x...x-"


def test_vector_decrypt():
    """Published ciphertext decrypts to the published plaintext (deterministic)."""
    p = Pollux()
    assert p.decode(PUBLISHED_CT, KEY) == PUBLISHED_PT


def test_vector_morse_stream():
    """The published digits map, via the key, to the source's morse stream."""
    mapping = {
        "1": ".",
        "2": ".",
        "3": "-",
        "4": "x",
        "5": "-",
        "6": "-",
        "7": ".",
        "8": "x",
        "9": "x",
        "0": "-",
    }
    stream = "".join(mapping[d] for d in PUBLISHED_CT if d.isdigit())
    assert stream == PUBLISHED_STREAM
    # And our own morse encoder agrees on the plaintext's stream.
    assert text_to_morse(PUBLISHED_PT) == PUBLISHED_STREAM


def test_encode_is_valid_realization():
    """Our (deterministic) encode is a valid homophonic realization: it decodes back."""
    p = Pollux()
    ct = p.encode(PUBLISHED_PT, KEY)
    assert ct.isdigit()
    # Same morse stream as the published vector.
    mapping = {
        "1": ".",
        "2": ".",
        "3": "-",
        "4": "x",
        "5": "-",
        "6": "-",
        "7": ".",
        "8": "x",
        "9": "x",
        "0": "-",
    }
    stream = "".join(mapping[d] for d in ct)
    assert stream == PUBLISHED_STREAM
    assert p.decode(ct, KEY) == PUBLISHED_PT


def test_round_trip_deterministic():
    p = Pollux()
    msg = "Attack at dawn, the enemy sleeps!"
    expected = "ATTACK AT DAWN THE ENEMY SLEEPS"
    ct = p.encode(msg, KEY)
    assert p.decode(ct, KEY) == expected


def test_round_trip_randomized():
    """A randomized realization still decrypts deterministically."""
    p = Pollux()
    msg = "MEET ME AT THE OLD BRIDGE TONIGHT"
    ct = p.encode(msg, KEY, rng=random.Random(42))
    assert p.decode(ct, KEY) == msg


@pytest.mark.slow
def test_crack_recovers():
    p = Pollux()
    scorer = get_scorer()
    pt = (
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THEN RUNS "
        "AWAY INTO THE DARK FOREST BEFORE THE COLD WINTER NIGHT FALLS"
    )
    expected = " ".join(pt.split())
    ct = p.encode(pt, KEY, rng=random.Random(3))
    res = p.crack(ct, scorer, rng=random.Random(1), timeout=120)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == expected
