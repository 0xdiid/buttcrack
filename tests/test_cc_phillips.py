"""Tests for the Phillips cipher.

VECTOR source: CryptoCrack user guide, Phillips page (worked example, keyword
PATIENCE). The published example builds the keyed alphabet
``PATIENCBDFGHKLMOQRSUVWXYZ`` (J->I) and enciphers the plaintext in groups of
five using eight row-shifted squares; its published first groups are
``DRNDR MAQZL TRSKW OVYAY`` for plaintext ``THE THINGS THAT COME TO...``. Those
groups exercise squares #1-#4 exactly; the remaining groups follow the same
documented Row-variant scheme.
"""

import random

import pytest

from buttcrack.ciphers.phillips import Phillips
from buttcrack.scoring import get_scorer


def test_vector_cryptocrack_patience():
    p = Phillips()
    # CryptoCrack worked example (keyword PATIENCE). The first four 5-letter
    # groups DRNDR MAQZL TRSKW OVYAY are the published ciphertext.
    ct = p.encode("THE THINGS THAT COME TO THOSE WHO WAIT", "PATIENCE")
    assert ct.startswith("DRNDRMAQZLTRSKWOVYAY")
    assert ct == "DRNDRMAQZLTRSKWOVYAYRWZNTBWTKML"


def test_round_trip():
    # decode(encode(msg)) recovers the prepared plaintext: letters only,
    # uppercased, J->I. Encryption (down-right) and decryption (up-left) differ.
    p = Phillips()
    key = "MONARCHY"
    msg = "Attack the eastern junction at dawn, jolly good show old chap!"
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z").replace("J", "I")
    ct = p.encode(msg, key)
    assert p.decode(ct, key) == prepared


def test_round_trip_spans_full_period():
    # A message longer than the 40-letter period cycles through all eight squares
    # more than once, exercising every working square in both directions.
    p = Phillips()
    key = "PATIENCE"
    msg = "X" * 0 + ("THEQUICKBROWNFOXIUMPSOVERTHELAZYDOGWHILETHENIGHTWATCHMANSLEPTSOUNDLYAGAIN")
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z").replace("J", "I")
    ct = p.encode(msg, key)
    assert len(prepared) > 40
    assert p.decode(ct, key) == prepared


@pytest.mark.slow
def test_crack_recovers():
    p = Phillips()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
    )
    prepared = "".join(c for c in pt.upper() if "A" <= c <= "Z").replace("J", "I")
    ct = p.encode(pt, "CHARLES")
    res = p.crack(ct, scorer, rng=random.Random(7), timeout=180)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == prepared
