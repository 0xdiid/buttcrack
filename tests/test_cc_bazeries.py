"""Tests for the Bazeries cipher.

VECTOR source:
  * CryptoCrack user guide, "Bazeries" page. Key number 7352; plaintext
    "A CLEAR CONSCIENCE IS USUALLY THE SIGN OF A BAD MEMORY". The keyed
    right-hand alphabet is SEVNTHOUADRFIYWBCGJKLMPQXZ (the number spelled out,
    SEVENTHOUSANDTHREEHUNDREDANDFIFTYTWO, deduplicated then completed). The
    transposition cycles group lengths 7,3,5,2 and reverses each group, giving
    the intermediate "craelca sno cneic ie llausus hty ngise fo memdaba yro";
    substitution then yields ciphertext
    "RASMVRS YIG RIMCR CM VVSXYXY FKL IOCYM EG UMUBSHS LAG".
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.bazeries import Bazeries
from buttcrack.scoring import get_scorer


def _letters(s: str) -> str:
    return "".join(c for c in s.upper() if "A" <= c <= "Z")


def test_vector_cryptocrack():
    b = Bazeries()
    plaintext = "A CLEAR CONSCIENCE IS USUALLY THE SIGN OF A BAD MEMORY"
    ciphertext = _letters("RASMVRS YIG RIMCR CM VVSXYXY FKL IOCYM EG UMUBSHS LAG")
    assert b.encode(plaintext, "7352") == ciphertext


def test_round_trip():
    # decode(encode(msg)) recovers the prepared plaintext: letters only,
    # uppercased, J merged into I.
    b = Bazeries()
    key = "53124"
    msg = "Attack the junction at dawn, jolly good fellows!"
    prepared = _letters(msg).replace("J", "I")
    ct = b.encode(msg, key)
    assert b.decode(ct, key) == prepared


def test_round_trip_vector_key():
    b = Bazeries()
    msg = "DEFEND THE EAST WALL OF THE CASTLE AT ONCE"
    prepared = _letters(msg).replace("J", "I")
    ct = b.encode(msg, "7352")
    assert b.decode(ct, "7352") == prepared


@pytest.mark.slow
def test_crack_recovers():
    b = Bazeries()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITY"
    )
    prepared = _letters(pt).replace("J", "I")
    ct = b.encode(pt, "7352")
    res = b.crack(ct, scorer, rng=random.Random(3), timeout=120, max_key=10000)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == prepared
    assert res[0].key == "7352"
