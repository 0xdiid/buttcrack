"""Tests for the Periodic Gromark cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.periodic_gromark import PeriodicGromark
from buttcrack.scoring import get_scorer

# Published worked example from the CryptoCrack user guide, Periodic Gromark page.
# Keyword WRIGHT (period 6); primer 643125 (alphabetical rank of W R I G H T).
# Running key by chain addition from primer 643125: 6431250743757170...
VECTOR_PLAINTEXT = "ADOCTORCANBURYHISMISTAKESBUTANARCHITECTCANONLYADVISEHISCLIENTSTOPLANTVINES"
VECTOR_KEY = "WRIGHT"
VECTOR_CIPHERTEXT = "DMRPZKUWZMXSNVAZMAZGGMKHMLVUIJFRPPYTHNOJMLHUMTWWSYMFXXYUOCBKEENFOPPRORILHX"


def test_vector_encode():
    assert PeriodicGromark().encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_vector_decode():
    assert PeriodicGromark().decode(VECTOR_CIPHERTEXT, VECTOR_KEY) == VECTOR_PLAINTEXT


def test_roundtrip():
    c = PeriodicGromark()
    msg = "ATTACKATDAWNTHEENEMYISNEARANDWEMUSTHOLDTHELINEUNTILREINFORCEMENTS"
    key = "TEACHER"
    assert c.decode(c.encode(msg, key), key) == msg


def test_roundtrip_distinct_keyword():
    c = PeriodicGromark()
    msg = "MEETMEATMIDNIGHTBYTHEOLDOAKTREENEARTHERIVERBEND"
    key = "KEYWORD"
    assert c.decode(c.encode(msg, key), key) == msg


def test_encode_strips_non_letters():
    c = PeriodicGromark()
    spaced = c.encode("a doctor can!", VECTOR_KEY)
    clean = c.encode("ADOCTORCAN", VECTOR_KEY)
    assert spaced == clean


@pytest.mark.slow
def test_crack_with_keyword_hint():
    scorer = get_scorer()
    c = PeriodicGromark()
    rng = random.Random(11)
    plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHESLEEPINGHOUNDDREAMSOFCHASINGRABBITS"
    key = "MERCURY"
    ct = c.encode(plaintext, key)
    # Decoy keywords plus the true one; the scorer must pick the real decryption.
    out = c.crack(
        ct,
        scorer,
        top=5,
        rng=rng,
        timeout=30.0,
        keywords=["WRIGHT", "TEACHER", "KEYWORD", "MERCURY", "PLANETS"],
    )
    assert out, "expected at least one candidate"
    assert out[0].plaintext.upper().replace(" ", "") == plaintext
    assert out[0].key == "MERCURY"
