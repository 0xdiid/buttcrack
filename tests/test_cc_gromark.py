"""Tests for the Gromark cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.gromark import Gromark
from buttcrack.scoring import get_scorer

# Published worked example from the CryptoCrack user guide, Gromark page.
# Primer 32941, mixed-alphabet keyword GRONSFELD. Running key (chain addition
# from primer 32941): 32941513566481202932212543379760663662992818109991988007680734870725779724...
VECTOR_PLAINTEXT = "ONLYTWOTHINGSAREINFINITETHEUNIVERSEANDHUMANSTUPIDITYANDIMNOTSUREABOUTTHEFORMER"
VECTOR_KEY = "32941/GRONSFELD"
VECTOR_CIPHERTEXT = "OHRERPHTMNUQDPUYQTGQHABASQXPTHPYSIXJUFVKNGNDRRIOMAEJGZKHCBNDBIWLDGVWDDVLXCSCZS"


def test_vector_encode():
    assert Gromark().encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_vector_decode():
    assert Gromark().decode(VECTOR_CIPHERTEXT, VECTOR_KEY) == VECTOR_PLAINTEXT


def test_roundtrip():
    c = Gromark()
    msg = "ATTACKATDAWNTHEENEMYISNEARANDWEMUSTHOLDTHELINE"
    key = "58032/CRYPTOGRAM"
    assert c.decode(c.encode(msg, key), key) == msg


def test_roundtrip_alt_key_form():
    # The "<digits><letters>" fallback form parses the same as slash form.
    c = Gromark()
    msg = "MEETMEATMIDNIGHTBYTHEOLDOAKTREE"
    assert c.decode(c.encode(msg, "12345TEMPEST"), "12345TEMPEST") == msg


@pytest.mark.slow
def test_crack_with_keyword_hint():
    scorer = get_scorer()
    c = Gromark()
    rng = random.Random(7)
    plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHESLEEPINGHOUNDDREAMSOFCHASINGRABBITS"
    key = "47185/MERCURY"
    ct = c.encode(plaintext, key)
    out = c.crack(ct, scorer, top=5, rng=rng, timeout=60.0, keyword="MERCURY")
    assert out, "expected at least one candidate"
    assert out[0].plaintext.upper().replace(" ", "") == plaintext
