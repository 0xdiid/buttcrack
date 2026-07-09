"""Tests for the Compressocrat cipher.

VECTOR source: ACA cipher sheet "COMPRESSOCRAT"
(https://www.cryptogram.org/downloads/aca.info/ciphers/Compressocrat.pdf),
worked example. The sheet builds the keyed alphabet
``YZFRACTIONBDEGHJKLMPQSUVWX`` (keyword ``YZFRACTION``) over the trigram columns
``111``..``332`` (base-3 order; ``333`` unused) and enciphers the plaintext
"The keyword may be shifted to avoid WXYZ always encoding the same digits"
to the ciphertext::

    ROYZP FNGVD NFNXZ MROHD CRPEM YHGIZ SYNXY QBNDN MJJZG OGXFR OYDNT DUSOE NN.

The published letter->digit compression alphabet and the column layout were
verified self-consistently: decoding the published ciphertext with the published
keyed alphabet recovers exactly the published plaintext, and re-encoding that
plaintext reproduces the published ciphertext verbatim.
"""

import random

import pytest

from buttcrack.ciphers.compressocrat import Compressocrat
from buttcrack.scoring import get_scorer

# ACA worked-example plaintext: "The keyword may be shifted to avoid WXYZ always
# encoding the same digits" (letters only).
_VECTOR_PT = "THEKEYWORDMAYBESHIFTEDTOAVOIDWXYZALWAYSENCODINGTHESAMEDIGITS"
_VECTOR_KEY = "YZFRACTION"
_VECTOR_CT = "ROYZPFNGVDNFNXZMROHDCRPEMYHGIZSYNXYQBNDNMJJZGOGXFROYDNTDUSOENN"


def test_vector_aca_sheet():
    c = Compressocrat()
    # ACA Compressocrat cipher sheet worked example (keyword YZFRACTION).
    assert c.encode(_VECTOR_PT, _VECTOR_KEY) == _VECTOR_CT


def test_vector_decode():
    # Decryption recovers the exact ACA plaintext from the published ciphertext.
    c = Compressocrat()
    assert c.decode(_VECTOR_CT, _VECTOR_KEY) == _VECTOR_PT


def test_round_trip():
    # decode(encode(msg)) recovers the prepared plaintext (letters only, upper).
    c = Compressocrat()
    key = "VICTORY"
    msg = "Attack at dawn, jolly good show old chap! Quick zephyrs blow."
    prepared = "".join(ch for ch in msg.upper() if "A" <= ch <= "Z")
    ct = c.encode(msg, key)
    assert c.decode(ct, key) == prepared


def test_round_trip_unkeyed():
    # Empty key uses the straight A-Z alphabet and still round-trips.
    c = Compressocrat()
    msg = "THEFIVEBOXINGWIZARDSJUMPQUICKLY"
    assert c.decode(c.encode(msg, ""), "") == msg


@pytest.mark.slow
def test_crack_recovers():
    c = Compressocrat()
    scorer = get_scorer()
    # ~135 letters of varied prose (ACA recommends 110-150 plaintext letters).
    pt = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENRUNSBACKACROSSTHEFIELDWHILE"
        "THEOLDFARMERWATCHESFROMHISPORCHSIPPINGCOFFEEINTHEEARLYMORNINGLIGHT"
    )
    ct = c.encode(pt, "NIGHTOWL")
    res = c.crack(ct, scorer, rng=random.Random(11), timeout=180)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == pt
