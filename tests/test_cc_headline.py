"""Tests for the Headline cipher (ACA "HEADLINES": K3 keyed substitution)."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.headline import Headline
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

# Authoritative vector: ACA "HEADLINES" cipher description PDF
# (https://www.cryptogram.org/downloads/aca.info/ciphers/Headlines.pdf).
# Worked example in that PDF:
#   Hat = APOTHECARY, Key = CHEMIST, Setting = DRUGS.
#   The keyed alphabet of CHEMIST (CHEMISTABDFGJKLNOPQRUVWXYZ) is written into a
#   block 10 columns wide (= len Hat); reading the columns top-to-bottom in the
#   alphabetical-rank order of APOTHECARY (1 7 6 9 5 4 3 2 8 10) gives the mixed
#   alphabet CFUAPTOSNZILYEJWHGVBQMKXDR. Each of the five headlines uses a cipher
#   row = that mixed alphabet rotated to start at the n-th letter of DRUGS.
#
# Headline 1 (setting letter D): the PDF lists
#   pt: "Bush Signs Intelligence Overhaul Legislation"
#   CT: "*GCTJ TNWOT NOALZZNWLODL PHLXJFCZ ZLWNTZFANPO"
# Verified self-consistent by encoding all five headlines and matching the
# PDF ciphertext exactly (proper-noun asterisks and layout stripped).
VECTOR_PLAINTEXT = only_letters("Bush Signs Intelligence Overhaul Legislation")
VECTOR_KEY = "APOTHECARY/CHEMIST/D"
VECTOR_CIPHERTEXT = only_letters("GCTJ TNWOT NOALZZNWLODL PHLXJFCZ ZLWNTZFANPO")

# Headline 3 of the same example uses the 3rd DRUGS letter (U); the key form
# "DRUGS:3" selects that setting.
VECTOR3_PLAINTEXT = only_letters("Pfizer Painkiller may pose increased cardiovascular risk")
VECTOR3_KEY = "APOTHECARY/CHEMIST/DRUGS:3"
VECTOR3_CIPHERTEXT = only_letters("OAYLWF OTYIDYEEWF XTJ ONZW YIUFWTZWC UTFCYNQTZUPETF FYZD")


def test_vector_encode():
    cipher = Headline()
    assert cipher.encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_vector_decode():
    cipher = Headline()
    assert cipher.decode(VECTOR_CIPHERTEXT, VECTOR_KEY) == VECTOR_PLAINTEXT


def test_vector_setting_index():
    cipher = Headline()
    assert cipher.encode(VECTOR3_PLAINTEXT, VECTOR3_KEY) == VECTOR3_CIPHERTEXT
    assert cipher.decode(VECTOR3_CIPHERTEXT, VECTOR3_KEY) == VECTOR3_PLAINTEXT


def test_round_trip():
    cipher = Headline()
    msg = "MEETMEATTHEOLDMILLATMIDNIGHTANDBRINGTHESEALEDDOCUMENTS"
    key = "FORTRESS/SENTINEL/W"
    assert cipher.decode(cipher.encode(msg, key), key) == msg


def test_round_trip_setting_word_default_index():
    cipher = Headline()
    msg = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILENOBODYWASLOOKING"
    # Setting word with no index defaults to its first letter (B).
    key = "MONARCHY/CADENCE/BRAVO"
    assert cipher.decode(cipher.encode(msg, key), key) == msg


@pytest.mark.slow
def test_crack_recovers_plaintext():
    cipher = Headline()
    # A single headline at one setting is a monoalphabetic substitution, so the
    # keyless crack recovers the plaintext (not the keywords) via hill climbing.
    plaintext = only_letters(
        "THE GENERAL ASSEMBLY VOTED TODAY TO APPROVE THE NEW BUDGET "
        "RESOLUTION AFTER A LENGTHY DEBATE THAT STRETCHED INTO THE EVENING"
    )
    key = "APOTHECARY/CHEMIST/DRUGS:5"
    ct = cipher.encode(plaintext, key)
    best = cipher.crack(ct, get_scorer(), rng=random.Random(7), timeout=40)
    assert best, "crack returned no candidates"
    assert only_letters(best[0].plaintext) == plaintext
