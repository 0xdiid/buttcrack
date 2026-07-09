"""Tests for the Quagmire I cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.quagmire1 import QuagmireI, keyed_alphabet
from buttcrack.scoring import get_scorer


def test_keyed_alphabet():
    # ACA QuagmireI PDF worked example.
    assert keyed_alphabet("SPRINGFEVER") == "SPRINGFEVABCDHJKLMOQTUWXYZ"


def test_vector_aca():
    """Published worked example.

    Source: ACA cipher description 'QUAGMIRE I',
    cryptogram.org/downloads/aca.info/ciphers/QuagmireI.pdf -- verified
    computationally column-by-column.

    Alphabet keyword SPRINGFEVER -> keyed plaintext alphabet
    SPRINGFEVABCDHJKLMOQTUWXYZ; indicator FLOWER (period 6) aligned under A.
    """
    cipher = QuagmireI()
    pt = (
        "The Quag One is a periodic cipher with a keyed plain "
        "alphabet run against a straight cipher alphabet"
    )
    ct = cipher.encode(pt, "SPRINGFEVER/FLOWER")
    expected = (
        "QPMGQ RBUJU YIFDM PYAIF QYYJJ JHJYC JLUUT PIDVW YMFSG "
        "AESDW HIZRB LIRVC FCZPE LBPZY YJJJH WLJJL PUP"
    ).replace(" ", "")
    assert ct == expected


def test_vector_explicit_align():
    # The default alignment is A; spelling it out must be identical.
    cipher = QuagmireI()
    pt = "The Quag One is a periodic cipher"
    assert cipher.encode(pt, "SPRINGFEVER/FLOWER") == cipher.encode(pt, "SPRINGFEVER/FLOWER/A")


def test_round_trip():
    cipher = QuagmireI()
    key = "SPRINGFEVER/FLOWER"
    msg = "The Quag One is a periodic cipher with a keyed plain alphabet"
    prepared = "THEQUAGONEISAPERIODICCIPHERWITHAKEYEDPLAINALPHABET"
    ct = cipher.encode(msg, key)
    assert cipher.decode(ct, key) == prepared


def test_round_trip_other_keys():
    cipher = QuagmireI()
    for kw, ind in (("KEYWORD", "SECRET"), ("PALMER", "STONE"), ("ZEBRAS", "VODKA")):
        key = f"{kw}/{ind}"
        prepared = cipher.decode(cipher.encode("ATTACKATDAWNQUICKLY", key), key)
        assert prepared == "ATTACKATDAWNQUICKLY"


@pytest.mark.slow
def test_crack_recovers():
    """The keyless crack should recover plaintext + period on a long sample."""
    cipher = QuagmireI()
    scorer = get_scorer()
    key = "SPRINGFEVER/FLOWER"  # period 6
    # Quag I columns are short keyed monoalphabets; the crack folds them onto one
    # stream by monogram cross-correlation, which needs enough per-column data to
    # be reliable, so use a long sample (~480 letters, ~80 per column).
    plaintext = (
        "THEQUAGMIREONECIPHERISAPERIODICPOLYALPHABETICSYSTEMTHATUSESAKEYED"
        "PLAINTEXTALPHABETRUNAGAINSTSTRAIGHTCIPHERTEXTALPHABETSANDISWEAKER"
        "THANITLOOKSBECAUSEALLTHECOLUMNSSHARETHESAMEMIXEDALPHABETDIFFERING"
        "ONLYBYANADDITIVESHIFTWHICHCOUPLESTHEMTOGETHERTIGHTLY"
    ) * 2
    ct = cipher.encode(plaintext, key)
    results = cipher.crack(ct, scorer, top=3, rng=random.Random(7), timeout=120)
    assert results, "crack returned no candidates"
    assert results[0].plaintext == plaintext
    assert results[0].meta["period"] == 6
