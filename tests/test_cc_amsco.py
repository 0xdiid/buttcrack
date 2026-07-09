"""Tests for the AMSCO incomplete columnar transposition cipher."""

from __future__ import annotations

import pytest

from buttcrack.ciphers.amsco import Amsco
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def test_vector_dcode():
    """Published dCode.fr AMSCO vector: DCODEAMSCO, key 2,1,3, start 1."""
    amsco = Amsco()
    assert amsco.encode("DCODEAMSCO", "2,1,3:1") == "COMDEAODSC"


@pytest.mark.slow
def test_vector_cryptocrack():
    """Published Thonky/CryptoCrack vector: key 31452, cutting start 2."""
    amsco = Amsco()
    plaintext = "A person who smiles in the face of adversity probably has a scapegoat"
    expected = "EOSNEOSBAAOANWEFAVPRHPEAPHSICEROASGRSMTHFITBSCTOILEADYLYA"
    assert amsco.encode(plaintext, "31452:2") == expected


def test_roundtrip_start_one():
    amsco = Amsco()
    msg = "The quick brown fox jumps over the lazy dog."
    key = "3,1,4,2:1"
    assert amsco.decode(amsco.encode(msg, key), key) == only_letters(msg).upper()


def test_roundtrip_start_two():
    amsco = Amsco()
    msg = "Defend the east wall of the castle at all costs tonight."
    key = "2,4,1,5,3:2"
    assert amsco.decode(amsco.encode(msg, key), key) == only_letters(msg).upper()


def test_roundtrip_keyword():
    amsco = Amsco()
    msg = "Meet me by the old oak tree at midnight and bring the documents."
    key = "ZEBRA:1"
    assert amsco.decode(amsco.encode(msg, key), key) == only_letters(msg).upper()


@pytest.mark.slow
def test_crack_recovers_plaintext():
    amsco = Amsco()
    scorer = get_scorer()
    plaintext = (
        "It was the best of times it was the worst of times it was the age of "
        "wisdom it was the age of foolishness it was the epoch of belief it was "
        "the epoch of incredulity it was the season of light"
    )
    cipher = amsco.encode(plaintext, "31452:2")
    results = amsco.crack(cipher, scorer, top=5)
    assert results, "crack returned no candidates"
    assert only_letters(results[0].plaintext) == only_letters(plaintext).upper()
