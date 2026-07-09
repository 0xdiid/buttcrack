"""Tests for the Null (concealment) cipher."""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.null_cipher import NullCipher
from buttcrack.scoring import get_scorer


def test_vector():
    """Published worked example.

    Source: ACA cipher sheet "Null" (cryptogram.org/downloads/aca.info/ciphers/Null.pdf).
    CT: "THE GREAT OLD PUMPERS."  key: middle  ->  pt: HELP, because the middle
    letter of each word (THE/GREAT/OLD/PUMPERS) is H/E/L/P.
    """
    cipher = NullCipher()
    assert cipher.decode("THE GREAT OLD PUMPERS.", "middle") == "HELP"


def test_vector_first_and_last():
    # First-letter and last-letter nulls are the most common ACA forms.
    cipher = NullCipher()
    # First letter of each word spells the message.
    assert cipher.decode("Help Every Lost Person", "first") == "HELP"
    # Last letter of each word spells the message.
    assert cipher.decode("FISH APPLE WELL DEEP", "last") == "HELP"


def test_round_trip():
    cipher = NullCipher()
    msg = "ATTACKATDAWN"
    for key in ("first", "last", "middle", "1", "2", "3", "-1", "-2"):
        ct = cipher.encode(msg, key)
        assert cipher.decode(ct, key) == msg


def test_round_trip_longer():
    cipher = NullCipher()
    msg = "MEETMEBYTHEOLDCLOCKTOWERATMIDNIGHT"
    for key in ("first", "last", "middle"):
        assert cipher.decode(cipher.encode(msg, key), key) == msg


@pytest.mark.slow
def test_crack_recovers():
    """The keyless crack recovers a hidden message and its position key.

    With no key the only unknown is which letter of each word is significant, so
    the crack scores extraction under each candidate position and ranks them.
    """
    cipher = NullCipher()
    scorer = get_scorer()
    msg = (
        "MEETMEATMIDNIGHTBYTHEOLDCLOCKTOWERNEARTHERIVERBRINGTHE"
        "DOCUMENTSANDTELLNOONEOFOURPLANSUNTILWEARESAFELYAWAY"
    )
    ct = cipher.encode(msg, "first")
    results = cipher.crack(ct, scorer, top=3, rng=random.Random(7), timeout=60)
    assert results, "crack returned no candidates"
    assert results[0].plaintext == msg
    assert results[0].key == "first"
