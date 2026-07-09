"""Tests for the Myszkowski transposition cipher."""

from __future__ import annotations

import pytest

from buttcrack.ciphers.myszkowski import Myszkowski
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def test_published_vector() -> None:
    """Crypto Corner / Wikipedia TOMATO vector (letters only, uppercased)."""
    cipher = Myszkowski()
    plaintext = "THE TOMATO IS A PLANT IN THE NIGHTSHADE FAMILY"
    key = "TOMATO"
    expected = only_letters("TINES AXEOA HTFXH MTALI TIHAE IYXTO ASPTN NGHDM LX")
    assert cipher.encode(plaintext, key) == expected


def test_published_decrypt_vector() -> None:
    """Wikipedia POTATO worked decrypt vector."""
    cipher = Myszkowski()
    ciphertext = only_letters("ARESA SXOOS ITIHA EIYEL XPENG DLLTT AEHNT HFMAW XX")
    decoded = cipher.decode(ciphertext, "POTATO")
    assert decoded == only_letters("POTATOES ARE IN THE NIGHTSHADE FAMILY AS WELL")


def test_round_trip() -> None:
    cipher = Myszkowski()
    msg = "We hold these truths to be self evident, that all are equal."
    key = "BANANA"
    encoded = cipher.encode(msg, key)
    assert cipher.decode(encoded, key) == only_letters(msg).upper()


@pytest.mark.slow
def test_crack_recovers_plaintext() -> None:
    cipher = Myszkowski()
    scorer = get_scorer()
    plaintext = (
        "IT WAS THE BEST OF TIMES IT WAS THE WORST OF TIMES IT WAS THE AGE OF "
        "WISDOM IT WAS THE AGE OF FOOLISHNESS IT WAS THE EPOCH OF BELIEF"
    )
    key = "TOMATO"
    ciphertext = cipher.encode(plaintext, key)
    results = cipher.crack(ciphertext, scorer, top=5)
    assert results
    best = results[0].plaintext
    expected = only_letters(plaintext)
    # Decode of a complete rectangle may carry trailing X padding.
    assert best.rstrip("X") == expected.rstrip("X")
    assert best.startswith(expected[:40])
