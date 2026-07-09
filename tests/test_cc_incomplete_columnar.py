"""Tests for the incomplete columnar transposition cipher."""

from __future__ import annotations

import pytest

from buttcrack.ciphers.incomplete_columnar import IncompleteColumnar
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters


def _ungroup(grouped: str) -> str:
    """Strip the group-by-5 spacing from a published ciphertext."""
    return only_letters(grouped)


def test_vector():
    """Published CryptoCrack User Guide vector (key REALITY, incomplete grid)."""
    cipher = IncompleteColumnar()
    plaintext = "EXPERIENCE IS SOMETHING YOU DON'T GET UNTIL JUST AFTER YOU NEED IT"
    key = "REALITY"
    expected = _ungroup("PETUT URDXC EOEJE ERSIO NTOTE IHDUS YIENM YGLTE ISNNT AUEOG TIFN")
    assert cipher.encode(plaintext, key) == expected


def test_vector_numeric_key():
    """The same vector via the equivalent 0-based read order key."""
    cipher = IncompleteColumnar()
    plaintext = "EXPERIENCE IS SOMETHING YOU DON'T GET UNTIL JUST AFTER YOU NEED IT"
    expected = _ungroup("PETUT URDXC EOEJE ERSIO NTOTE IHDUS YIENM YGLTE ISNNT AUEOG TIFN")
    # REALITY ranks to physical read order 2,1,4,3,0,5,6.
    assert cipher.encode(plaintext, "2,1,4,3,0,5,6") == expected


def test_round_trip():
    cipher = IncompleteColumnar()
    msg = "Defend the east wall of the castle at all costs, hold the line!"
    key = "MONARCH"
    assert cipher.decode(cipher.encode(msg, key), key) == only_letters(msg).upper()


def test_round_trip_various_lengths():
    """Round-trip across lengths that exercise long/short column boundaries."""
    cipher = IncompleteColumnar()
    key = "REALITY"
    base = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGRUNSAWAYINTOTHEFOREST"
    for end in range(8, len(base) + 1):
        msg = base[:end]
        assert cipher.decode(cipher.encode(msg, key), key) == msg


@pytest.mark.slow
def test_crack_recovers_english():
    cipher = IncompleteColumnar()
    scorer = get_scorer()
    plaintext = (
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THEN RUNS AWAY INTO "
        "THE FOREST WHERE IT FINDS SHELTER AMONG THE TALL TREES AND WAITS FOR "
        "THE COMING OF THE NIGHT WHEN ALL IS QUIET AND STILL"
    )
    expected = only_letters(plaintext).upper()
    ciphertext = cipher.encode(plaintext, "MONARCH")
    candidates = cipher.crack(ciphertext, scorer, top=5, timeout=30)
    assert candidates
    assert candidates[0].plaintext == expected
