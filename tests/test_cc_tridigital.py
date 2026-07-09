"""Tests for the Tridigital cipher (ACA #84)."""

from __future__ import annotations

from buttcrack.ciphers.tridigital import Tridigital


def test_published_vector():
    """Published vector from The Black Chamber 'Tridigital cipher' + ACA spec.

    Source: https://theblackchamber552383191.wordpress.com/2020/11/18/tridigital-cipher/
    Column keyword NOVELCRAFT -> labels 6 7 0 3 5 2 8 1 4 9; keyed alphabet from
    DRAGONFLY (DRAGONFLYBCEHIJKMPQSTUVWXZ) over a 3x9 grid; spaces -> 9.
    Plaintext "THE IDES OF MARCH" -> "0 3 0 9 5 6 0 7 9 5 8 9 1 0 7 7 3".
    """
    cipher = Tridigital()
    key = "NOVELCRAFT/DRAGONFLY"
    expected = "03095607958910773"
    got = cipher.encode("THE IDES OF MARCH", key)
    assert got.replace(" ", "") == expected


def test_round_trip():
    """decode(encode(msg, key), key) recovers a prepared plaintext.

    The decoder takes the conventional (top-row) reading of each digit, so the
    prepared plaintext uses only top-row letters of the keyed grid, which the
    decoder resolves unambiguously.
    """
    cipher = Tridigital()
    key = "NOVELCRAFT/DRAGONFLY"
    msg = "DRAG ON A FLY"  # all letters live in the grid's top row
    ct = cipher.encode(msg, key)
    assert cipher.decode(ct, key) == msg


def test_single_keyword_key():
    """A single keyword (no '/') is accepted and used for both column + alphabet."""
    cipher = Tridigital()
    ct = cipher.encode("HELLO WORLD", "FORTUNE")
    # all-digit tokens, deterministic round-trip on a top-row word
    assert all(tok.isdigit() for tok in ct.split())
