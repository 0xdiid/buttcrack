"""Tests for the ADFGVX cipher (6x6 fractionation + columnar transposition)."""

from __future__ import annotations

from buttcrack.ciphers.adfgvx import ADFGVX

# Wikipedia "ADFGVX cipher" worked example
# (en.wikipedia.org/wiki/ADFGVX_cipher). The 6x6 grid is given explicitly there
# (row by row): N A 1 C 3 H / 8 T B 2 O M / E 5 W R P D / 4 F 6 G 7 I /
# 9 J 0 K L Q / S U V X Y Z, with row/col labels A D F G V X. The transposition
# keyword is PRIVACY. Plaintext "attackat1200am" -> "DGDDDAGDDGAFADDFDADVDVFAADVX".
VECTOR_GRID = "NA1C3H8TB2OME5WRPD4F6G7I9J0KLQSUVXYZ"
VECTOR_KEY = f"{VECTOR_GRID}/PRIVACY"
VECTOR_PLAINTEXT = "attackat1200am"
VECTOR_CIPHERTEXT = "DGDDDAGDDGAFADDFDADVDVFAADVX"


def test_vector():
    """Encode reproduces the published Wikipedia ADFGVX ciphertext exactly."""
    assert ADFGVX().encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_roundtrip_explicit_grid():
    """decode(encode(...)) recovers the prepared (uppercased) plaintext.

    ADFGVX has NO I/J merge (6x6 square holds all 26 letters + digits 0-9), so
    every encodable A-Z0-9 character round-trips with no information loss.
    """
    c = ADFGVX()
    prepared = VECTOR_PLAINTEXT.upper()  # letters + digits, already encodable
    assert c.decode(c.encode(VECTOR_PLAINTEXT, VECTOR_KEY), VECTOR_KEY) == prepared


def test_roundtrip_keyword_square():
    """Round-trip with a keyword-derived 6x6 square and keyword transposition."""
    c = ADFGVX()
    key = "SECRET/GERMAN"
    msg = "MEETMEATDAWN42"
    assert c.decode(c.encode(msg, key), key) == msg


def test_roundtrip_strips_punctuation_and_uppercases():
    """Non-encodable characters (spaces/punctuation) are dropped; rest uppercased."""
    c = ADFGVX()
    key = "PASSWORD/VICTORY"
    msg = "Fall back at 9:99!"
    # prepared = uppercased A-Z0-9 only: "FALLBACKAT999"
    prepared = "FALLBACKAT999"
    assert c.decode(c.encode(msg, key), key) == prepared
