"""Interrupted Key cipher tests.

Validated against the CryptoCrack worked example (primary, authoritative — this
project mirrors CryptoCrack) and cross-checked against an independent ACA-style
example from The Black Chamber. Both are Vigenere with the keyword pointer reset
at every word division; on a clean (space-free) letter stream the word lengths
are supplied as explicit group lengths via the ``/G=...`` key rule.

Crack is best-effort only (interruptor mode has no period to exploit and needs a
supplied wordlist), so no crack recovery is asserted here.
"""

from __future__ import annotations

from buttcrack.ciphers.interrupted_key import InterruptedKey
from buttcrack.text import only_letters

# Source: CryptoCrack user guide, Interrupted Key page
# https://sites.google.com/site/cryptocrackprogram/user-guide/cipher-types/substitution/interrupted-key
# Type Vigenere, keyword TWAIN, key restarted at each word division.
# Plaintext "If you tell the truth, you dont have to remember anything"
# (word lengths 2,3,4,3,5,3,4,4,2,8,8) -> ciphertext below.
CC_PT = "If you tell the truth, you dont have to remember anything"
CC_KEY = "TWAIN/G=2,3,4,3,5,3,4,4,2,8,8"
CC_CT = "BBRKUMALTMDEMNUBURKUWKNBAWVMMKKAMMZUARTJYBUBJG"

# Cross-check source: The Black Chamber, "Autokey, Running Key, Interrupted Key"
# https://theblackchamber552383191.wordpress.com/2020/11/17/autokey-running-key-interrupted-key/
# Vigenere, keyword GIVEUP, reset at each word division.
BC_PT = "i have done worse than some people and better than others"
BC_KEY = "GIVEUP/G=1,4,4,5,4,4,6,3,6,4,6"
BC_CT = "ONIQIJWIICWMWYZPVRYWHIVMJTFTGVYHMOXYGZPVRUBCILH"

# A longer message for round-trip exercises across families/modes.
PT = (
    "the analysis routines need a healthy stretch of perfectly ordinary english "
    "prose so that the scoring model can lock onto the underlying message and "
    "recover the original text without any trouble whatsoever today"
)


def test_interrupted_key_cryptocrack_vector():
    c = InterruptedKey()
    assert only_letters(c.encode(CC_PT, CC_KEY)) == CC_CT


def test_interrupted_key_cryptocrack_vector_decode():
    c = InterruptedKey()
    assert only_letters(c.decode(CC_CT, CC_KEY)) == only_letters(CC_PT)


def test_interrupted_key_blackchamber_vector():
    c = InterruptedKey()
    assert only_letters(c.encode(BC_PT, BC_KEY)) == BC_CT


def test_interrupted_key_roundtrip_group_mode():
    c = InterruptedKey()
    key = "SECRET/G=4,3,9,7,5,2,8,6"
    assert only_letters(c.decode(c.encode(PT, key), key)) == only_letters(PT)


def test_interrupted_key_roundtrip_interruptor_mode():
    c = InterruptedKey()
    for key in (
        "TWAIN/I=E",
        "BEAU:SECRET/I=T",
        "VAR:CODE/I=A",
        "PORTA:KEYWORD/I=O",
    ):
        assert only_letters(c.decode(c.encode(PT, key), key)) == only_letters(PT)


def test_interrupted_key_roundtrip_plain_no_interruption():
    c = InterruptedKey()
    # With no '/' rule the cipher degenerates to a plain periodic one.
    for key in ("TWAIN", "BEAU:HELLO", "VAR:WORLD", "PORTA:KEYWORD"):
        assert only_letters(c.decode(c.encode(PT, key), key)) == only_letters(PT)


def test_interrupted_key_vigenere_not_reciprocal():
    c = InterruptedKey()
    key = "TWAIN/I=E"
    assert c.encode(PT, key) != c.decode(PT, key)
