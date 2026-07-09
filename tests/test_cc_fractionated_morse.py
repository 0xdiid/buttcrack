"""Tests for the Fractionated Morse cipher."""

from __future__ import annotations

from buttcrack.ciphers.fractionated_morse import FractionatedMorse


def test_vector_unkeyed():
    # Source: dCode "Fractionated Morse Cipher" (https://www.dcode.fr/fractionated-morse)
    # and asecuritysite.com/encryption/frac. Unkeyed (straight A-Z) table.
    # morse("HELLO WORLD") = ....x.x.-..x.-..x---xx.--x---x.-.x.-..x-.. (42 symbols)
    # -> AGTCDHOTQODTCJ
    cipher = FractionatedMorse()
    assert cipher.encode("HELLO WORLD") == "AGTCDHOTQODTCJ"


def test_vector_keyed_crowded():
    # Source: CryptoCrack user guide, Fractionated Morse page
    # (https://sites.google.com/site/cryptocrackprogram/user-guide/cipher-types/
    #  substitution/fractionated-morse). Keyword CROWDED ->
    # keyed alphabet CROWDEABFGHIJKLMNPQSTUVXYZ; "NOBODY GOES THERE" -> IKUOKUBDZI...
    cipher = FractionatedMorse()
    ct = cipher.encode("NOBODY GOES THERE", "CROWDED")
    assert ct.startswith("IKUOKUBDZI")


def test_keyed_alphabet_build():
    # The keyed alphabet is exactly CROWDEABFGHIJKLMNPQSTUVXYZ for CROWDED.
    from buttcrack.ciphers.fractionated_morse import _keyed_alphabet

    assert _keyed_alphabet("CROWDED") == "CROWDEABFGHIJKLMNPQSTUVXYZ"
    assert _keyed_alphabet("") == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_roundtrip_unkeyed():
    cipher = FractionatedMorse()
    msg = "ATTACK AT DAWN"
    assert cipher.decode(cipher.encode(msg), "") == msg


def test_roundtrip_keyed():
    cipher = FractionatedMorse()
    msg = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    key = "SECRET"
    assert cipher.decode(cipher.encode(msg, key), key) == msg


def test_roundtrip_single_word():
    cipher = FractionatedMorse()
    msg = "CRYPTOGRAPHY"
    key = "FIDDLE"
    assert cipher.decode(cipher.encode(msg, key), key) == msg
