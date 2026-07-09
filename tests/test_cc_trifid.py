"""Tests for the Trifid cipher.

Vector sources:
  * ACA official 'TRIFID' cipher sheet (cryptogram.org/downloads/aca.info/
    ciphers/Trifid.pdf): keyword EXTRAORDINARY, period 10, 27th symbol '#'.
    Plaintext 'trifidsarefractionatedciphers' -> 'EYMXVUCRYYYYEAYVYOVVXITDPATHE'
    (published with spacing 'EYMXV UCRYY YYEAY VYOVV XITDP ATHE').
  * Boxentriq 'Trifid Cipher' worked example (boxentriq.com/ciphers/
    trifid-cipher): keyword CRYPTOGRAPHY, period 5, 27th symbol '+'.
    Plaintext 'FELIX' -> 'ILASF'.

Note: Trifid has NO I/J merge; all 26 letters live in the cube alongside a 27th
symbol. encode/decode operate on a clean uppercase stream, so round-trip recovers
the uppercased letters-only plaintext exactly (no padding is added).
"""

from buttcrack.ciphers.trifid import Trifid


def test_vector_aca():
    t = Trifid()
    ct = t.encode("trifidsarefractionatedciphers", "EXTRAORDINARY/10")
    assert ct == "EYMXVUCRYYYYEAYVYOVVXITDPATHE"


def test_vector_boxentriq():
    t = Trifid()
    ct = t.encode("FELIX", "CRYPTOGRAPHY/5/+")
    assert ct == "ILASF"


def test_roundtrip():
    t = Trifid()
    key = "SECRETKEYWORD/7"
    msg = "Attack the eastern gate at dawn, before they wake!"
    # Prepared plaintext is uppercase letters only (no I/J merge, no padding).
    prepared = "ATTACKTHEEASTERNGATEATDAWNBEFORETHEYWAKE"
    assert t.decode(t.encode(msg, key), key) == prepared


def test_roundtrip_default_period():
    # A bare keyword defaults to period 5 and '#' as the 27th symbol.
    t = Trifid()
    key = "ZEBRA"
    prepared = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    assert t.decode(t.encode("The quick brown fox jumps over the lazy dog", key), key) == prepared
