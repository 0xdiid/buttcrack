"""Condi: validated against the bionspot 'CONDI Ciphers' worked example.

Crack is best-effort only (self-keying offset chain makes simple hill-climbing
unreliable), so no crack test is asserted here.
"""

from __future__ import annotations

from buttcrack.ciphers.condi import Condi, keyed_alphabet
from buttcrack.text import only_letters

# A longer plaintext for the round-trip exercise.
PT = (
    "the analysis routines need a healthy stretch of perfectly ordinary english "
    "prose so that the frequency statistics and quadgram fitness scores can lock "
    "onto the underlying message and recover the original text"
)


def test_condi_keyed_alphabet():
    # Keyword CRYPTOGRAM yields keyed alphabet CRYPTOGAMBDEFHIJKLNQSUVWXZ
    # (bionspot 'CONDI Ciphers' worked example; full 26-letter alphabet, J kept).
    assert keyed_alphabet("CRYPTOGRAM") == "CRYPTOGAMBDEFHIJKLNQSUVWXZ"


def test_condi_bionspot_vector():
    # Source: bionspot.google.site "CONDI Ciphers" worked example.
    # Keyword CRYPTOGRAM -> keyed alphabet CRYPTOGAMBDEFHIJKLNQSUVWXZ, initial
    # offset 10. Plaintext "ON THE":
    #   O(pos6)+10=16->J, offset=6: N(pos19)+6=25->X, offset=19:
    #   T(pos5)+19=24->W, offset=5:  H(pos14)+5=19->N, offset=14:
    #   E(pos11)+14=25->Z. -> ciphertext J X W N Z.
    c = Condi()
    assert only_letters(c.encode("ON THE", "CRYPTOGRAM 10")) == "JXWNZ"


def test_condi_vector_decode():
    # The reverse of the published vector recovers the plaintext.
    c = Condi()
    assert only_letters(c.decode("JXWNZ", "CRYPTOGRAM 10")) == "ONTHE"


def test_condi_roundtrip():
    c = Condi()
    key = "CRYPTOGRAM 10"
    assert only_letters(c.decode(c.encode(PT, key), key)) == only_letters(PT)


def test_condi_roundtrip_alt_separators_and_plain_alphabet():
    c = Condi()
    # '/' and ':' separators are accepted, and an empty keyword uses A-Z.
    assert only_letters(c.decode(c.encode(PT, "SECRET/7"), "SECRET/7")) == only_letters(PT)
    assert only_letters(c.decode(c.encode(PT, "/3"), "/3")) == only_letters(PT)


def test_condi_not_reciprocal():
    # encode != decode for the Condi cipher (self-keying, non-reciprocal).
    c = Condi()
    key = "CRYPTOGRAM 10"
    assert c.encode(PT, key) != c.decode(PT, key)
