"""Tri-Square (Three-Square) cipher tests.

Authoritative vector source: dCode 'Three-Squares Cipher'
(https://www.dcode.fr/three-squares-cipher). dCode's worked example uses three
keyed 5x5 grids built from the keywords ONE / TWO / THREE over its default
25-letter alphabet (J kept, Z dropped: "ABCDEFGHIJKLMNOPQRSTUVWXY") and gives
the ciphertext trigraph stream "TKDGNVSAFRAV" for the plaintext "MESSAGE".

The dCode page states the cell coordinates for the digraph ME explicitly:
M is at grid1 (row3, col5), E is at grid2 (row2, col3), and the grid3 (middle)
intersection of M's row and E's column is K at (row3, col3) -- exactly
reproduced by the alphabet/keywords above. Encryption is homophonic (the first
and third trigraph letters are free choices), so the vector is validated by
DECODE: decode("TKDGNVSAFRAV") == "MESSAGE" (+ one pad letter).
"""

from __future__ import annotations

import random

from buttcrack.ciphers.tri_square import TriSquare

# dCode default alphabet for the reference grids: keep J, drop Z.
DCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXY"


def test_tri_square_dcode_vector():
    # dCode 'Three-Squares Cipher': grids ONE/TWO/THREE, ct "TKDGNVSAFRAV"
    # decrypts to "MESSAGE" (7 letters -> one trailing pad letter on the 8th).
    ts = TriSquare(alphabet=DCODE_ALPHABET)
    plain = ts.decode("TKDGNVSAFRAV", "ONE/TWO/THREE")
    assert plain[:7] == "MESSAGE"
    # The recovered stream is the 8 letters of 4 digraphs; the 8th is the pad.
    assert len(plain) == 8


def test_tri_square_roundtrip_default_alphabet():
    # Default alphabet is the ACA/CryptoCrack 25-letter square (J->I, Z kept).
    ts = TriSquare()
    key = "SECRET/CIPHER/PUZZLE"
    # Even-length, J-free plaintext so the prepared stream == recovered stream.
    pt = "THEQUICKBROWNFOX"
    ct = ts.encode(pt, key)
    assert len(ct) == 3 * (len(pt) // 2)  # trigraph per digraph: 3:2 expansion
    assert ts.decode(ct, key) == pt


def test_tri_square_roundtrip_homophonic_and_padding():
    # With an rng the homophonic C1/C3 letters are random, yet decode is unique.
    # Odd length + a 'J' exercise the X-pad and J->I merge of the default square.
    ts = TriSquare()
    key = "OSCAR/WILDE/QUOTE"
    pt = "THE QUICK BROWN FOX JUMPS"
    prepared = ts._prepare(pt)
    if len(prepared) % 2:
        prepared += "X"
    ct = ts.encode(pt, key, rng=random.Random(11))
    assert ts.decode(ct, key) == prepared
    # J->I merge: 'JUMPS' became 'IUMPS' in the prepared stream.
    assert "J" not in prepared and "IUMPS" in prepared
