"""Tests for the Quagmire III cipher (K3: one keyed alphabet, both sides)."""

from __future__ import annotations

from buttcrack.ciphers.quagmire3 import QuagmireIII

# Published vector from the ACA cipher description PDF "QUAGMIRE III"
# (cryptogram.org/downloads/aca.info/ciphers/QuagmireIII.pdf): keyed alphabet
# from AUTOMOBILE -> AUTOMBILECDFGHJKNPQRSVWXYZ, indicator HIGHWAY (period 7),
# alignment letter A. Verified computationally column-by-column.
VECTOR_PLAINTEXT = "THESAMEKEYEDALPHABETISUSEDFORPLAINANDCIPHERALPHABETS"
VECTOR_KEY = "AUTOMOBILE/HIGHWAY"
VECTOR_CIPHERTEXT = "KRSLWMITJDVIABMRGQMTMLLIVIFUIXRHTNYONVRHHIIIRMCAOVEI"


def test_vector_encode():
    cipher = QuagmireIII()
    assert cipher.encode(VECTOR_PLAINTEXT, VECTOR_KEY) == VECTOR_CIPHERTEXT


def test_vector_decode():
    cipher = QuagmireIII()
    assert cipher.decode(VECTOR_CIPHERTEXT, VECTOR_KEY) == VECTOR_PLAINTEXT


def test_round_trip():
    cipher = QuagmireIII()
    msg = "MEETMEATTHEOLDMILLATMIDNIGHTBRINGTHEDOCUMENTSANDSAYNOTHING"
    key = "FORTRESS/SENTINEL"
    assert cipher.decode(cipher.encode(msg, key), key) == msg


def test_round_trip_explicit_alignment():
    cipher = QuagmireIII()
    msg = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGAGAINANDAGAINFORLUCK"
    key = "MONARCHY/CADENCE/E"  # explicit alignment letter E
    assert cipher.decode(cipher.encode(msg, key), key) == msg


def test_default_alignment_is_keyed_alphabet_first_letter():
    """The alignment letter defaults to the keyed alphabet's FIRST letter (a plain
    Vigenere in the keyed alphabet), NOT the straight-alphabet 'A'. For a keyword
    that does not start with 'A' the two differ, which pins the default. The
    AUTOMOBILE vector above cannot catch a regression here: there header[0] == 'A',
    so both conventions coincide. (Regression guard: an earlier hardcoded 'A'
    default silently returned garbage for every keyed alphabet not starting 'A'.)
    """
    cipher = QuagmireIII()
    msg = "MEETATDAWNBRINGTHELANTERNANDSAYNOTHINGTOTHEOTHERS"
    key = "MONARCHY/SENTINEL"  # keyed alphabet starts with 'M', not 'A'
    ct = cipher.encode(msg, key)  # encodes under the default alignment (= 'M')
    assert cipher.decode(ct, key) == msg  # round-trips under the default
    # default == aligning on the keyed alphabet's first letter ...
    assert cipher.decode(ct, key) == cipher.decode(ct, key + "/M")
    # ... and differs from the straight-'A' alignment, which is still available
    assert cipher.decode(ct, key) != cipher.decode(ct, key + "/A")


# NOTE: no crack test. crack() is implemented as a best-effort keyless search
# (period detection by per-column IoC, then simulated annealing over the shared
# keyed alphabet + per-column rotations), but jointly recovering a 26-letter
# keyed permutation and the per-column rotations from a single short ACA-length
# message is not reliable (it converges to ~20-30% letter recovery, well below a
# clean solve). Per the task guidance, a crack test is added only when crack
# actually recovers, so none is included here.
