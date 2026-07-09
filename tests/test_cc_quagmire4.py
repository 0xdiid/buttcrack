"""Quagmire IV (ACA K4 keying): two keyed alphabets + indicator.

Vector source: ACA cipher description PDF "QUAGMIRE IV"
(cryptogram.org/downloads/aca.info/ciphers/QuagmireIV.pdf). The worked example
uses a keyed plaintext alphabet from SENSORY (-> SENORYABCDFGHIJKLMPQTUVWXZ), a
keyed ciphertext alphabet from PERCEPTION (-> PERCTIONABDFGHJKLMQSUVWXYZ), and
indicator EXTRA (period 5) written under the first plaintext-alphabet letter S.
The five cipher rows are rotations of the keyed ciphertext alphabet starting at
each indicator letter (E, X, T, R, A). Encrypting

    THISONEEMPLOYSTHREEKEYWORDS

yields ciphertext (grouped in the PDF as)

    VBMRF CYISP MPBRR HEICX RREIG DX

i.e. VBMRFCYISPMPBRRHEICXRREIGDX. Verified column-by-column.
"""

from buttcrack.ciphers.quagmire4 import QuagmireIV, keyed_alphabet


def test_quagmire4_keyed_alphabet_construction():
    # The two keyed alphabets used by the ACA worked example.
    assert keyed_alphabet("SENSORY") == "SENORYABCDFGHIJKLMPQTUVWXZ"
    assert keyed_alphabet("PERCEPTION") == "PERCTIONABDFGHJKLMQSUVWXYZ"


def test_quagmire4_aca_vector():
    cs = QuagmireIV()
    plaintext = "THISONEEMPLOYSTHREEKEYWORDS"
    key = "SENSORY/PERCEPTION/EXTRA"
    assert cs.encode(plaintext, key) == "VBMRFCYISPMPBRRHEICXRREIGDX"


def test_quagmire4_decode_vector():
    cs = QuagmireIV()
    ciphertext = "VBMRFCYISPMPBRRHEICXRREIGDX"
    key = "SENSORY/PERCEPTION/EXTRA"
    assert cs.decode(ciphertext, key) == "THISONEEMPLOYSTHREEKEYWORDS"


def test_quagmire4_roundtrip():
    cs = QuagmireIV()
    key = "MONARCHY/COMPUTER/CIPHERKEY"
    msg = "Defend the east wall of the castle at dawn, the enemy approaches now."
    prepared = "DEFENDTHEEASTWALLOFTHECASTLEATDAWNTHEENEMYAPPROACHESNOW"
    assert cs.decode(cs.encode(msg, key), key) == prepared


def test_quagmire4_optional_alignment_letter():
    cs = QuagmireIV()
    # The default alignment is the first plaintext-alphabet letter; passing it
    # explicitly must reproduce the published vector exactly.
    plaintext = "THISONEEMPLOYSTHREEKEYWORDS"
    assert cs.encode(plaintext, "SENSORY/PERCEPTION/EXTRA/S") == "VBMRFCYISPMPBRRHEICXRREIGDX"
