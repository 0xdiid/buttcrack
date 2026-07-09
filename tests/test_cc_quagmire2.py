"""Quagmire II cipher (ACA K2: straight plaintext vs keyed ciphertext alphabet).

VECTOR source: ACA cipher description sheet "QUAGMIRE II"
(cryptogram.org/downloads/aca.info/ciphers/QuagmireII.pdf). Keyed ciphertext
alphabet from keyword SPRINGFEVER -> SPRINGFEVABCDHJKLMOQTUWXYZ; indicator
FLOWER (period 6) aligned under plaintext A. The published worked example:

    plaintext : "In the Quag Two a straight plain alphabet is run against a
                 keyed cipher alphabet"
    ciphertext: JICIC OSLYK ILFVC HEBDX CCORJ IOEWA FMWKK TXBGW HRJIB KEDBJ
                WZABU XWHEH UXOXC U

Key is packed as "ALPHABETKEY/INDICATORKEY" (alignment letter defaults to A).
"""

from buttcrack.ciphers.quagmire2 import QuagmireII


def test_vector_aca():
    q = QuagmireII()
    pt = "In the Quag Two a straight plain alphabet is run against a keyed cipher alphabet"
    expected = (
        "JICIC OSLYK ILFVC HEBDX CCORJ IOEWA FMWKK TXBGW HRJIB KEDBJ WZABU XWHEH UXOXC U"
    ).replace(" ", "")
    assert q.encode(pt, "SPRINGFEVER/FLOWER") == expected


def test_vector_explicit_alignment():
    # Alignment letter A is the default; stating it explicitly is identical.
    q = QuagmireII()
    pt = "In the Quag Two a straight plain alphabet is run against a keyed cipher alphabet"
    assert q.encode(pt, "SPRINGFEVER/FLOWER") == q.encode(pt, "SPRINGFEVER/FLOWER/A")


def test_round_trip():
    q = QuagmireII()
    key = "MYSTERY/CALENDAR"
    msg = "Attack the eastern wall of the castle at dawn, jolly good show!"
    prepared = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    ct = q.encode(msg, key)
    assert q.decode(ct, key) == prepared


# No crack test: keyless recovery of a K2 keyed-alphabet polyalphabetic requires
# jointly searching a 26-letter keyed alphabet and per-column rotations, a hard
# landscape that QuagmireII.crack attempts (period IoC + annealed hill-climb) but
# does not reliably solve within a practical timeout. The crack is best-effort.
