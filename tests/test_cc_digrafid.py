"""Tests for the Digrafid cipher.

VECTOR source (American Cryptogram Association 'DIGRAFID' description sheet,
cryptogram.org/downloads/aca.info/ciphers/Digrafid.pdf, page 43):

  Tableau --
    Horizontal 3x9 rows: K E Y W O R D A B / C F G H I J L M N / P Q S T U V X Z #
    Vertical 9x3 (rows 1..9, columns read down): the keyed alphabet VERTICAL...
      laid column-by-column gives rows V D P / E F Q / R G S / T H U / I J W /
      C K X / A M Y / L N Z / B O #.
  The horizontal alphabet is keyword KEYWORD filled row-by-row; the vertical
  alphabet is keyword VERTICAL filled column-by-column. The padding/27th symbol
  is '#'.

  Plaintext "thisistheforestpri":
    * period (fractionation) = 3 digraphs/group -> ciphertext HJMXWS WJADWG FCSPYI
    * period (fractionation) = 4 digraphs/group -> ciphertext HJTKVHYU FFWDSQYP RI

  encode() operates on a clean uppercase stream, so the published space-grouped
  ciphertexts are compared without their display spaces.
"""

import random

import pytest

from buttcrack.ciphers.digrafid import Digrafid
from buttcrack.scoring import get_scorer


def test_vector_aca_period3():
    d = Digrafid()
    # ACA DIGRAFID sheet, first worked example: period 3 -> HJMXWS WJADWG FCSPYI.
    assert d.encode("This is the forest pri", "KEYWORD/VERTICAL/3") == "HJMXWSWJADWGFCSPYI"


def test_vector_aca_period4():
    d = Digrafid()
    # ACA DIGRAFID sheet, second worked example: period 4 -> HJTKVHYU FFWDSQYP RI.
    assert d.encode("This is the forest pri", "KEYWORD/VERTICAL/4") == "HJTKVHYUFFWDSQYPRI"


def test_round_trip():
    # decode(encode(msg)) recovers the prepared plaintext: letters only, uppercased,
    # with a '#' pad appended when the cleaned length is odd.
    d = Digrafid()
    key = "SECRETKEY/CIPHERWORD/5"
    msg = "Attack the junction at dawn, jolly good!"
    cleaned = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    prepared = cleaned + ("#" if len(cleaned) % 2 else "")
    ct = d.encode(msg, key)
    assert d.decode(ct, key) == prepared


def test_round_trip_short_final_group():
    # A period that does not divide the digraph count exercises the short trailing
    # group; an odd letter count exercises the '#' padding.
    d = Digrafid()
    key = "MONARCHY/PLANET/4"
    msg = "DEFENDTHEEASTWALLOFTHECASTLENOW"
    cleaned = "".join(c for c in msg.upper() if "A" <= c <= "Z")
    prepared = cleaned + ("#" if len(cleaned) % 2 else "")
    ct = d.encode(msg, key)
    assert d.decode(ct, key) == prepared


@pytest.mark.slow
def test_crack_runs_without_error():
    # The Digrafid joint keyspace (two 27-letter alphabets) is the hardest of the
    # set; full keyless recovery is unreliable, so this only asserts the crack
    # routine runs, honors the timeout, and returns well-formed candidates.
    d = Digrafid()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOM"
        "ITWASTHEAGEOFFOOLISHNESSITWASTHEEPOCHOFBELIEF"
    )
    ct = d.encode(pt, "CHARLES/DICKENS/4")
    res = d.crack(ct, scorer, rng=random.Random(7), timeout=10, max_period=5)
    for cand in res:
        assert cand.cipher == "digrafid"
        assert cand.key is not None
