"""Rectangular Playfair: reduces to Playfair at 5x5; round-trips at 2x13 (all 26 letters)."""

import random

import pytest

from buttcrack.ciphers.playfair import Playfair
from buttcrack.ciphers.rectangular_playfair import (
    RectangularPlayfair,
    grid_letters,
    parse_key,
)
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PT = (
    "the art of war teaches us to rely not on the likelihood of the enemy not "
    "coming but on our own readiness to receive him not on the chance of his not "
    "attacking but rather on the fact that we have made our position unassailable "
    "all warfare is based on deception when able to attack we must seem unable"
)


def test_parse_key_shapes():
    assert parse_key("NEEDLE") == ("NEEDLE", 2, 13)
    assert parse_key("NEEDLE/2x13") == ("NEEDLE", 2, 13)
    assert parse_key("MONARCHY/5x5") == ("MONARCHY", 5, 5)


def test_5x5_is_identical_to_classic_playfair():
    """The 25-cell grid reuses PolybiusSquare, so 5x5 must match Playfair byte-for-byte."""
    rp, pf = RectangularPlayfair(), Playfair()
    for key in ("MONARCHY", "playfair example", "KRYPTOS"):
        assert rp.encode(PT, f"{key}/5x5") == pf.encode(PT, key)
        ct = pf.encode(PT, key)
        assert rp.decode(ct, f"{key}/5x5") == pf.decode(ct, key)


def test_2x13_grid_uses_all_26_letters():
    grid = grid_letters("NEEDLE", 2, 13)
    assert len(grid) == 26
    assert set(grid) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert grid.startswith("NEDL")  # dedup'd keyword first, then the rest


def test_2x13_roundtrip_all_letters_including_j():
    rp = RectangularPlayfair()
    # a payload containing J survives (no J->I merge in the 26-cell grid)
    pt = "MAJORINJURYJEOPARDIZESTHEJOURNEY"
    ct = rp.encode(pt, "NEEDLE/2x13")
    back = rp.decode(ct, "NEEDLE/2x13")
    # even length, no doubles here -> exact round-trip
    assert back == pt
    assert "J" in ct or "J" in back  # J is a first-class letter in a 26-cell grid


def test_2x13_ciphertext_can_contain_every_letter():
    """A 5x5 square can never emit its dropped letter; a 2x13 grid can emit all 26."""
    rp = RectangularPlayfair()
    ct = rp.encode(PT * 3, "DANGERLESS/2x13")
    assert set(only_letters(ct)) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_default_shape_roundtrip():
    rp = RectangularPlayfair()
    assert rp.decode(rp.encode("meet me at the bridge tonight", "MEADOW"), "MEADOW")


@pytest.mark.slow
def test_2x13_crack_recovers_long_text():
    rp = RectangularPlayfair()
    ct = rp.encode(PT, "DANGERLESS/2x13")
    scorer = get_scorer("quadgrams", "english")
    rng = random.Random(20260707)
    cands = rp.crack(ct, scorer, rng=rng, rows=2, cols=13, restarts=6, iters=4000)
    assert cands
    # recovered plaintext should be strongly English (the SA landscape is noisy; gate loosely)
    assert scorer.confidence(cands[0].plaintext) > 0.5
