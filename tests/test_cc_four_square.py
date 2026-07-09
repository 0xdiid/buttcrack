"""Four-square cipher: validated against the Wikipedia vector; seeded SA crack.

Vector source: Wikipedia 'Four-square cipher' and practicalcryptography.com.
Uses the 25-letter Q-omitted alphabet (I and J both kept). Key format is
two keywords separated by '/': "TOPRIGHT/BOTTOMLEFT".
"""

import random

import pytest

from buttcrack.ciphers.four_square import ALPHABET, FourSquare, _prepare
from buttcrack.scoring import get_scorer

# A clean, Q-free plaintext keeps prep == the recoverable stream for round-trip.
PT = (
    "the quick brown fox does jump over many a lazy sleeping dog while the "
    "early morning sun rises slowly above the wide green eastern valley and "
    "the river winds its way down toward the distant silver sea below us all "
    "as we watch the light grow brighter over the meadows and the gentle hills"
)


def test_four_square_wikipedia_vector():
    fs = FourSquare()
    # Wikipedia: plaintext "help me obi wan kenobi" with cipher squares keyed
    # EXAMPLE (top-right) and KEYWORD (bottom-left) -> FYGMKYHOBXMFKKKIMD.
    assert fs.encode("help me obi wan kenobi", "EXAMPLE/KEYWORD") == "FYGMKYHOBXMFKKKIMD"
    assert fs.decode("FYGMKYHOBXMFKKKIMD", "EXAMPLE/KEYWORD") == "HELPMEOBIWANKENOBI"


def test_four_square_roundtrip_prepared():
    fs = FourSquare()
    key = "MONARCHY/PALMERSTON"
    # The recoverable stream is the prepared plaintext: uppercased, Q dropped,
    # padded with a trailing X if odd length. PT has no Q and even length here.
    prepared = _prepare(PT)
    assert fs.decode(fs.encode(PT, key), key) == prepared


def test_four_square_prepare_drops_q_and_pads():
    # Q is dropped (not in the 25-letter alphabet); odd length pads with X.
    assert "Q" not in ALPHABET
    assert _prepare("quiz") == "UIZX"  # q dropped -> "uiz" -> pad -> UIZX


@pytest.mark.slow
def test_four_square_crack_recovers_long_text():
    # Cracking is stochastic simulated annealing; seed pins the run.
    fs = FourSquare()
    key = "EXAMPLE/KEYWORD"
    ct = fs.encode(PT, key)
    prepared = _prepare(PT)
    cands = fs.crack(ct, get_scorer(), rng=random.Random(7), timeout=60)
    assert cands, "crack returned no candidates"
    best = cands[0]
    matches = sum(a == b for a, b in zip(best.plaintext, prepared, strict=False))
    assert matches / len(prepared) >= 0.9
