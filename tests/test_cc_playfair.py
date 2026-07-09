"""Playfair: validated against the Wikipedia vector; SA crack on a seeded run."""

import random

import pytest

from buttcrack.ciphers.playfair import Playfair, _prepare
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PT = (
    "the art of war teaches us to rely not on the likelihood of the enemy not "
    "coming but on our own readiness to receive him not on the chance of his not "
    "attacking but rather on the fact that we have made our position unassailable "
    "all warfare is based on deception when able to attack we must seem unable"
)


def test_playfair_wikipedia_vector():
    pf = Playfair()
    assert pf.encode("hide the gold in the tree stump", "playfair example") == (
        "BMODZBXDNABEKUDMUIXMMOUVIF"
    )
    assert pf.decode("BMODZBXDNABEKUDMUIXMMOUVIF", "playfair example") == (
        "HIDETHEGOLDINTHETREXESTUMP"
    )


def test_playfair_roundtrip_prepared():
    pf = Playfair()
    prepared = "".join(a + b for a, b in _prepare(only_letters("meet me at the bridge")))
    assert pf.decode(pf.encode("meet me at the bridge", "KEYWORD"), "KEYWORD") == prepared


@pytest.mark.slow
def test_playfair_crack_recovers_long_text():
    # Playfair cracking is stochastic (simulated annealing); seed pins it.
    pf = Playfair()
    ct = pf.encode(PT, "MONARCHY")
    prepared = "".join(a + b for a, b in _prepare(only_letters(PT)))
    best = pf.crack(ct, get_scorer(), rng=random.Random(3), timeout=40)[0]
    matches = sum(a == b for a, b in zip(best.plaintext, prepared, strict=False))
    assert matches / len(prepared) >= 0.9
