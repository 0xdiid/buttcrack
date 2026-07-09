"""The generic simulated-annealing engine cracks a monoalphabetic substitution."""

import random
import string

from buttcrack import search
from buttcrack.scoring import get_scorer

A = string.ascii_uppercase
PLAINTEXT = (
    "THEEXPEDITIONREACHEDTHEHIGHPASSJUSTBEFOREDAWNANDFOUNDTHEANCIENTMARKERSEXACTLY"
    "WHERETHEOLDMAPHADPROMISEDEACHSTONEWASCARVEDWITHTHESAMESPIRALEMBLEMANDWECOPIED"
    "THEMCAREFULLYBYLANTERNLIGHTBEFORETHESUNROSEANDTHEMISTBURNEDAWAYREVEALINGTHE"
    "VALLEYFARBELOWWHERETHELOSTCITYWASSAIDTOLIESILENTBENEATHCENTURIESOFDRIFTINGSAND"
)


def test_anneal_cracks_substitution():
    rng = random.Random(7)
    key = list(A)
    rng.shuffle(key)
    enc = {p: c for p, c in zip(A, key, strict=True)}
    ciphertext = "".join(enc[ch] for ch in PLAINTEXT)
    scorer = get_scorer()

    def score(dec_key: list) -> float:
        table = {c: p for p, c in zip(A, dec_key, strict=True)}
        return scorer.score("".join(table[ch] for ch in ciphertext))

    best, _ = search.anneal(
        init=lambda: search.shuffled(list(A), rng),
        neighbour=search.swap_neighbour,
        score=score,
        rng=rng,
        restarts=6,
    )
    table = {c: p for p, c in zip(A, best, strict=True)}
    recovered = "".join(table[ch] for ch in ciphertext)
    assert recovered == PLAINTEXT
