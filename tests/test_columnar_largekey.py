"""Large-key columnar recovery: simulated annealing past the brute-force wall."""

import random

import pytest

from buttcrack.ciphers.columnar import BRUTE_MAX_WIDTH, Columnar, _encode_letters
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PLAIN = only_letters(
    "WEHOLDTHESETRUTHSTOBESELFEVIDENTTHATALLMENARECREATEDEQUALTHATTHEYAREENDOWED"
    "BYTHEIRCREATORWITHCERTAINUNALIENABLERIGHTSTHATAMONGTHESEARELIFELIBERTYANDTHE"
    "PURSUITOFHAPPINESSTHATTOSECURETHESERIGHTSGOVERNMENTSAREINSTITUTEDAMONGMEN"
)


def test_default_small_width_is_exact_brute_force():
    """Unchanged path: small widths solve exactly and aren't labelled annealing."""
    cipher = Columnar()
    ct = _encode_letters(PLAIN, [2, 0, 3, 1])
    res = cipher.crack(ct, get_scorer(), timeout=10, rng=random.Random(0))
    assert only_letters(res[0].plaintext) == PLAIN
    assert res[0].meta.get("method") != "annealing"  # brute force


@pytest.mark.slow
def test_anneal_recovers_wide_key():
    """A width-13 key is far past the 8! brute-force ceiling; SA still recovers it."""
    width = 13
    assert width > BRUTE_MAX_WIDTH
    order = list(range(width))
    random.Random(99).shuffle(order)
    cipher = Columnar()
    ct = _encode_letters(PLAIN, order)
    res = cipher.crack(ct, get_scorer(), width=width, timeout=30, rng=random.Random(1))
    assert res, "no candidate"
    assert only_letters(res[0].plaintext) == PLAIN
    assert res[0].meta["method"] == "annealing"
    # the reported key must round-trip through decode
    assert only_letters(cipher.decode(ct, res[0].key)) == PLAIN
