"""Blind two-phase ADFGX/ADFGVX recovery (transposition by digraph-IoC, then square)."""

import random

import pytest

from buttcrack.ciphers import _fractionation as frac
from buttcrack.ciphers.adfgvx import ADFGVX
from buttcrack.ciphers.adfgx import ADFGX
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PLAIN = only_letters(
    "WEHOLDTHESETRUTHSTOBESELFEVIDENTTHATALLMENARECREATEDEQUALTHATTHEYAREENDOWED"
    "BYTHEIRCREATORWITHCERTAINUNALIENABLERIGHTSTHATAMONGTHESEARELIFELIBERTYANDTHE"
    "PURSUITOFHAPPINESSTHATTOSECURETHESERIGHTSGOVERNMENTSAREINSTITUTEDAMONGMEN"
).replace("J", "I")


def test_digraph_ioc_is_mapping_independent_and_peaks_on_true_order():
    cipher = ADFGVX()
    width = 7
    order = list(range(width))
    random.Random(2).shuffle(order)
    ct = cipher.encode(PLAIN, "KEYWORDSQUAREXMPLZBCFHJ/" + ",".join(map(str, order)))
    stream = "".join(c for c in ct if c in "ADFGVX")
    true_ioc = frac.digraph_ioc(frac.untranspose(stream, order))
    # a wrong column order flattens the digraph distribution
    wrong = order[::-1]
    if wrong != order:
        assert true_ioc > frac.digraph_ioc(frac.untranspose(stream, wrong))
    # mapping-independent: relabelling the symbols doesn't change the IoC
    relabel = dict(zip("ADFGVX", "XVGFDA", strict=True))
    relabelled = "".join(relabel[c] for c in stream)
    assert abs(frac.digraph_ioc(relabelled) - frac.digraph_ioc(stream)) < 1e-12


@pytest.mark.slow
def test_adfgvx_blind_two_phase_recovers():
    cipher = ADFGVX()
    width = 9
    order = list(range(width))
    random.Random(5).shuffle(order)
    ct = cipher.encode(PLAIN, "KEYWORDSQUAREXMPLZBCFHJ/" + ",".join(map(str, order)))
    res = cipher.crack(ct, get_scorer(), width=width, timeout=45, rng=random.Random(1))
    assert res, "no candidate"
    assert only_letters(res[0].plaintext) == PLAIN
    assert res[0].meta["method"] == "two-phase-anneal"


@pytest.mark.slow
def test_adfgx_blind_two_phase_recovers():
    cipher = ADFGX()
    width = 8
    order = list(range(width))
    random.Random(4).shuffle(order)
    ct = cipher.encode(PLAIN, "SECURITYABDFGHKLMNOPQVWXZ/" + ",".join(map(str, order)))
    res = cipher.crack(ct, get_scorer(), width=width, timeout=45, rng=random.Random(1))
    assert res, "no candidate"
    assert only_letters(res[0].plaintext) == PLAIN
