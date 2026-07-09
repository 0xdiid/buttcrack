"""Blind two-stream additive split (Reddy-Knight). Marked slow: the beam decode is
a few seconds and wants ~150 letters, so it is excluded from the fast suite."""

import random

import pytest

from buttcrack.twostream import encode, metric, split

A = (
    "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEINHERLEDGER"
    "WHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHERISINGSUNOUTSIDE"
)
B = (
    "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGEACHTREEFORRIPEFRUIT"
    "ANDPRUNESTHEDEADBRANCHESWITHACURVEDKNIFEHEKEEPSSHARPENEDONAWHETSTONEBYTHEGARDENGATE"
)


def _longest_span(recovered: str, original: str) -> int:
    for length in range(min(len(recovered), len(original)), 3, -1):
        for i in range(len(original) - length + 1):
            if original[i : i + length] in recovered:
                return length
    return 0


@pytest.mark.slow
def test_split_recovers_a_verbatim_span_of_an_original():
    n = 130
    a, b = A[:n], B[:n]
    ct = encode(a, b)
    r = split(ct, beam=300)
    span = max(
        _longest_span(r["stream_a"], a),
        _longest_span(r["stream_a"], b),
        _longest_span(r["stream_b"], a),
        _longest_span(r["stream_b"], b),
    )
    assert span >= 12


@pytest.mark.slow
def test_metric_separates_real_sum_from_shuffle():
    n = 130
    ct = encode(A[:n], B[:n])
    real = metric(ct)
    rng = random.Random(0)
    shuf = list(ct)
    rng.shuffle(shuf)
    assert real > metric("".join(shuf))
