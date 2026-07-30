"""Gauge-normalized, excision and anchored scoring."""

from __future__ import annotations

import random

import pytest

from buttcrack.scoring import (
    GaugeNormalizedScorer,
    anchored_score,
    best_caesar_gauge,
    excision_score,
    get_scorer,
)

ENGLISH = (
    "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTED"
    "ITSTITLEINHERLEDGERWHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREAD"
)


def _caesar(text: str, shift: int) -> str:
    return "".join(chr((ord(c) - 65 + shift) % 26 + 65) for c in text)


# ------------------------------------------------------------- caesar gauge

def test_best_caesar_gauge_recovers_the_applied_shift():
    for applied in (0, 3, 11, 25):
        shifted = _caesar(ENGLISH, applied)
        shift, _ = best_caesar_gauge(shifted)
        # normalize must undo the applied shift: applied + shift ≡ 0 (mod 26)
        assert (applied + shift) % 26 == 0, f"applied={applied} recovered={shift}"


def test_gauge_normalized_scorer_is_shift_invariant():
    g = GaugeNormalizedScorer()
    base = get_scorer()
    plain = g.score(ENGLISH)
    for applied in (5, 13, 21):
        shifted = _caesar(ENGLISH, applied)
        # raw scorer collapses on the shifted text; gauge-normalized does not
        assert base.score(shifted) < plain - 100
        assert g.score(shifted) == pytest.approx(plain)


def test_gauge_normalized_scorer_restores_gradient():
    """A decode one gauge-step from English must outscore junk under the wrapper."""
    g = GaugeNormalizedScorer()
    rng = random.Random(1)
    junk = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in ENGLISH)
    assert g.score(_caesar(ENGLISH, 7)) > g.score(junk) + 100


# ------------------------------------------------------------- excision

def test_excision_finds_a_contiguous_key_block():
    rng = random.Random(2)
    insert = "".join(rng.choice("QXZJKVW") for _ in range(17))
    contaminated = ENGLISH[:60] + insert + ENGLISH[60:]
    res = excision_score(contaminated, excise_len=17)
    assert res["at"] == 60
    assert res["excised"] == insert
    # excised read must beat the whole-text read
    assert res["score"] > get_scorer().average(contaminated)


def test_excision_column_mode_finds_the_polluted_column():
    rng = random.Random(3)
    width = 9
    chars = list(ENGLISH[:108])
    for i in range(4, 108, width):
        chars[i] = rng.choice("QXZJ")
    polluted = "".join(chars)
    res = excision_score(polluted, mode="column", width=width)
    assert res["at"] == 4


def test_excision_rejects_bad_args():
    with pytest.raises(ValueError):
        excision_score(ENGLISH, excise_len=0)
    with pytest.raises(ValueError):
        excision_score(ENGLISH)
    with pytest.raises(ValueError):
        excision_score(ENGLISH, excise_len=5, mode="column")


# ------------------------------------------------------------- anchors

def test_anchored_english_near_one_random_near_zero():
    scorer = get_scorer()
    rng = random.Random(4)
    junk = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(400))
    assert scorer.anchored(ENGLISH) > 0.8
    assert scorer.anchored(junk) < 0.25


def test_anchored_score_generic_helper():
    assert anchored_score(5.0, 0.0, 10.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        anchored_score(1.0, 2.0, 2.0)
