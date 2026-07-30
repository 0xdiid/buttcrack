"""Insert shapes and the sensitivity sweep."""

from __future__ import annotations

import random

import pytest

from buttcrack.contamination import MODES, embed, sensitivity_sweep
from buttcrack.scoring import get_scorer, index_of_coincidence
from buttcrack.validate import _FILLER

ENGLISH = _FILLER[:200]


def test_embed_length_and_positions_every_mode():
    rng = random.Random(1)
    for mode in MODES:
        out, pos = embed(ENGLISH, 17, mode, rng=rng, width=9, period=7)
        assert len(out) == len(ENGLISH) + 17, mode
        assert len(pos) == 17, mode
        # removing the insert restores the original text exactly
        kept = "".join(c for i, c in enumerate(out) if i not in set(pos))
        assert kept == ENGLISH, mode


def test_embed_uses_supplied_filler_and_column_stride():
    rng = random.Random(2)
    out, pos = embed(ENGLISH, 10, "column", rng=rng, width=9, filler="KRYPTOSKEY")
    assert "".join(out[i] for i in pos) == "KRYPTOSKEY"
    strides = {b - a for a, b in zip(pos, pos[1:], strict=False)}
    assert strides == {9}


def test_embed_zero_k_is_identity():
    out, pos = embed(ENGLISH, 0, "contiguous", rng=random.Random(3))
    assert out == ENGLISH and pos == []


def test_embed_validates():
    with pytest.raises(ValueError):
        embed(ENGLISH, 5, "spiral")
    with pytest.raises(ValueError):
        embed(ENGLISH, 5, "column")  # no width
    with pytest.raises(ValueError):
        embed(ENGLISH, 5, "contiguous", filler="AB")


def test_sensitivity_sweep_finds_the_budget():
    scorer = get_scorer()
    res = sensitivity_sweep(
        ENGLISH,
        scorer.average,
        ks=[5, 20, 60],
        modes=("contiguous", "scattered"),
        trials=8,
        tolerance=0.3,
        rng=random.Random(4),
    )
    # a large random insert must break an n-gram gate; a tiny one must not
    assert res["budget"]["contiguous"] in (20, 60)
    assert res["budget"]["scattered"] in (5, 20)  # scattering damages more windows per char
    shifts = {(r["mode"], r["k"]): r["shift"] for r in res["records"]}
    assert shifts[("contiguous", 60)] < shifts[("contiguous", 5)] < 0


def test_sensitivity_sweep_ioc_is_more_tolerant_than_ngrams():
    """Order-invariant statistics absorb inserts better — the reason exclusions
    must state their insert budget per statistic, not per text."""
    res = sensitivity_sweep(
        ENGLISH,
        index_of_coincidence,
        ks=[5, 20],
        modes=("contiguous",),
        trials=8,
        tolerance=0.01,
        rng=random.Random(5),
    )
    assert res["budget"]["contiguous"] is None  # 10% insert barely moves IoC
