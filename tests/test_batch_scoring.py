"""BatchNgramScorer must reproduce NgramScorer.score exactly (fast path == slow path)."""

from __future__ import annotations

import random

import pytest

from buttcrack.scoring import (
    BatchNgramScorer,
    get_batch_scorer,
    get_scorer,
    text_to_ordinals,
)

np = pytest.importorskip("numpy")


def _random_text(rng: random.Random, length: int) -> str:
    return "".join(chr(65 + rng.randrange(26)) for _ in range(length))


def test_batch_matches_scalar_score_exactly():
    scorer = get_scorer("quadgrams", "english")
    batch = get_batch_scorer("quadgrams", "english")
    assert batch.vectorized  # numpy present, quadgram LUT fits
    rng = random.Random(1)
    texts = [_random_text(rng, 60) for _ in range(200)] + [
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
        "ATTACKATDAWNFROMTHENORTHERNRIDGELINE",
    ]
    mat = np.array([text_to_ordinals(t) for t in texts[:200]], dtype=np.int64)
    got = batch.score_batch(mat)
    want = [scorer.score(t) for t in texts[:200]]
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-9)


def test_score_texts_handles_ragged_lengths():
    scorer = get_scorer("quadgrams", "english")
    batch = get_batch_scorer("quadgrams", "english")
    texts = ["SHORT", "A", "", "ATTACKATDAWN", "THEQUICKBROWNFOX", "AB"]
    got = batch.score_texts(texts)
    want = [scorer.score(t) for t in texts]
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-9)


def test_short_rows_match_floor_branch():
    scorer = get_scorer("quadgrams", "english")
    batch = get_batch_scorer("quadgrams", "english")
    # rows shorter than n -> floor * max(1, L)
    mat = np.array([[0, 1], [2, 3]], dtype=np.int64)  # length 2 < 4
    got = batch.score_batch(mat)
    assert got[0] == pytest.approx(scorer.score("AB"), abs=1e-12)


def test_fitness_batch_matches_scalar():
    scorer = get_scorer("quadgrams", "english")
    batch = get_batch_scorer("quadgrams", "english")
    rng = random.Random(7)
    texts = [_random_text(rng, 50) for _ in range(50)]
    mat = np.array([text_to_ordinals(t) for t in texts], dtype=np.int64)
    got = batch.fitness_batch(mat)
    want = [scorer.fitness(t) for t in texts]
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-9)


def test_fallback_path_agrees_with_lut(monkeypatch):
    # Force the non-numpy fallback and confirm it still matches the scalar scorer.
    import buttcrack.scoring as scoring

    scorer = get_scorer("quadgrams", "english")
    monkeypatch.setattr(scoring, "_np", None)
    fallback = BatchNgramScorer(scorer)
    assert not fallback.vectorized
    rng = random.Random(3)
    texts = [_random_text(rng, 40) for _ in range(20)]
    rows = [text_to_ordinals(t) for t in texts]
    got = fallback.score_batch(rows)
    want = [scorer.score(t) for t in texts]
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-12)
