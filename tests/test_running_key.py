"""Blind joint running-key recovery.

These tests pin two different kinds of fact: that the machinery is self-consistent, and that the
*measured limitation* of n-gram scoring still holds. The second matters as much as the first —
the whole reason this solver exists is that the obvious scorer provably cannot drive it, and a
future change that appears to "fix" that is far more likely to be a scoring bug than a
breakthrough.
"""

from __future__ import annotations

import pytest

from buttcrack.ring_tables import ALPHABET_RING, KRYPTOS_RING
from buttcrack.running_key import (
    CONVENTIONS,
    NgramStreamScorer,
    charmatch,
    joint_beam,
    key_stream,
)

PT = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHEN"
KEY = "WHENINTHECOURSEOFHUMANEVENTSITBECOMESNECES"


def _pos(text: str, ring: str) -> list[int]:
    return [ring.index(c) for c in text]


def _encipher(pt: list[int], key: list[int], mode: str) -> list[int]:
    """Invert CONVENTIONS: find the ct that yields this key for this pt."""
    out = []
    for p, k in zip(pt, key, strict=True):
        (ct,) = [c for c in range(26) if CONVENTIONS[mode](c, p) == k]
        out.append(ct)
    return out


class TestConventions:
    @pytest.mark.parametrize("mode", sorted(CONVENTIONS))
    def test_key_stream_inverts_the_encipherment(self, mode: str) -> None:
        pt = _pos(PT, KRYPTOS_RING)
        key = _pos(KEY, KRYPTOS_RING)
        ct = _encipher(pt, key, mode)
        assert key_stream(ct, pt, mode) == key

    def test_the_three_conventions_are_distinct(self) -> None:
        got = {mode: CONVENTIONS[mode](7, 3) for mode in CONVENTIONS}
        assert len(set(got.values())) == len(got), got

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            joint_beam([0, 1, 2], NgramStreamScorer(), mode="nope")


class TestCharmatch:
    def test_swapped_streams_count_as_a_full_recovery(self) -> None:
        """pt + key == key + pt, so a swapped answer is correct, not noise."""
        pt = _pos(PT, KRYPTOS_RING)
        key = _pos(KEY, KRYPTOS_RING)
        frac, swapped = charmatch(key, pt, pt, key)
        assert frac == 1.0
        assert swapped is True

    def test_direct_orientation_is_preferred_when_both_fit(self) -> None:
        pt = _pos(PT, KRYPTOS_RING)
        key = _pos(KEY, KRYPTOS_RING)
        frac, swapped = charmatch(pt, key, pt, key)
        assert frac == 1.0
        assert swapped is False

    def test_mismatched_lengths_raise_rather_than_truncate(self) -> None:
        with pytest.raises(ValueError):
            charmatch([1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3])


class TestJointBeam:
    @pytest.mark.parametrize("mode", sorted(CONVENTIONS))
    def test_returned_decomposition_reproduces_the_ciphertext(self, mode: str) -> None:
        """Whatever it finds must be a *valid* decomposition of the actual ciphertext."""
        pt = _pos(PT[:32], KRYPTOS_RING)
        key = _pos(KEY[:32], KRYPTOS_RING)
        ct = _encipher(pt, key, mode)
        rec = joint_beam(ct, NgramStreamScorer(), beam=32, mode=mode)
        assert len(rec.plaintext) == len(ct)
        assert key_stream(ct, rec.plaintext, mode) == rec.key

    def test_per_char_is_the_comparable_quantity(self) -> None:
        ct = _pos(PT[:24], KRYPTOS_RING)
        rec = joint_beam(ct, NgramStreamScorer(), beam=16)
        assert rec.per_char == pytest.approx(rec.score / 24)

    def test_empty_ciphertext_is_handled(self) -> None:
        rec = joint_beam([], NgramStreamScorer(), beam=8)
        assert rec.plaintext == [] and rec.key == []

    def test_progress_callback_sees_every_position(self) -> None:
        """A long search with no progress signal is a job that cannot be supervised."""
        seen: list[int] = []
        ct = _pos(PT[:20], KRYPTOS_RING)
        joint_beam(ct, NgramStreamScorer(), beam=8, progress=lambda i, n, s: seen.append(i))
        assert seen == list(range(1, 20))

    def test_the_ring_actually_changes_the_search(self) -> None:
        ct = _pos(PT[:28], KRYPTOS_RING)
        a = joint_beam(ct, NgramStreamScorer(KRYPTOS_RING), beam=32)
        b = joint_beam(ct, NgramStreamScorer(ALPHABET_RING), beam=32)
        assert a.plaintext != b.plaintext, "ring must not be a no-op"


class TestMeasuredLimitation:
    """n-gram scoring cannot drive blind recovery — pinned so a 'fix' gets scrutinised.

    If this test ever fails, the overwhelmingly likely cause is a scoring bug (an A-Z table being
    indexed with ring positions, say), not that n-grams became sufficient.
    """

    def test_ngram_objective_does_not_put_its_optimum_on_the_truth(self) -> None:
        scorer = NgramStreamScorer()
        pt = _pos(PT, KRYPTOS_RING)
        key = _pos(KEY, KRYPTOS_RING)
        ct = _encipher(pt, key, "vig")

        def objective(p: list[int], k: list[int]) -> float:
            total = 0.0
            for i in range(1, len(p)):
                lp = scorer.next_logprobs([p[:i], k[:i]])
                total += lp[0][p[i]] + lp[1][k[i]]
            return total

        found = joint_beam(ct, scorer, beam=256, mode="vig")
        truth = objective(pt, key)
        beam_score = objective(found.plaintext, found.key)
        assert beam_score >= truth, (
            "the n-gram beam found a decomposition scoring no better than the truth; "
            "verify the scorer before believing it"
        )

    def test_ngram_recovery_is_poor(self) -> None:
        pt = _pos(PT, KRYPTOS_RING)
        key = _pos(KEY, KRYPTOS_RING)
        ct = _encipher(pt, key, "vig")
        found = joint_beam(ct, NgramStreamScorer(), beam=256, mode="vig")
        frac, _ = charmatch(found.plaintext, found.key, pt, key)
        assert frac < 0.9, f"n-gram scoring should not recover a blind running key, got {frac}"
