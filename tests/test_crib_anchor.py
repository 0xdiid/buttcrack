"""Tests for crib_anchor: crib-anchored scoring for register-resistant cracks
(anchor an SA on a guessed crib when the plaintext register defeats n-gram fitness)."""

from __future__ import annotations

from buttcrack.crib_anchor import (
    SWEET_SPOT,
    CribAnchoredScorer,
    best_position_match,
    crib_bonus,
    crib_length_advice,
)
from buttcrack.scoring import get_scorer


def test_best_position_match_finds_planted_crib() -> None:
    m, pos = best_position_match("XXXXXATTACKATDAWNYYYYY", "ATTACKATDAWN")
    assert (m, pos) == (12, 5)


def test_best_position_match_partial_and_nonletters_ignored() -> None:
    # one substituted letter (DAWN -> DOWN) -> 11 of 12; punctuation/case ignored
    m, pos = best_position_match("aaaa attack-at-down aaaa", "ATTACKATDAWN")
    assert m == 11


def test_best_position_match_edge_cases() -> None:
    assert best_position_match("SHORT", "AVERYLONGCRIB") == (0, 0)
    assert best_position_match("ANYTHING", "") == (0, 0)


def test_crib_bonus_scales_with_weight() -> None:
    assert crib_bonus("XXXATTACKATDAWNXXX", "ATTACKATDAWN", weight=30.0) == 12 * 30.0


def test_anchored_scorer_prefers_the_placement() -> None:
    scorer = get_scorer()
    anchored = CribAnchoredScorer("ATTACKATDAWN", weight=30.0, scorer=scorer)
    with_crib = "WEWILLATTACKATDAWNTOMORROW"
    without = "THEQUICKBROWNFOXJUMPEDOVERALAZYDOG"
    # placement dominates fluency: the crib-bearing decode wins despite `without`
    # being at least as fluent
    assert anchored.score(with_crib) > anchored.score(without)
    assert anchored.placement(with_crib)[0] == anchored.full == 12


def test_length_advice_encodes_the_calibration() -> None:
    assert crib_length_advice("BATTLESHIP").startswith("too short")  # 10
    assert crib_length_advice("ATTACKATDAWNSHARP") == "ideal"  # 17, in SWEET_SPOT
    assert crib_length_advice("A" * 30).startswith("too long")
    assert SWEET_SPOT == (16, 20)
