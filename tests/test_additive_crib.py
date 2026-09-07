"""Synthetic and exhaustive checks for exact additive-sum constraints."""

import itertools
import random

import pytest

from buttcrack.additive_crib import (
    analyze_periods,
    drag_additive_crib,
    period_sets,
    solve_additive_crib,
)
from buttcrack.crib_algebra import crib_solve

AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encrypt(text, keys, alphabet=AZ):
    # Separate round-by-round construction, rather than a composite pad.
    for key in keys:
        text = "".join(
            alphabet[(alphabet.index(c) + alphabet.index(key[i % len(key)])) % 26]
            for i, c in enumerate(text)
        )
    return text


def test_non_coprime_multiple_keys_recover_message_and_representative():
    text = "THEBAKERCHECKSTHEOVENBEFOREMIXINGANOTHERBATCHOFDOUGH"
    keys = ["UP", "INK", "BREEZE", "NOTEBOOK"]
    periods = list(map(len, keys))
    ct = encrypt(text, keys)
    result = solve_additive_crib(ct, periods, [(0, text[:12])])
    assert result["effective_parameters"] == 12
    assert result["gauge_dimensions"] == 7
    assert result["constraint_ranks"] == {"2": 12, "13": 12}
    assert result["status"] == "determined"
    assert result["plaintext"] == text
    representative = ["".join(AZ[x] for x in key) for key in result["representative_keys"]]
    assert encrypt(text, representative) == ct


@pytest.mark.parametrize(
    "periods,dimension,bound", [([2, 6], 6, 6), ([3, 3, 3], 3, 3), ([3, 5], 7, 15)]
)
def test_shared_and_repeated_period_analysis(periods, dimension, bound):
    result = analyze_periods(periods)
    assert result["effective_parameters"] == dimension
    assert result["period_bound"] == bound
    assert result["raw_parameters"] == sum(periods)


def test_period_sets_include_shared_factors_and_budget_effective_rank():
    assert (2, 6) in period_sets(2, 2, 6, 6)
    assert (2, 6) not in period_sets(2, 2, 6, 6, coprime_only=True)
    assert (3, 5) not in period_sets(2, 2, 6, 6)


def test_partial_result_never_fills_unknowns_with_a_guessed_representative():
    text = "SOMESYNTHETICTEXT"
    ct = encrypt(text, ["FOX", "OWL"])
    result = solve_additive_crib(ct, [3, 3], [(1, text[1])])
    assert result["status"] == "partial"
    assert result["determined_positions"] == list(range(1, len(text), 3))
    assert result["plaintext"] == "".join(c if i % 3 == 1 else "?" for i, c in enumerate(text))
    assert solve_additive_crib(ct, [3, 3], [])["plaintext"] == "?" * len(ct)


@pytest.mark.parametrize("wrong", ["B", "N", "C"])
def test_conflicts_in_either_field_disclose_no_plaintext(wrong):
    result = solve_additive_crib("AAAA", [1, 2], [(0, "A"), (0, wrong)])
    assert not result["consistent"] and result["status"] == "inconsistent"
    assert result["conflicts"]
    assert result["keystream"] == [None] * 4
    assert result["representative_keys"] is None


def test_exhaustive_small_model_agrees_on_every_determined_value():
    text = "ABCDE"
    ct = encrypt(text, ["D", "AZ"])
    fixed = [(0, text[0])]
    result = solve_additive_crib(ct, [1, 2], fixed)
    possible = [set() for _ in ct]
    for a, b, c in itertools.product(range(26), repeat=3):
        pad = [(a + [b, c][i % 2]) % 26 for i in range(len(ct))]
        candidate = [(AZ.index(ch) - pad[i]) % 26 for i, ch in enumerate(ct)]
        if candidate[0] == AZ.index(text[0]):
            for i, value in enumerate(candidate):
                possible[i].add(value)
    for i, values in enumerate(possible):
        assert (result["keystream"][i] is not None) == (len(values) == 1)
        if len(values) == 1:
            assert result["plaintext"][i] == AZ[next(iter(values))]


def test_agrees_with_independent_two_key_graph_solver():
    rng = random.Random(417)
    for periods in [(2, 6), (3, 5), (4, 4)]:
        text = "".join(rng.choice(AZ) for _ in range(31))
        keys = ["".join(rng.choice(AZ) for _ in range(p)) for p in periods]
        ct = encrypt(text, keys)
        positions = sorted(rng.sample(range(len(text)), 7))
        actual = solve_additive_crib(ct, periods, [(i, text[i]) for i in positions])
        reference = crib_solve(
            [AZ.index(c) for c in ct],
            [AZ.index(text[i]) for i in positions],
            positions,
            *periods,
        )
        assert actual["determined_positions"] == reference["determined_positions"]
        assert actual["consistent"] == reference["consistent"]


def test_custom_alphabet_and_fragment_offsets_count_only_letters():
    alphabet = AZ[::-1]
    text = "BREADANDJAM"
    ct = encrypt(text, ["UP", "BAG"], alphabet)
    result = solve_additive_crib(ct[:3] + "! " + ct[3:], [2, 3], [(0, "BRE AD")], alphabet=alphabet)
    assert result["plaintext"] == text


def test_float_crib_preserves_ambiguity_and_reports_caps():
    text = "THEBAKERMIXESTHEDOUGH"
    ct = encrypt(text, ["RED"])
    result = drag_additive_crib(ct, [3], "BAKER", fragments=[(0, text[:3])])
    assert result["status"] == "exhausted_placements"
    assert any(c["offset"] == 3 and c["result"]["plaintext"] == text for c in result["candidates"])
    capped = drag_additive_crib(ct, [3], "B", max_placements=2)
    assert capped["status"] == "capped" and capped["evaluated"] == 2


@pytest.mark.parametrize("periods", [[], [0], [-1], [True], [1.5], [513]])
def test_invalid_periods(periods):
    with pytest.raises(ValueError):
        analyze_periods(periods)


@pytest.mark.parametrize("fragments", [[(-1, "A")], [(3, "AB")], [(0, "!!!")]])
def test_invalid_fragments(fragments):
    with pytest.raises(ValueError):
        solve_additive_crib("ABCD", [2], fragments)


def test_scattered_constraints_with_different_field_ranks_preserve_ambiguity():
    # A three-component incidence matrix can lose rank in characteristic two.
    # Exhaustively enumerate a genuine mod-26 subspace, independently of Gaussian
    # elimination: each binary key coordinate is lifted by multiplying by 13.
    positions = [0, 7, 12, 13, 17, 21, 23, 26]
    result = solve_additive_crib("A" * 30, [2, 3, 5], [(i, "A") for i in positions])
    assert result["constraint_ranks"] == {"2": 6, "13": 7}
    witnessed = [set() for _ in range(30)]
    for values in itertools.product((0, 13), repeat=10):
        pad = [(values[i % 2] + values[2 + i % 3] + values[5 + i % 5]) % 26 for i in range(30)]
        if all(pad[i] == 0 for i in positions):
            for i, value in enumerate(pad):
                witnessed[i].add(value)
    assert result["determined_positions"] == [
        i for i, possibilities in enumerate(witnessed) if len(possibilities) == 1
    ]
    for i, value in enumerate(result["keystream"]):
        if value is None:
            # Two explicit valid key assignments disagree here. The CRT solver
            # must not publish a value from just its higher-rank field.
            assert witnessed[i] == {0, 13}
        else:
            assert value == 0
    assert result["keystream"][1] is None


def test_period_analysis_does_not_need_to_expand_the_full_cycle():
    result = analyze_periods([101, 103, 107])
    assert result["effective_parameters"] == 101 + 103 + 107 - 2
    assert result["period_bound"] == 101 * 103 * 107
    assert result["gauge_dimensions"] == 2
