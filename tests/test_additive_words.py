"""Independent synthetic dictionary-sum examples and enumeration oracles."""

import itertools
import json

import pytest

from buttcrack.additive_words import recover_word_keys

AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def pad(keys, n, alphabet=AZ):
    return [sum(alphabet.index(w[i % len(w)]) for w in keys) % len(alphabet) for i in range(n)]


def oracle(stream, periods, bank, alphabet=AZ):
    choices = [[w for w in bank if len(w) == p] for p in periods]
    return {
        keys
        for keys in itertools.product(*choices)
        if all(
            v is None or v == c
            for v, c in zip(stream, pad(keys, len(stream), alphabet), strict=True)
        )
    }


def test_non_coprime_short_words_and_filter_accounting():
    bank = ["at", "AT", "IN", "DOVE", "LARK", "Q", "PUNCT!", ""]
    result = recover_word_keys(pad(["AT", "DOVE"], 19), [2, 4], bank)
    assert result["candidates"] == [["AT", "DOVE"]]
    assert result["unique_in_bank"]
    assert result["exhausted"] and result["bank_complete"]
    assert result["bank_filters"] == {
        "invalid_symbols": 2,
        "length_not_requested": 1,
        "duplicate": 1,
        "accepted": 4,
    }
    assert result["counts"] == {
        "bank_items": 8,
        "indexed_words": 2,
        "prefixes": 2,
        "verified_tuples": 1,
    }
    assert result["ordered_tuple_space"] == 4
    json.dumps(result)


@pytest.mark.parametrize("periods", [[2], [2, 2], [2, 4, 2], [1, 2, 4]])
@pytest.mark.parametrize("partial", [False, True])
def test_against_cartesian_oracle(periods, partial):
    bank = ["A", "B", "AT", "IN", "DO", "DOVE", "LARK"]
    keys = [next(w for w in bank if len(w) == p) for p in periods]
    stream = pad(keys, 29)
    if partial:
        stream = [v if i in (1, 8, 17) else None for i, v in enumerate(stream)]
    result = recover_word_keys(stream, periods, bank, max_results=1000)
    assert result["exhausted"]
    assert {tuple(w) for w in result["candidates"]} == oracle(stream, periods, bank)


def test_repeated_periods_keep_equivalent_ordered_decompositions():
    result = recover_word_keys(pad(["AT", "IN"], 20), [2, 2], ["AT", "IN"])
    assert result["candidates"] == [["AT", "IN"], ["IN", "AT"]]
    assert not result["unique_in_bank"]


def test_unknown_positions_are_not_zero_and_cap_never_claims_unique():
    result = recover_word_keys([None] * 12, [2, 2], ["AT", "IN"], max_results=1)
    assert result["candidates"] == [["AT", "AT"]]
    assert result["capped"] and not result["exhausted"] and not result["unique_in_bank"]
    assert result["limit_reason"] == "max_results"


def test_contradictory_known_residues():
    assert recover_word_keys([0, 0, 1], [2], ["AA"])["candidates"] == []


def test_custom_alphabet():
    alphabet = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    stream = pad(["AT", "DOVE"], 20, alphabet)
    result = recover_word_keys(stream, [2, 4], ["AT", "IN", "DOVE"], alphabet=alphabet)
    assert result["candidates"] == [["AT", "DOVE"]]


def test_unbounded_bank_is_bounded():
    result = recover_word_keys([0], [1], itertools.repeat("A"), max_nodes=7)
    assert result["nodes_used"] == 7 and result["counts"]["bank_items"] == 7
    assert result["capped"] and not result["bank_complete"]
    assert result["ordered_tuple_space"] is None


def test_search_cap_with_complete_bank():
    result = recover_word_keys([None], [2, 2], ["AT", "IN"], max_nodes=5)
    assert result["bank_complete"] and result["capped"]
    assert result["nodes_used"] == 5
    assert result["limit_reason"] == "max_nodes:verification"


def test_missing_length_exhausts_empty_space():
    result = recover_word_keys([1], [2, 9], ["AT"])
    assert result["exhausted"] and result["ordered_tuple_space"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"periods": []},
        {"periods": [0]},
        {"periods": [True]},
        {"keystream": [-1]},
        {"keystream": [26]},
        {"keystream": [True]},
        {"alphabet": "AA"},
        {"alphabet": "ab"},
        {"max_nodes": 0},
        {"max_results": 0},
    ],
)
def test_invalid_inputs(kwargs):
    options = {"keystream": [0], "periods": [1], "words": ["A"]} | kwargs
    with pytest.raises(ValueError):
        recover_word_keys(**options)


def test_unicode_word_is_not_transliterated_into_the_bank():
    result = recover_word_keys([None], [2], ["ß", "SS"])
    assert result["candidates"] == [["SS"]]
    assert result["bank_filters"]["invalid_symbols"] == 1
    assert result["bank_filters"]["accepted"] == 1
