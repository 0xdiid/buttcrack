"""Tests for decoupled recovery of a periodic substitution over a bifid (Tasks 1 & 3).

CT = periodic_Vigenere/Quagmire( bifid(PT) ). The recovery does NOT jointly blind-search
key x square: given a candidate inner square it recovers the outer key by the drop-letter
constrained coordinate descent, and a driver ranks candidate squares by the recovered
decode. Payload-agnostic objectives (fitness / ioc / repeats) handle non-English payloads.
"""

from __future__ import annotations

import random

import pytest

from buttcrack.scoring import get_scorer
from buttcrack.sub_fractionation import (
    crack_sub_over_bifid,
    encrypt_sub_over_bifid,
    make_objective,
    rank_key,
    recover_outer_key_over_bifid,
    resolve_square,
    sub_decode,
    sub_encode,
)
from buttcrack.text import only_letters

ENGLISH = only_letters(
    "WHENINTHECOURSEOFHUMANEVENTSITBECOMESNECESSARYFORONEPEOPLETODISSOLVETHE"
    "POLITICALBANDSWHICHHAVECONNECTEDTHEMWITHANOTHERANDTOASSUMEAMONGTHEPOWERS"
    "OFTHEEARTHTHESEPARATEANDSTATIONTOWHICHTHELAWSOFNATUREENTITLETHEMADECENT"
    "RESPECTREUIRESTHATTHEYDECLARETHECAUSES"
).replace("J", "I")


def _plant(pt, square_kw, shifts, *, inner=7, drop="J", alpha="KRYPTOS"):
    return encrypt_sub_over_bifid(
        pt, square_kw, outer_alphabet=alpha, inner_period=inner, outer_shifts=shifts, drop_letter=drop
    )


def test_sub_encode_decode_roundtrip():
    alpha = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    shifts = [3, 17, 0, 9, 22, 5, 14]
    inter = ENGLISH[:120]
    assert sub_decode(sub_encode(inter, alpha, shifts), alpha, shifts) == inter


# --- Task 1: key-given-structure recovery ---------------------------------- #
def test_recover_outer_key_given_correct_square():
    """Given the correct inner square, the outer Vig7 key is recovered exactly."""
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "KEYWORD", shifts)
    sq = resolve_square("KEYWORD", "J")
    rec_shifts, pt, score = recover_outer_key_over_bifid(
        ct, sq, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7,
        objective="fitness", rng=random.Random(0),
    )
    assert rec_shifts == shifts
    assert only_letters(pt) == ENGLISH


def test_wrong_square_plateaus_far_below():
    """A near-miss square (two cells swapped) plateaus in the noise (~ -6.9/char)."""
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "KEYWORD", shifts)
    sc = get_scorer()
    good = resolve_square("KEYWORD", "J")
    _, pt_good, s_good = recover_outer_key_over_bifid(
        ct, good, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7, rng=random.Random(0)
    )
    near = list(good)
    near[0], near[1] = near[1], near[0]
    _, pt_bad, s_bad = recover_outer_key_over_bifid(
        ct, "".join(near), outer_alphabet="KRYPTOS", inner_period=7, outer_period=7,
        rng=random.Random(0),
    )
    assert s_good > s_bad + 1.0  # objective (fitness) clearly separates
    assert sc.average(pt_good) > -5.0 > sc.average(pt_bad)  # readable vs plateau


def test_driver_recovers_square_key_and_plaintext():
    """The square-scanning driver picks the true square and recovers the plaintext."""
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(ENGLISH, "KEYWORD", shifts)
    res = crack_sub_over_bifid(
        ct, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7,
        squares=["KEYWORD", "MYSTERY", "PALIMPSEST", "SHADOW", "CIPHER"],
        objective="fitness", top=3, rng=random.Random(0),
    )
    top_square, _key, top_pt, top_score = res[0]
    assert top_square == resolve_square("KEYWORD", "J")
    assert only_letters(top_pt) == ENGLISH
    assert top_score > res[1][3] + 1.0  # true square well above the runner-up plateau


# --- Task 2 x Task 1: drop-letter as a free parameter ----------------------- #
def test_dropletter_sweep_recovers_non_j_bifid():
    """A Q-dropped construction: the J assumption fails, the drop sweep recovers it."""
    qfree = only_letters(ENGLISH.replace("Q", ""))  # ensure Q-free payload
    rng = random.Random(4)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(qfree, "RIVERBANK", shifts, drop="Q")
    squares = ["RIVERBANK", "MAPLE", "PALIMPSEST", "KEYWORD"]
    j_only = crack_sub_over_bifid(
        ct, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7,
        squares=squares, objective="fitness", drop_letter="J", top=1, rng=random.Random(0),
    )
    assert only_letters(j_only[0][2]) != qfree  # wrong drop -> structurally blind
    swept = crack_sub_over_bifid(
        ct, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7,
        squares=squares, objective="fitness", drop_letter="IJQZ", top=1, rng=random.Random(0),
    )
    assert only_letters(swept[0][2]) == qfree


# --- Task 3: payload-agnostic objectives ------------------------------------ #
def _route_payload(n=210):
    """A repetitive coordinate/route payload whose quadgrams are un-English."""
    toks = ["NORDKZ", "ESTQV", "SUDXW", "OVESTZK", "PASSIQ", "GRIGLIAK", "ZULUX", "KILOWY"]
    r = random.Random(0)
    s = ""
    while len(s) < n:
        s += r.choice(toks)
    return only_letters(s[:n]).replace("J", "I")


def test_route_payload_ioc_and_repeats_recover_english_fitness_misses():
    """A route payload: ioc/repeats recover the square+key; English quadgram fitness misses."""
    route = _route_payload()
    rng = random.Random(1)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _plant(route, "RIVERBANK", shifts)
    squares = ["RIVERBANK", "MAPLE", "PALIMPSEST", "ABSCISSA", "KEYWORD", "MYSTERY"]
    sq_true = resolve_square("RIVERBANK", "J")

    def run(obj):
        return crack_sub_over_bifid(
            ct, outer_alphabet="KRYPTOS", inner_period=7, outer_period=7,
            squares=squares, objective=obj, top=1, rng=random.Random(0),
        )[0]

    for obj in ("repeats", "ioc"):
        square, _key, pt, _score = run(obj)
        assert square == sq_true and only_letters(pt) == route, f"{obj} failed"

    # Pure English quadgram fitness cannot recover this non-English payload.
    _sq, _k, pt_fit, _s = run("fitness")
    assert only_letters(pt_fit) != route


def test_make_objective_rejects_unknown():
    with pytest.raises(ValueError):
        make_objective("nonsense")


# --- Task 3: final ranking -------------------------------------------------- #
def test_rank_key_prefers_readable_and_counts_vocab():
    english = ENGLISH[:120]
    junk = "QXZJKWVQXZJKWV" * 8
    cands = [
        ("SQ1", "AAA", junk, 0.1),
        ("SQ2", "BBB", english, 0.2),
    ]
    ranked = rank_key(cands, vocab=["COURSE", "PEOPLE"])
    assert ranked[0]["plaintext"] == english
    assert ranked[0]["fitness"] > ranked[1]["fitness"]
    assert ranked[0]["vocab_hits"] >= 1  # COURSE / PEOPLE appear in the English decode
