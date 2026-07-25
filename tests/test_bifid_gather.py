"""Tests for Bifid seriation gather-variant coverage (generalized from the
fractionation campaign: several read-orders are statistically indistinguishable,
so a blind solver must be able to try each)."""

from __future__ import annotations

from buttcrack.fractionation import (
    GATHER_VARIANTS,
    bifid6_alphabet,
    bifid6_decode,
    bifid6_encode,
)

PT = (
    "THEQUICKBROWNFOXIUMPSOVERTHELAZYDOGWHILETHEOLDCLOCKINTHEHALLSTRUCK"
    "MIDNIGHTANDTHEWINDCARRIEDTHESCENTOFRAINACROSSTHEQUIETFIELDS"
)


def test_all_gather_variants_round_trip() -> None:
    alpha = bifid6_alphabet("KRYPTOS")
    for name in GATHER_VARIANTS:
        for period in (5, 7, 11, 13):
            ct = bifid6_encode(PT, alpha, period, gather=name)
            back = bifid6_decode(ct, alpha, period, gather=name)
            assert back == PT, f"round-trip failed for gather={name} period={period}"


def test_std_is_backward_compatible() -> None:
    # the default and "std" must equal the classic row-major behaviour
    alpha = bifid6_alphabet("KRYPTOS")
    for period in (5, 7, 12):
        default = bifid6_encode(PT, alpha, period)
        named = bifid6_encode(PT, alpha, period, gather="std")
        spec = bifid6_encode(PT, alpha, period, gather=("RC", False, False))
        assert default == named == spec


def test_variants_are_actually_distinct() -> None:
    # different gathers must generally produce different ciphertext (else the
    # "coverage" would be vacuous)
    alpha = bifid6_alphabet("KRYPTOS")
    cts = {name: bifid6_encode(PT, alpha, 7, gather=name) for name in GATHER_VARIANTS}
    assert len(set(cts.values())) >= 6  # at least most of the 8 are distinct


def test_eight_variants_present() -> None:
    assert len(GATHER_VARIANTS) == 8
    assert GATHER_VARIANTS["std"] == ("RC", False, False)
