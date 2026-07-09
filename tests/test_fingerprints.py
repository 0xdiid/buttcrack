"""Extended statistical fingerprints: period detectors, cross-text tests, discriminants.

Each is checked against a planted control — a period-6/4 Vigenere for the period tools, two
same-key texts for the cross-text test, and English vs a bifid flattener for the discriminants.
"""

from __future__ import annotations

import random

from buttcrack import analysis as an
from buttcrack.ciphers.bifid import Bifid
from buttcrack.text import only_letters

_A = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EN = only_letters(
    "INTHEQUIETOFTHELIBRARYTHESCHOLARTURNSTHEBRITTLEPAGESOFANANCIENTATLASTRACINGF"
    "ADEDCOASTLINESWITHAGLOVEDFINGEREVERYMAPTELLSOFHARBORSLONGSILTEDOVERANDROADST"
    "HATVANISHINTOFORESTSHECOPIESEACHLEGENDINTOHERNOTEBOOKANDNUMBERSEVERYPLATEFOR"
    "THEARCHIVESHEWILLREBINDTH"
)
EN2 = only_letters(
    "THEHARBORMASTERKEEPSALOGOFEVERYVESSELTHATPASSESTHEBREAKWATERANDNOTESTHEWEATH"
    "ERINAMARGINWITHBLUEINKHISDAUGHTERPAINTSSMALLPORTRAITSOFTHECAPTAINSWHILETHEYW"
    "AITFORT"
)


def _vig(pt, key):
    return "".join(_A[(ord(c) - 65 + ord(key[i % len(key)]) - 65) % 26] for i, c in enumerate(pt))


def _rand(n, seed=1):
    r = random.Random(seed)
    return "".join(r.choice(_A) for _ in range(n))


V6 = _vig(EN, "MEADOW")
V4 = _vig(EN, "MAPL")
BIFID = Bifid().encode(EN, "GREENHOUSE/13")


# ---- period detectors ---------------------------------------------------- #
def test_twist_finds_period():
    assert an.twist_periods(V6, max_period=12)[0]["period"] == 6
    # like every period detector, twist can rank a divisor first; the true period is surfaced
    assert 4 in [r["period"] for r in an.twist_periods(V4, max_period=12)[:3]]


def test_spectral_comb_finds_fundamental_not_harmonic():
    top = an.spectral_periods(V6, max_period=15)
    assert top[0]["period"] == 6  # comb suppresses the period-3 harmonic
    # a flattener has no periodic coincidence structure
    assert max(r["z"] for r in an.spectral_periods(BIFID, max_period=15)) < 3.0


def test_hamming_detects_key_length_or_multiple():
    hits = [r["period"] for r in an.hamming_periods(V6, max_period=15) if r["z"] > 3]
    assert 6 in hits or 12 in hits  # fires at the key length and its multiples
    assert max(r["z"] for r in an.hamming_periods(BIFID, max_period=15)) < 3.5


def test_friedman_estimate_is_a_sane_scalar():
    est = an.friedman_period_estimate(V6)
    assert est is not None and 2 <= est <= 15  # coarse, but in the right ballpark


# ---- two-message cross-text test ----------------------------------------- #
def test_mutual_ioc_separates_related_from_random():
    assert an.mutual_index_of_coincidence(EN, EN2) > 0.058  # two English texts
    assert an.mutual_index_of_coincidence(_rand(200, 2), _rand(200, 3)) < 0.045


def test_mutual_kappa_scan_detects_shared_key_depth():
    a, b = _vig(EN[:180], "MEADOW"), _vig(EN2[:170], "MEADOW")  # SAME key -> depth
    same = an.mutual_kappa_scan(a, b, max_shift=30)
    c = _vig(EN2[:170], "ANOTHERKEY")  # different key
    diff = an.mutual_kappa_scan(a, c, max_shift=30)
    assert same[0]["z"] > 3.0  # depth detected
    assert same[0]["z"] > diff[0]["z"] + 0.8  # and clearly beats the unrelated pair


# ---- distributional discriminants ---------------------------------------- #
def test_trigraphic_ioc_orders_english_above_flattener():
    assert an.trigraphic_ioc(EN) > an.trigraphic_ioc(BIFID)


def test_conditional_entropy_separates_structure_from_flattener():
    ce_en = an.conditional_entropy(EN)
    ce_bf = an.conditional_entropy(BIFID)
    assert ce_en["conditional_entropy"] < ce_bf["conditional_entropy"]
    assert ce_en["redundancy"] > ce_bf["redundancy"]  # English is more redundant


def test_sukhotin_finds_vowels_in_english():
    vowels = an.sukhotin_vowels(EN)["vowels"]
    # the recovered set should be vowel-heavy: at least A and E, and mostly true vowels
    assert "A" in vowels and "E" in vowels
    real = sum(1 for c in vowels if c in "AEIOU")
    assert real >= len(vowels) - 3  # a few false positives tolerated


def test_frequency_profile_match_directional():
    en = an.frequency_profile_match(EN)
    bf = an.frequency_profile_match(BIFID)
    assert en["z"] is not None and bf["z"] is not None
    assert en["z"] >= bf["z"]  # English no less monoalphabetic-shaped than a flattener


def test_serial_correlation_and_runs_return_scalars():
    sc = an.serial_correlation(V6)
    assert sc is not None and -1.0 <= sc <= 1.0
    assert an.runs_test(EN)["runs"] is not None
