"""Tests for the validate-on-synthetic harness.

Two things must hold for the harness to be trustworthy:
  1. every family-grammar structure round-trips (synthetic decodes back to its plaintext);
  2. a real solver passes its own ``positive_control`` while a deliberately-broken attack
     fails it — the whole point of the ritual.
"""

from __future__ import annotations

from buttcrack.ciphers.columnar import Columnar, _decode_letters, _read_order
from buttcrack.layered import crack_quagmire_over_columnar
from buttcrack.scoring import get_scorer
from buttcrack import validate
from buttcrack.validate import (
    STRUCTURES,
    SUBSTITUTIONS,
    decode_substitution,
    encode_columnar,
    encode_substitution,
    genuine_solve_signature,
    make_synthetic,
    positive_control,
    random_key,
)


def test_substitution_round_trips_all_families_and_alphabets():
    pt = "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARROWST"
    for sub in SUBSTITUTIONS:
        for alpha in ("KRYPTOS", "STD"):
            ct = encode_substitution(pt, "MEADOW", substitution=sub, alphabet=alpha)
            back = decode_substitution(ct, "MEADOW", substitution=sub, alphabet=alpha)
            assert back == pt, (sub, alpha)
            assert ct != pt  # actually transformed


def test_substitution_conventions_match_grammar():
    # Vigenere enc c=p+k, Beaufort enc c=k-p (reciprocal), Variant enc c=p-k — in keyed
    # index space. Spot-check on the standard alphabet where indices are obvious.
    # P='A'(0), key='C'(2): vigenere -> 'C', variant -> 'Y' (0-2 mod 26 = 24), beaufort 'C'.
    assert encode_substitution("A", "C", substitution="vigenere", alphabet="STD") == "C"
    assert encode_substitution("A", "C", substitution="variant", alphabet="STD") == "Y"
    assert encode_substitution("A", "C", substitution="beaufort", alphabet="STD") == "C"
    # Beaufort is reciprocal: encoding twice with the same key returns the plaintext.
    pt = "ATTACKATDAWN"
    once = encode_substitution(pt, "KEY", substitution="beaufort", alphabet="STD")
    assert encode_substitution(once, "KEY", substitution="beaufort", alphabet="STD") == pt


def test_columnar_uses_rankorder_in_standard_alphabet():
    pt = "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARX"
    ct = encode_columnar(pt, "WORKSHOP")
    # read-order = rankorder(keyword) ranked in standard A-Z, ties left-to-right.
    assert _decode_letters(ct, _read_order("WORKSHOP")) == pt


def test_make_synthetic_every_structure_has_exact_length():
    spec_base = {
        "substitution": "vigenere",
        "alphabet": "KRYPTOS",
        "sub_key": "MEADOW",
        "columnar_keyword": "WORKSHOP",
        "columnar_keywords": ["TELESCOPE", "HURRICANE"],
    }
    for structure in STRUCTURES:
        synth = make_synthetic({**spec_base, "structure": structure}, length=144)
        assert synth["length"] == 144
        assert len(synth["ciphertext"]) == 144
        assert synth["structure"] == structure
        # ciphertext must differ from plaintext for every structure
        assert synth["ciphertext"] != synth["plaintext"]


def test_make_synthetic_substitution_over_columnar_decodes_back():
    # Sub-over-columnar shape: CT = Vig(columnar(PT)). Manually invert to confirm the synthetic
    # is the exact composition the layered solver targets.
    synth = make_synthetic(
        {
            "structure": "substitution-over-columnar",
            "substitution": "vigenere",
            "alphabet": "KRYPTOS",
            "sub_key": "MEADOW",
            "columnar_keyword": "WORKSHOP",
        },
        length=128,
    )
    inner = decode_substitution(
        synth["ciphertext"], "MEADOW", substitution="vigenere", alphabet="KRYPTOS"
    )
    plain = _decode_letters(inner, _read_order("WORKSHOP"))
    assert plain == synth["plaintext"]


def test_genuine_solve_signature_anchored_at_272():
    sig = genuine_solve_signature(272)
    assert sig["qscore_per_char"] == -4.2
    assert sig["word_cov"] == 0.69
    # shorter text loosens both bars but never tightens past the 272 anchor
    short = genuine_solve_signature(80)
    assert short["qscore_per_char"] <= sig["qscore_per_char"]
    assert short["word_cov"] <= sig["word_cov"]


def test_positive_control_passes_real_columnar_solver():
    scorer = get_scorer()

    def attack(ct: str) -> str:
        cands = Columnar().crack(ct, scorer, top=1, width=6)
        return cands[0].plaintext if cands else ""

    spec = {"structure": "columnar", "columnar_keyword": "CIPHER"}
    res = positive_control(attack, spec, {"columnar_keyword": "CIPHER"}, length=132)
    assert res["recovered"] is True
    assert res["word_cov"] >= 0.4
    assert res["plaintext_head"][:32] in res["decode_preview"] or res["word_cov"] >= 0.5


def test_positive_control_passes_layered_solver_on_pk4_shape():
    # The sub-over-columnar grammar: Vigenere (KRYPTOS) OVER a single columnar. The layered
    # solver must recover its own structure before any negative it reports is trusted.
    scorer = get_scorer()
    spec = {
        "structure": "substitution-over-columnar",
        "substitution": "vigenere",
        "alphabet": "KRYPTOS",
        "sub_key": "MEADOW",
        "columnar_keyword": "WORKSHOP",
    }

    def attack(ct: str):
        return crack_quagmire_over_columnar(ct, scorer, period=6, widths=[8])

    res = positive_control(attack, spec, spec, length=160)
    assert res["recovered"] is True
    assert res["word_cov"] >= 0.4


def test_positive_control_fails_a_broken_attack():
    # A "solver" that returns the ciphertext unchanged must NOT pass its positive control;
    # this is the bug-catching guarantee that makes a later negative trustworthy.
    def broken(ct: str) -> str:
        return ct

    spec = {"structure": "columnar", "columnar_keyword": "CIPHER"}
    res = positive_control(broken, spec, {"columnar_keyword": "CIPHER"}, length=132)
    assert res["recovered"] is False
    assert res["word_cov"] < 0.4


def test_positive_control_fails_an_empty_attack():
    # An attack that gives up (returns nothing) is a clean non-recovery, not a crash.
    def gives_up(ct: str):
        return None

    spec = {"structure": "columnar", "columnar_keyword": "CIPHER"}
    res = positive_control(gives_up, spec, {"columnar_keyword": "CIPHER"}, length=120)
    assert res["recovered"] is False
    assert res["score"] is None


def test_random_key_is_deterministic_and_in_alphabet():
    k1 = random_key(12, alphabet="KRYPTOS", seed=7)
    k2 = random_key(12, alphabet="KRYPTOS", seed=7)
    assert k1 == k2 and len(k1) == 12
    assert set(k1) <= set("KRYPTOSABCDEFGHIJLMNQUVWXZ")


# ------------------------------------------------------------ control battery

def _good_attack(ct):
    """A working attack for the substitution family used by the battery specs."""
    from buttcrack.validate import decode_substitution

    return [decode_substitution(ct, "PALIMPSEST", substitution="vigenere",
                                alphabet="KRYPTOS")]


def _broken_attack(ct):
    """The observed failure mode: confident output unrelated to the input."""
    return ["X" * len(ct)]


def _battery_spec():
    from buttcrack.validate import encode_substitution

    pt = validate._FILLER[:200]
    ct = encode_substitution(pt, "PALIMPSEST", substitution="vigenere",
                             alphabet="KRYPTOS")
    sibling = {"ciphertext": ct, "plaintext": pt}
    plant = {
        "structure_spec": {"structure": "substitution"},
        "substitution": "vigenere",
        "alphabet": "KRYPTOS",
        "sub_key": "PALIMPSEST",
        "length": 200,
    }
    return sibling, plant


def test_control_battery_trusts_a_working_attack():
    sibling, plant = _battery_spec()
    res = validate.control_battery(_good_attack, sibling=sibling, plant=plant)
    assert res["verdict"] == "TRUSTED"
    assert all(t["passed"] for t in res["tiers"])
    assert len(res["tiers"]) == 2


def test_control_battery_voids_a_broken_attack():
    sibling, plant = _battery_spec()
    res = validate.control_battery(_broken_attack, sibling=sibling, plant=plant)
    assert res["verdict"] == "VOID"
    assert not any(t["passed"] for t in res["tiers"])


def test_control_battery_requires_at_least_one_tier():
    import pytest

    with pytest.raises(ValueError):
        validate.control_battery(_good_attack)


def test_void_feeds_the_evidence_module():
    from buttcrack.evidence import Finding

    sibling, _ = _battery_spec()
    res = validate.control_battery(_broken_attack, sibling=sibling)
    assert res["verdict"] == "VOID"
    f = Finding("no key found in 20k cells").voided(res["tiers"][0]["detail"])
    assert f.verdict() == "void"
