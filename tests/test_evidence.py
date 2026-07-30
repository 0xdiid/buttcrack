"""Regression tests for buttcrack.evidence.

Each `test_regression_*` reconstructs an actual failure from a real cryptanalytic
program — a confident-looking number produced by an instrument that was never
checked. The point of the module is that these are now *unsayable*.
"""

from __future__ import annotations

import pytest

from buttcrack.evidence import Coverage, Finding, PlantGate, Unverified, searched_fraction

# --------------------------------------------------------------------- basics

def test_bare_finding_will_not_render():
    f = Finding("bifid5 square recovery", observed=-640.2)
    with pytest.raises(Unverified) as e:
        f.render()
    msg = str(e.value)
    assert "plant gate" in msg
    assert "null" in msg
    assert "coverage" in msg


def test_fully_attested_finding_renders():
    f = (Finding("keyword square x word strip, all phases", observed=-640.2)
         .with_plant(12, 12, "additive7 o bifid5", "English prose")
         .with_null("shuffles the square, preserves the strip and phase", p_value=0.61)
         .with_coverage(1_180, 1_180, exhaustive=True))
    assert f.verdict() == "closed"
    out = f.render()
    assert "closed" in out and "recall=1.00" in out


def test_str_does_not_bypass_the_gate():
    with pytest.raises(Unverified):
        str(Finding("unattested"))


# ------------------------------------------------------- plant-gate semantics

def test_failed_plant_gate_blocks_the_claim():
    f = (Finding("phase-6 sweep", observed=-573.0)
         .with_plant(1, 5, "additive7 o bifid5 phase 6", "English prose")
         .with_null("coset-preserving shuffle", p_value=0.4)
         .with_coverage(100, 100))
    with pytest.raises(Unverified) as e:
        f.verdict()
    assert "plant gate FAILED" in str(e.value)


def test_plant_gate_requires_register():
    with pytest.raises(ValueError):
        PlantGate(5, 5, "bifid5", "")


def test_recall_is_reported_not_rounded_away():
    assert PlantGate(4, 5, "bifid5", "prose").recall == pytest.approx(0.8)
    assert not PlantGate(2, 5, "bifid5", "prose").passed


# ---------------------------------------------------------- coverage semantics

def test_timeouts_are_not_negatives():
    """The FM case: node cap fired on every instance; timeouts tallied as rejections."""
    cov = Coverage(evaluated=200, intended=200, timeouts=200)
    assert not cov.complete
    assert "NOT negatives" in cov.summary()


def test_capped_search_is_inconclusive_not_closed():
    f = (Finding("crib enumeration", observed=-698.6)
         .with_plant(8, 10, "bifid5 crib CSP", "English prose")
         .with_null("random square placements", p_value=0.5)
         .with_coverage(1_764, 1_764, capped=41))
    assert f.verdict() == "inconclusive"


def test_zero_evaluated_is_caught():
    """The decode-signature case: workers crashed, results list came back empty."""
    f = (Finding("crib driver sweep")
         .with_plant(9, 10, "bifid5", "prose")
         .with_null("shuffled ciphertext")
         .with_coverage(0, 1_764))
    with pytest.raises(Unverified) as e:
        f.verdict()
    assert "nothing was actually evaluated" in str(e.value)


def test_partial_coverage_is_not_a_closure():
    f = (Finding("gather sweep", observed=-637.0)
         .with_plant(5, 5, "bifid5 std", "prose")
         .with_null("coset-preserving shuffle", p_value=0.3)
         .with_coverage(2, 112))
    assert f.verdict() == "null (partial)"


# ------------------------------------------------------ multiplicity handling

def test_family_correction_demotes_a_best_of_many():
    """cIC7: raw p = 0.0037 looks decisive until you price the 25-period scan."""
    f = (Finding("period-7 coset IC", observed=1.4010)
         .with_plant(10, 10, "period-7 additive", "English prose")
         .with_null("full-letter shuffle: destroys order, preserves the multiset",
                    p_value=0.0037)
         .with_coverage(25, 25, exhaustive=True)
         .over_family(25))
    assert f.corrected_p > 0.05
    assert f.verdict() != "positive"


def test_single_hypothesis_needs_no_correction():
    f = Finding("x", p_value=0.01).over_family(1)
    assert f.corrected_p == pytest.approx(0.01)


def test_family_size_must_be_positive():
    with pytest.raises(ValueError):
        Finding("x").over_family(0)


# --------------------------------------------------------------- null wording

def test_null_must_be_described():
    with pytest.raises(ValueError):
        Finding("x").with_null("   ")


# --------------------------------------------------------------- serialisation

def test_to_dict_round_trips_the_attestations():
    f = (Finding("PK7-derived strips", observed=-586.2)
         .with_plant(5, 5, "additive7 o bifid5", "English prose")
         .with_null("strip drawn from the same derivation family", p_value=0.42)
         .with_coverage(186, 186, exhaustive=True)
         .over_family(186))
    d = f.to_dict()
    assert d["verdict"] == "closed"
    assert d["plant"]["recall"] == 1.0
    assert d["coverage"]["complete"] is True
    assert d["corrected_p"] is not None


# --------------------------------------------------------------- searched fraction

def test_searched_fraction_multiplies_axes():
    sf = searched_fraction({"orientation": 2, "period": 5, "square": 100}, 250)
    assert sf["declared"] == 1000
    assert sf["fraction"] == pytest.approx(0.25)
    assert "orientation=2" in sf["summary"]


def test_searched_fraction_rejects_overcount_and_empty():
    with pytest.raises(ValueError):
        searched_fraction({}, 0)
    with pytest.raises(ValueError):
        searched_fraction({"a": 10}, 11)


def test_coverage_of_axes_matches_product():
    cov = Coverage.of_axes(50, orientation=2, square=100)
    assert cov.intended == 200
    assert cov.evaluated == 50


# --------------------------------------------------------------- power / silent

def _closed_negative(**kwargs):
    return (Finding("no signal", **kwargs)
            .with_plant(5, 5, "bifid5", "English prose")
            .with_null("within-coset shuffle: preserves coset multisets, destroys order",
                       p_value=0.4)
            .with_coverage(100, 100, exhaustive=True))


def test_negative_without_power_keeps_legacy_verdict():
    assert _closed_negative().verdict() == "closed"


def test_underpowered_negative_is_silent_not_closed():
    f = _closed_negative().with_power(1.2)
    assert f.verdict() == "silent"
    assert "SILENT" in f.render()


def test_powered_negative_still_closes():
    assert _closed_negative().with_power(8.0).verdict() == "closed"


def test_power_does_not_suppress_a_positive():
    f = _closed_negative().with_power(1.2)
    f.p_value = 0.001
    f.family_size = 1
    assert f.verdict() == "positive"


# --------------------------------------------------------------- scoped negatives

def test_scoped_negative_renders_as_closed_scoped():
    f = _closed_negative().scoped(["non-standard ring", "terse register"],
                                  "any ring with alignability z > 3")
    assert f.verdict() == "closed (scoped)"
    out = f.render()
    assert "not closed  : non-standard ring" in out
    assert "reopen if" in out
    assert f.to_dict()["not_closed"] == ["non-standard ring", "terse register"]


# --------------------------------------------------------------- void

def test_void_renders_without_attestations():
    f = Finding("47k cells scored").voided("solver was handed a file path, scored heap garbage")
    assert f.verdict() == "void"
    assert "instrument broken" in f.render()
    assert f.to_dict()["void_reason"].startswith("solver was handed")


def test_void_reason_must_be_stated():
    with pytest.raises(ValueError):
        Finding("x").voided("  ")
