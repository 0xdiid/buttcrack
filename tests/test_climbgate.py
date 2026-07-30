"""Solver-as-detector: gate bands, target positioning, and the ungated guard."""

from __future__ import annotations

import random

from buttcrack import climbgate, registry
from buttcrack.validate import _FILLER


def test_caesar_is_gated_at_moderate_length():
    band = climbgate.solver_band("caesar", 120, trials=2, rng=random.Random(1))
    assert band.gated, band.summary()
    assert band.gate_z >= climbgate.MIN_GATE_Z


def test_detector_places_a_caesar_target_in_the_gate_band():
    ct = registry.get("caesar").encode(_FILLER[:120], "7")
    rows = climbgate.detector_sweep(ct, ["caesar"], trials=2, rng=random.Random(2))
    assert rows[0].verdict == "in-gate-band", rows[0].summary()
    assert rows[0].position is not None and rows[0].position > 0.75


def test_detector_reads_random_text_as_noise_or_ctrl():
    rng = random.Random(3)
    junk = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(120))
    rows = climbgate.detector_sweep(junk, ["caesar"], trials=2, rng=random.Random(4))
    assert rows[0].verdict in ("noise", "above-ctrl"), rows[0].summary()
    assert rows[0].position is None or rows[0].position < 0.75


def test_ungated_solver_is_suppressed_not_believed():
    """At absurdly short length no solver separates from its own noise band —
    the row must say 'ungated', never 'noise' (silence is not evidence)."""
    band = climbgate.solver_band("vigenere", 8, trials=2, rng=random.Random(5))
    assert not band.gated
    rows = climbgate.detector_sweep("ABCDEFGH", ["vigenere"], trials=2, rng=random.Random(6))
    assert rows[0].verdict == "ungated"
    assert rows[0].target_score is None


def test_length_threshold_turns_on_with_n():
    res = climbgate.length_threshold("caesar", [3, 120], trials=2, rng=random.Random(7))
    assert res["threshold"] == 120, res
    assert res["curve"][0]["gated"] is False
