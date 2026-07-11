"""Objective power calibration: separation, powerless detection, and the power curve."""

from __future__ import annotations

import random

from buttcrack import power
from buttcrack.scoring import get_scorer
from buttcrack.validate import _FILLER


def test_separation_of_disjoint_populations():
    sep = power.separation([10.0, 11.0, 9.5, 10.5], [0.0, 1.0, 0.5, -0.5])
    assert sep.z > 3
    assert sep.auc == 1.0
    assert sep.separated


def test_separation_single_signal_point_is_zscore():
    sep = power.separation([5.0], [0.0, 1.0, 2.0, 1.0])  # null mean 1.0, sd 0.707
    assert sep.n_signal == 1
    assert sep.z == (5.0 - 1.0) / sep.sd_null


def test_no_separation_when_overlapping():
    rng = random.Random(0)
    a = [rng.gauss(0, 1) for _ in range(200)]
    b = [rng.gauss(0, 1) for _ in range(200)]
    sep = power.separation(a, b)
    assert abs(sep.z) < 1.5
    assert 0.4 < sep.auc < 0.6
    assert not sep.separated


def test_powerless_objective_flags_constant():
    rep = power.powerless_objective(lambda x: 3.0, list(range(50)))
    assert rep.powerless
    assert rep.distinct == 1


def test_powerless_objective_passes_varying():
    rep = power.powerless_objective(lambda x: float(x), list(range(50)))
    assert not rep.powerless
    assert rep.distinct == 50


def test_key_invariant_objective_is_powerless():
    # An objective that ignores its argument (a real bug pattern) is caught.
    obj = lambda key: 42.0  # noqa: E731
    keys = [f"KEY{i}" for i in range(30)]
    assert power.powerless_objective(obj, keys).powerless


def test_dilute_plaintext_interpolates_snr():
    rng = random.Random(1)
    pt = _FILLER[:200]
    assert power.dilute_plaintext(pt, 1.0, rng) == pt  # strength 1 keeps it intact
    noisy = power.dilute_plaintext(pt, 0.0, random.Random(2))
    assert noisy != pt  # strength 0 randomizes


def test_power_curve_rises_with_signal_strength():
    scorer = get_scorer()
    pt = _FILLER

    def make_signal(strength, rng):
        return power.dilute_plaintext(pt, strength, rng)

    def objective(text):
        letters = "".join(c for c in text if c.isalpha())
        return scorer.score(letters) / max(1, len(letters))

    curve = power.power_curve(
        make_signal, objective, [0.0, 0.5, 1.0], trials=15, rng=random.Random(3)
    )
    z0, z_half, z_full = (p.sep.z for p in curve)
    assert z_full > z_half > z0  # monotone: more English signal -> cleaner separation
    assert curve[-1].sep.separated


def test_power_curve_records_recovery_rate():
    scorer = get_scorer()
    pt = _FILLER

    def make_signal(strength, rng):
        return power.dilute_plaintext(pt, strength, rng)

    def objective(text):
        return scorer.score(text)

    # A trivial "recovered" gate: high per-char quadgram score.
    def recovered(text):
        letters = "".join(c for c in text if c.isalpha())
        return scorer.score(letters) / max(1, len(letters)) > -4.5

    curve = power.power_curve(
        make_signal,
        objective,
        [0.0, 1.0],
        trials=10,
        rng=random.Random(4),
        recover_fn=recovered,
    )
    assert curve[0].recover_rate == 0.0
    assert curve[-1].recover_rate == 1.0
