"""Landscape probes, checked on two known landscapes: a smooth hill and a needle."""

from __future__ import annotations

import random

from buttcrack import landscape

TRUTH = (3, 14, 15, 9, 2, 6)
N = len(TRUTH)


def neighbors(key):
    """All single-position, ±1 (mod 26) moves — a standard local move set."""
    for i in range(N):
        for d in (-1, 1):
            yield (*key[:i], (key[i] + d) % 26, *key[i + 1 :])


def _ring_dist(a, b):
    d = abs(a - b) % 26
    return min(d, 26 - d)


def smooth(key):
    """Climbable: score falls gently with ring distance from the truth."""
    return -sum(_ring_dist(k, t) for k, t in zip(key, TRUTH, strict=True))


def needle(key):
    """Needle: only the exact truth scores; everything else is flat noise."""
    return 1.0 if key == TRUTH else 0.0


def cliff(key):
    """Cliff: any wrong element collapses the score into flat NOISE (like a real
    n-gram objective past a fractionation cliff — the residual slope is swamped)."""
    wrong = sum(1 for k, t in zip(key, TRUTH, strict=True) if k != t)
    if wrong == 0:
        return 100.0
    noise = (hash(key) % 1000) / 1000.0  # deterministic per-key pseudo-noise
    return 10.0 - 0.001 * wrong + noise


def damage_fn(key, k, rng):
    key = list(key)
    for i in rng.sample(range(N), min(k, N)):
        key[i] = (key[i] + rng.randrange(1, 26)) % 26
    return tuple(key)


# ------------------------------------------------------------- probe A


def test_truth_is_local_max_on_smooth_and_needle():
    assert landscape.local_max_probe(TRUTH, neighbors, smooth).is_local_max
    assert landscape.local_max_probe(TRUTH, neighbors, needle).is_local_max


def test_probe_a_fails_when_objective_prefers_a_neighbor():
    def biased(key):  # objective whose optimum is NOT the truth
        return smooth(key) + (2.0 if key != TRUTH else 0.0)

    probe = landscape.local_max_probe(TRUTH, neighbors, biased)
    assert not probe.is_local_max
    assert "cannot identify" in probe.summary()


# ------------------------------------------------------------- probe B


def test_identifiable_on_smooth_landscape():
    rng = random.Random(1)
    seeds = [damage_fn(TRUTH, 3, rng) for _ in range(5)]
    res = landscape.identifiability(TRUTH, neighbors, smooth, seeds)
    assert res.verdict == "identifiable"
    assert res.reached == 5


def test_needle_needs_richer_seeds():
    rng = random.Random(2)
    seeds = [damage_fn(TRUTH, 3, rng) for _ in range(5)]
    res = landscape.identifiability(TRUTH, neighbors, needle, seeds)
    assert res.verdict == "needs-richer-seeds"
    assert res.reached == 0


def test_dead_lane_when_truth_not_local_max():
    def biased(key):
        return smooth(key) + (2.0 if key != TRUTH else 0.0)

    rng = random.Random(3)
    seeds = [damage_fn(TRUTH, 2, rng) for _ in range(3)]
    res = landscape.identifiability(TRUTH, neighbors, biased, seeds)
    assert res.verdict == "dead-lane"


# ------------------------------------------------------------- probe C


def test_crib_floor_on_needle_is_full_lock():
    """A needle objective recovers only when everything except noise is locked."""

    def recover_at(k, rng):
        # lock k positions to truth, climb the rest under the needle objective:
        # succeeds only if the unlocked remainder is already correct (k == N)
        start = tuple(TRUTH[i] if i < k else (TRUTH[i] + 5) % 26 for i in range(N))
        end, _, _ = landscape.best_improvement_climb(start, neighbors, needle)
        return end == TRUTH

    res = landscape.crib_floor(recover_at, sizes=[0, 2, 4, N], trials=4, rng=random.Random(4))
    assert res["floor"] == N
    assert res["curve"][0]["rate"] == 0.0


def test_crib_floor_on_smooth_is_zero():
    def recover_at(k, rng):
        start = tuple(TRUTH[i] if i < k else (TRUTH[i] + 7) % 26 for i in range(N))
        end, _, _ = landscape.best_improvement_climb(start, neighbors, smooth)
        return end == TRUTH

    res = landscape.crib_floor(recover_at, sizes=[0, 2], trials=4, rng=random.Random(5))
    assert res["floor"] == 0


# ------------------------------------------------------------- damage ladder


def test_damage_ladder_smooth_is_not_a_cliff():
    res = landscape.damage_ladder(
        TRUTH, damage_fn, smooth, levels=[1, 2, 3, 4], trials=30, rng=random.Random(6)
    )
    assert not res["cliff"]
    # every adjacent rung separates on a climbable landscape
    assert all(a["auc"] > 0.6 for a in res["adjacent"])


def test_damage_ladder_flags_the_cliff():
    res = landscape.damage_ladder(
        TRUTH, damage_fn, cliff, levels=[1, 2, 3, 4], trials=30, rng=random.Random(7)
    )
    assert res["cliff"]
    # first step carries the drop; deeper steps do not separate
    assert res["adjacent"][0]["auc"] == 1.0
