"""Objective-landscape profiling — is the truth findable BEFORE you scale the search?

The most expensive failure in a long cryptanalytic program is not a slow solver, it
is weeks of compute pointed at an objective that could never have identified the
answer: a "cliff" landscape where one wrong key element erases 84% of the score, a
needle objective with no gradient toward the truth, a crib bonus that doesn't create
a climbable basin. Every instrument here answers, on a plant whose key is KNOWN,
some form of "would my search find this?" — and each has a sharp verdict attached,
because the conclusions differ in kind:

* :func:`local_max_probe` (probe A) — is the TRUE key even a local maximum of the
  objective under the search's own move set? If not, **no amount of search
  improvement helps**: the objective cannot identify the target and the lane is
  dead as posed. Fix the objective, not the search.
* :func:`best_improvement_climb` / :func:`identifiability` (probe B) — from seed
  keys (oracle cribs, perturbed truth), does pure best-improvement climbing reach
  the truth? If A passes but B fails, the basin is too small: you need richer
  seeds/cribs, and :func:`crib_floor` measures how many.
* :func:`crib_floor` (probe C) — sweep "letters locked to truth" and measure
  recovery rate, giving the crib-size floor a real attack must reach.
* :func:`damage_ladder` — score vs key damage, with ADJACENT-level separation
  (AUC): a healthy landscape degrades monotonically with discriminating steps; a
  cliff shows one huge step and flat noise past it, which is why "SA got close" is
  meaningless there.

All are generic over ``(key, neighbors, objective)`` callables — nothing here knows
what a cipher is.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .power import separation

Objective = Callable[[object], float]
Neighbors = Callable[[object], Iterable[object]]


@dataclass
class LocalMaxProbe:
    """Whether the true key is a local maximum under the search's move set."""

    is_local_max: bool
    true_score: float
    better_neighbors: int
    best_neighbor_score: float | None
    n_neighbors: int

    def summary(self) -> str:
        if self.is_local_max:
            return (
                f"truth is a local max over {self.n_neighbors} moves (score {self.true_score:.4f})"
            )
        return (
            f"truth is NOT a local max: {self.better_neighbors}/{self.n_neighbors} moves "
            f"score higher (best {self.best_neighbor_score:.4f} vs {self.true_score:.4f}) "
            "— the objective cannot identify the target; fix it before searching"
        )


def local_max_probe(true_key: object, neighbors: Neighbors, objective: Objective) -> LocalMaxProbe:
    """Probe A: is the TRUE key a local maximum of ``objective`` under ``neighbors``?

    Run on a plant with the key known. A failure here is a verdict about the
    *objective*, not the search: if a single move away from the truth scores higher,
    every hill climber will walk off the answer even when handed it, and any
    negative from that search is void.
    """
    true_score = float(objective(true_key))
    better = 0
    best: float | None = None
    n = 0
    for nb in neighbors(true_key):
        s = float(objective(nb))
        n += 1
        if best is None or s > best:
            best = s
        if s > true_score:
            better += 1
    return LocalMaxProbe(
        is_local_max=better == 0,
        true_score=true_score,
        better_neighbors=better,
        best_neighbor_score=best,
        n_neighbors=n,
    )


def best_improvement_climb(
    start: object,
    neighbors: Neighbors,
    objective: Objective,
    *,
    max_steps: int = 10_000,
) -> tuple[object, float, int]:
    """Deterministic steepest-ascent to convergence: ``(key, score, steps)``.

    The reference climber for landscape probes — no annealing, no restarts, no
    randomness, so what it measures is the landscape, not the search's luck.
    """
    key = start
    score = float(objective(key))
    for step in range(max_steps):
        best_nb, best_s = None, score
        for nb in neighbors(key):
            s = float(objective(nb))
            if s > best_s:
                best_nb, best_s = nb, s
        if best_nb is None:
            return key, score, step
        key, score = best_nb, best_s
    return key, score, max_steps


@dataclass
class Identifiability:
    """Joint verdict of probes A and B. ``verdict`` is one of:

    * ``"identifiable"`` — truth is a local max AND seeded climbs reach it.
    * ``"needs-richer-seeds"`` — truth is a local max but climbing from the given
      seeds does not reach it: the basin is smaller than the seed radius. Measure
      the required crib size with :func:`crib_floor`.
    * ``"dead-lane"`` — truth is not even a local max; the objective cannot
      identify the target and NO search improvement helps.
    """

    verdict: str
    probe_a: LocalMaxProbe
    reached: int
    seeds: int
    details: list[dict] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        return (
            f"{self.verdict}: probe A {'pass' if self.probe_a.is_local_max else 'FAIL'}, "
            f"probe B {self.reached}/{self.seeds} seeded climbs reached truth"
        )


def identifiability(
    true_key: object,
    neighbors: Neighbors,
    objective: Objective,
    seed_keys: Sequence[object],
    *,
    reached: Callable[[object], bool] | None = None,
    max_steps: int = 10_000,
) -> Identifiability:
    """Probes A and B on one plant: can this objective/move-set find this truth?

    ``seed_keys`` are starting points a real attack could plausibly reach (crib-
    derived keys, mostly-correct keys); ``reached`` decides whether a climb endpoint
    counts as the truth (default: equality with ``true_key``, or matching its score
    when the endpoint is a symmetric equivalent).
    """
    probe_a = local_max_probe(true_key, neighbors, objective)
    true_score = probe_a.true_score
    is_truth = reached or (lambda k: k == true_key or float(objective(k)) >= true_score)
    hits = 0
    details: list[dict] = []
    for seed in seed_keys:
        end, score, steps = best_improvement_climb(seed, neighbors, objective, max_steps=max_steps)
        ok = bool(is_truth(end))
        hits += ok
        details.append({"seed": seed, "end": end, "score": score, "steps": steps, "reached": ok})
    if not probe_a.is_local_max:
        verdict = "dead-lane"
    elif hits == len(seed_keys) and seed_keys:
        verdict = "identifiable"
    else:
        verdict = "needs-richer-seeds" if hits < len(seed_keys) else "identifiable"
    return Identifiability(
        verdict=verdict,
        probe_a=probe_a,
        reached=hits,
        seeds=len(seed_keys),
        details=details,
    )


def crib_floor(
    recover_at: Callable[[int, random.Random], bool],
    sizes: Sequence[int],
    *,
    trials: int = 10,
    threshold: float = 0.9,
    rng: random.Random | None = None,
) -> dict:
    """Probe C: sweep crib size (letters locked to truth) and measure recovery rate.

    ``recover_at(k, rng)`` runs one attack attempt with ``k`` true letters locked and
    returns whether the truth was recovered. Returns ``{curve, floor}`` where
    ``curve`` is ``[{size, rate}]`` and ``floor`` is the smallest size whose rate
    reaches ``threshold`` (None if none does). The floor is the honest "how much
    crib does a real attack need" number — quote it instead of "cribbing helps".
    """
    rng = rng or random.Random()
    curve: list[dict] = []
    floor: int | None = None
    for k in sizes:
        hits = sum(1 for _ in range(trials) if recover_at(k, rng))
        rate = hits / trials
        curve.append({"size": k, "rate": rate})
        if floor is None and rate >= threshold:
            floor = k
    return {"curve": curve, "floor": floor}


def damage_ladder(
    true_key: object,
    damage: Callable[[object, int, random.Random], object],
    objective: Objective,
    levels: Sequence[int],
    *,
    trials: int = 20,
    rng: random.Random | None = None,
) -> dict:
    """Score-vs-damage profile with adjacent-level separation — the cliff detector.

    ``damage(key, k, rng)`` returns a copy of ``key`` with ``k`` units of damage
    (e.g. ``k`` random element swaps). For each level the objective is sampled
    ``trials`` times; ``rungs`` reports mean/sd per level and ``adjacent`` the
    :func:`buttcrack.power.separation` AUC between consecutive levels.

    Read it like this: a CLIMBABLE landscape has adjacent AUCs well above 0.5 all
    the way down (each step of repair is visible to the objective). A CLIFF has one
    huge score drop at the first level and AUC ≈ 0.5 everywhere past it — beyond
    the cliff the objective is noise, so a search can only find the key by landing
    exactly on it, and "the SA reached 84% of the score" means nothing.
    ``cliff`` flags that shape: the first gap carries most of the total drop while
    deeper adjacent rungs stop separating.
    """
    rng = rng or random.Random()
    lv = list(levels)
    if lv and lv[0] != 0:
        lv = [0, *lv]
    scores: list[list[float]] = []
    for k in lv:
        if k == 0:
            scores.append([float(objective(true_key)) for _ in range(trials)])
        else:
            scores.append([float(objective(damage(true_key, k, rng))) for _ in range(trials)])
    rungs = []
    for k, ss in zip(lv, scores, strict=True):
        mean = sum(ss) / len(ss)
        sd = (sum((s - mean) ** 2 for s in ss) / len(ss)) ** 0.5
        rungs.append({"damage": k, "mean": mean, "sd": sd})
    adjacent = []
    for j in range(len(lv) - 1):
        sep = separation(scores[j], scores[j + 1])
        adjacent.append({"from": lv[j], "to": lv[j + 1], "auc": sep.auc, "z": sep.z})
    cliff = False
    if len(rungs) >= 3:
        total_drop = rungs[0]["mean"] - rungs[-1]["mean"]
        first_drop = rungs[0]["mean"] - rungs[1]["mean"]
        deeper = adjacent[1:]
        if total_drop > 0 and first_drop / total_drop >= 0.7 and deeper:
            cliff = all(a["auc"] < 0.7 for a in deeper)
    return {"rungs": rungs, "adjacent": adjacent, "cliff": cliff}
