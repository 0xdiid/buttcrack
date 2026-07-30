"""Objective power calibration — does a statistic have the resolving power you assume?

:mod:`buttcrack.validate` proves an *attack* recovers its own structure. This module
asks the prior question about the *objective* the attack maximizes: can it tell signal
from noise at all, and at what signal strength does it start to? A negative result on
real ciphertext only means something if the objective would have lit up on a plant —
and an objective that scores every key the same (a "powerless" objective) can never
find one no matter how good the search.

Three primitives, all cipher-agnostic (you supply the scoring callable and the
sample-makers):

* :func:`separation` — given a population of *signal* scores and a population of *null*
  scores, how cleanly are they separated: the z of the signal above the null
  distribution, Cohen's d, and the rank AUC ``P(signal > null)``.
* :func:`powerless_objective` — evaluate an objective over many inputs and flag it if it
  barely varies (constant / key-invariant). The commonest silent failure: a "solver"
  whose objective is flat, so its search is a random walk and its negative is vacuous.
* :func:`power_curve` — sweep a *signal strength* knob and report separation (and optional
  recovery rate) at each level, so you can see the SNR at which the objective — and the
  search built on it — actually turns on. :func:`dilute_plaintext` is a ready-made knob.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .text import ALPHABET, only_letters

Objective = Callable[[object], float]


@dataclass
class Separation:
    """How cleanly a signal population sits apart from a null population.

    ``z`` is ``(mean_signal - mean_null) / sd_null`` — how many null standard deviations
    the signal's mean sits above the null (the honest "true key vs random-key
    distribution" separation; when ``signal`` is a single value this is exactly its
    z-score against the null). ``cohens_d`` is the pooled-sd standardized mean gap.
    ``auc`` is ``P(signal > null)`` over all pairs (0.5 = no separation, 1.0 = perfect).
    """

    z: float
    cohens_d: float
    auc: float
    mean_signal: float
    mean_null: float
    sd_signal: float
    sd_null: float
    n_signal: int
    n_null: int

    @property
    def separated(self) -> bool:
        """A practical bar: signal mean ≥ 3 null-sigma above null and AUC ≥ 0.95."""
        return self.z >= 3.0 and self.auc >= 0.95

    def summary(self) -> str:
        return (
            f"z={self.z:+.2f} d={self.cohens_d:+.2f} auc={self.auc:.3f} "
            f"(signal {self.mean_signal:.4f}±{self.sd_signal:.4f} vs "
            f"null {self.mean_null:.4f}±{self.sd_null:.4f})"
        )


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, math.sqrt(var)


def _auc(signal: Sequence[float], null: Sequence[float]) -> float:
    """Rank AUC = P(signal > null) with 0.5 credit for ties (Mann-Whitney)."""
    if not signal or not null:
        return 0.5
    wins = 0.0
    for s in signal:
        for x in null:
            if s > x:
                wins += 1.0
            elif s == x:
                wins += 0.5
    return wins / (len(signal) * len(null))


def separation(signal_scores: Sequence[float], null_scores: Sequence[float]) -> Separation:
    """Quantify how separable a *signal* score population is from a *null* one.

    ``signal_scores`` may be a single value (e.g. the true key's objective) or a
    population; ``null_scores`` is the reference distribution (e.g. objective over random
    keys). Higher ``z`` / ``auc`` = the objective genuinely resolves signal from noise.
    """
    sig = [float(x) for x in signal_scores]
    nul = [float(x) for x in null_scores]
    mean_s, sd_s = _mean_sd(sig)
    mean_n, sd_n = _mean_sd(nul)
    z = (mean_s - mean_n) / sd_n if sd_n > 0 else (0.0 if mean_s == mean_n else math.inf)
    pooled = math.sqrt((sd_s**2 + sd_n**2) / 2) if (sd_s or sd_n) else 0.0
    d = (mean_s - mean_n) / pooled if pooled > 0 else (0.0 if mean_s == mean_n else math.inf)
    return Separation(
        z=z,
        cohens_d=d,
        auc=_auc(sig, nul),
        mean_signal=mean_s,
        mean_null=mean_n,
        sd_signal=sd_s,
        sd_null=sd_n,
        n_signal=len(sig),
        n_null=len(nul),
    )


@dataclass
class PowerlessReport:
    """Verdict on whether an objective varies enough across inputs to have any power."""

    powerless: bool
    sd: float
    value_range: float
    distinct: int
    n: int
    mean: float

    def summary(self) -> str:
        verdict = "POWERLESS (flat objective)" if self.powerless else "varies"
        return (
            f"{verdict}: sd={self.sd:.3e} range={self.value_range:.3e} "
            f"distinct={self.distinct}/{self.n}"
        )


def powerless_objective(
    objective: Objective,
    inputs: Sequence[object],
    *,
    rel_tol: float = 1e-6,
) -> PowerlessReport:
    """Flag an objective that is (near-)constant across ``inputs`` — it cannot discriminate.

    Evaluates ``objective`` on every input and reports its spread. ``powerless`` is True
    when the standard deviation is within ``rel_tol`` of the mean magnitude (or the values
    are all identical): such an objective assigns essentially the same score to every
    candidate, so any search over it is a blind walk and any negative it yields is empty.
    Use it on random *keys* (key-invariance) or random *plaintexts* alike.
    """
    vals = [float(objective(x)) for x in inputs]
    if not vals:
        return PowerlessReport(True, 0.0, 0.0, 0, 0, 0.0)
    mean, sd = _mean_sd(vals)
    vrange = max(vals) - min(vals)
    distinct = len({round(v, 12) for v in vals})
    scale = max(abs(mean), 1.0)
    powerless = distinct <= 1 or sd <= rel_tol * scale
    return PowerlessReport(
        powerless=powerless,
        sd=sd,
        value_range=vrange,
        distinct=distinct,
        n=len(vals),
        mean=mean,
    )


@dataclass
class PowerPoint:
    """Separation (and optional recovery rate) of an objective at one signal strength."""

    strength: float
    sep: Separation
    recover_rate: float | None


def power_curve(
    make_signal: Callable[[float, random.Random], object],
    objective: Objective,
    strengths: Sequence[float],
    *,
    trials: int = 20,
    rng: random.Random | None = None,
    null_maker: Callable[[random.Random], object] | None = None,
    recover_fn: Callable[[object], bool] | None = None,
) -> list[PowerPoint]:
    """Sweep a signal-strength knob and report the objective's separation at each level.

    ``make_signal(strength, rng)`` produces one input carrying signal of the given
    strength (e.g. a plaintext diluted to fraction ``strength`` English, then encrypted);
    ``objective`` scores an input. For each strength, ``trials`` signal inputs are scored
    against ``trials`` null inputs (``null_maker`` — default ``make_signal(0.0, rng)``) and
    a :class:`Separation` is computed. Pass ``recover_fn`` to also record the fraction of
    signal inputs an actual solver recovers, so you can see the strength where separation
    *and* recovery turn on together.
    """
    rng = rng or random.Random()
    nullmk = null_maker or (lambda r: make_signal(0.0, r))
    curve: list[PowerPoint] = []
    for s in strengths:
        sig_scores = [float(objective(make_signal(s, rng))) for _ in range(trials)]
        null_scores = [float(objective(nullmk(rng))) for _ in range(trials)]
        recover = None
        if recover_fn is not None:
            hits = sum(1 for _ in range(trials) if recover_fn(make_signal(s, rng)))
            recover = hits / trials
        curve.append(
            PowerPoint(
                strength=s,
                sep=separation(sig_scores, null_scores),
                recover_rate=recover,
            )
        )
    return curve


def expected_null_max(mean: float, sd: float, n: int) -> float:
    """Expected maximum of ``n`` independent null draws: ``mean + sd * sqrt(2 ln n)``.

    The best score over a candidate bank grows with the bank even when every candidate
    is noise. This is the standard Gaussian-tail growth rate — compute it BEFORE
    spending the compute, because it is the bar a true signal must clear at that bank
    size, not the single-draw null mean.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return mean
    return mean + sd * math.sqrt(2.0 * math.log(n))


@dataclass
class BankScaling:
    """What scaling a candidate bank does to a detection margin.

    ``margin_before``/``margin_after`` are the true signal's clearance (in null sd
    units) over the expected null max at each bank size. ``survives`` is whether any
    clearance is left after scaling — if not, the bigger search DESTROYS the power
    that justified the smaller one, and running it buys noise.
    """

    signal_z: float
    n_before: int
    n_after: int
    null_max_z_before: float
    null_max_z_after: float
    margin_before: float
    margin_after: float

    @property
    def survives(self) -> bool:
        return self.margin_after > 0

    def summary(self) -> str:
        verdict = "survives" if self.survives else "power DESTROYED by scaling"
        return (
            f"signal z={self.signal_z:+.2f}: margin {self.margin_before:+.2f} at "
            f"n={self.n_before:,} -> {self.margin_after:+.2f} at n={self.n_after:,} "
            f"({verdict})"
        )


def bank_scaling(signal_z: float, n_before: int, n_after: int) -> BankScaling:
    """Check whether a detection margin survives scaling the candidate bank.

    A test justified at ``n_before`` candidates (signal sits ``signal_z`` null-sd above
    the null mean, comfortably over the expected null max) can be *worthless* at
    ``n_after``: the expected null max grows like ``sqrt(2 ln n)``, so scaling a
    200-candidate test to 7,000 can eat the entire margin. Run this before scaling —
    if ``survives`` is False, more compute makes the answer worse, not better.
    """
    z_max_before = expected_null_max(0.0, 1.0, n_before)
    z_max_after = expected_null_max(0.0, 1.0, n_after)
    return BankScaling(
        signal_z=signal_z,
        n_before=n_before,
        n_after=n_after,
        null_max_z_before=z_max_before,
        null_max_z_after=z_max_after,
        margin_before=signal_z - z_max_before,
        margin_after=signal_z - z_max_after,
    )


@dataclass
class LabelingPower:
    """Two-sample separability vs held-out one-sample labeling accuracy.

    These answer DIFFERENT questions and conflating them has produced real false
    closures: a statistic can separate two *populations* cleanly (high ``auc``) while
    still being unable to label ONE ciphertext (``heldout_accuracy`` near 0.5),
    because a single draw straddles the threshold that population means define.
    Quote ``heldout_accuracy`` when the claim is about one target text.
    """

    auc: float
    heldout_accuracy: float
    threshold: float
    n_signal: int
    n_null: int

    def summary(self) -> str:
        return (
            f"2-sample auc={self.auc:.3f} vs 1-sample held-out accuracy="
            f"{self.heldout_accuracy:.3f} (threshold {self.threshold:.4f}) — "
            "quote the accuracy when labeling ONE text"
        )


def labeling_power(
    signal_scores: Sequence[float],
    null_scores: Sequence[float],
    *,
    folds: int = 5,
) -> LabelingPower:
    """Measure both population separability AND single-sample labeling accuracy.

    ``folds``-fold cross-validation: each fold trains a midpoint threshold on the
    other folds' class means and labels the held-out samples. The returned
    ``threshold`` is the full-data midpoint (for labeling future samples);
    ``heldout_accuracy`` is the honest estimate of how often that rule labels one
    unseen sample correctly.
    """
    sig = [float(x) for x in signal_scores]
    nul = [float(x) for x in null_scores]
    if len(sig) < folds or len(nul) < folds:
        raise ValueError(f"need at least {folds} samples per class for {folds}-fold CV")
    correct = 0
    total = 0
    for k in range(folds):
        sig_tr = [x for i, x in enumerate(sig) if i % folds != k]
        nul_tr = [x for i, x in enumerate(nul) if i % folds != k]
        thr = (sum(sig_tr) / len(sig_tr) + sum(nul_tr) / len(nul_tr)) / 2
        sign = 1.0 if sum(sig_tr) / len(sig_tr) >= sum(nul_tr) / len(nul_tr) else -1.0
        for x in (x for i, x in enumerate(sig) if i % folds == k):
            correct += sign * (x - thr) > 0
            total += 1
        for x in (x for i, x in enumerate(nul) if i % folds == k):
            correct += sign * (x - thr) < 0
            total += 1
    thr_full = (sum(sig) / len(sig) + sum(nul) / len(nul)) / 2
    return LabelingPower(
        auc=_auc(sig, nul),
        heldout_accuracy=correct / total if total else 0.5,
        threshold=thr_full,
        n_signal=len(sig),
        n_null=len(nul),
    )


def family_power(
    generators: dict[str, Callable[[random.Random], object]],
    statistic: Callable[[object], float],
    null_fn: Callable[[object, random.Random], object],
    *,
    plants: int = 20,
    trials: int = 100,
    z_threshold: float = 3.0,
    rng: random.Random | None = None,
) -> dict[str, dict[str, float]]:
    """Convert a statistic into a classifier with measured per-family power.

    For each named generator family, draw ``plants`` synthetic instances, z-score
    each instance's statistic against ``trials`` draws of ``null_fn`` on that same
    instance (the matched null), and report ``power`` = the fraction of plants with
    ``z >= z_threshold`` plus the mean z. A statistic is only usable as a detector
    for the families where ``power`` is high — and a null on a family where power is
    low is SILENT, not negative (see :meth:`buttcrack.evidence.Finding.with_power`).

    This is the calibration to run for EVERY family you intend to rule in or out,
    before the statistic touches the real target.
    """
    rng = rng or random.Random()
    out: dict[str, dict[str, float]] = {}
    for name, gen in generators.items():
        zs: list[float] = []
        for _ in range(plants):
            inst = gen(rng)
            obs = float(statistic(inst))
            draws = [float(statistic(null_fn(inst, rng))) for _ in range(trials)]
            mu, sd = _mean_sd(draws)
            zs.append((obs - mu) / sd if sd > 0 else 0.0)
        mean_z, _ = _mean_sd(zs)
        out[name] = {
            "mean_z": mean_z,
            "power": sum(1 for z in zs if z >= z_threshold) / len(zs) if zs else 0.0,
            "plants": float(len(zs)),
        }
    return out


def dilute_plaintext(
    plaintext: str,
    strength: float,
    rng: random.Random,
    *,
    alphabet: str = ALPHABET,
) -> str:
    """Return ``plaintext`` with a ``strength`` fraction of letters kept, the rest randomized.

    A ready-made signal-strength knob for :func:`power_curve`: ``strength=1.0`` is the
    clean plaintext, ``0.0`` is pure noise, and intermediate values interpolate the SNR by
    replacing each letter with a uniform random one with probability ``1 - strength``.
    """
    letters = only_letters(plaintext)
    s = max(0.0, min(1.0, strength))
    return "".join(ch if rng.random() < s else rng.choice(alphabet) for ch in letters)
