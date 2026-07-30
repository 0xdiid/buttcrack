"""Structure-preserving null models and a look-elsewhere (multiplicity) correction.

A cryptanalytic "signal" — an elevated coset-IC at some period, a kappa spike at
some lag, a transposition arrangement that scores well — is only evidence if it
beats the **right** null. The right null destroys *exactly* the structure being
tested and preserves everything else; a null that also throws away an unrelated
structure manufactures false positives (any arrangement beats a null that has
already discarded the signal), and a null that preserves too much hides real ones.

This module supplies a small library of such nulls and one harness to test any
statistic against any of them:

* :func:`permutation` — reorder the whole message (preserves the exact letter
  multiset / composition, destroys all positional structure). The classic control.
* :func:`iid_empirical` / :func:`iid_uniform` — resample letters (destroys
  composition too); the weakest, most permissive controls.
* :func:`within_coset` — reorder only *within* each residue class mod ``p``
  (preserves every coset's letter multiset and hence coset-IC at ``p``; destroys
  the within-coset order). The honest null for "does the order inside a period-``p``
  arrangement carry a message?" — a plain shuffle is the *wrong* null there.
* :func:`block_shuffle` — permute whole ``b``-letter blocks (preserves within-block
  structure and every mod-``b`` coset; only teeth at periods that are proper
  multiples of ``b`` are testable against it).

Two further ideas the corpus kept re-deriving:

* **The honest max-over-search null is automatic.** If your observed number is the
  *maximum* over a search (a bank of transpositions, the best of many periods),
  pass a statistic that performs that same max internally — :func:`null_test` then
  maxes each null draw the same way, comparing a max to a distribution of maxes
  rather than to a mean. Nothing special is required; just don't hand it a
  pre-maximized scalar with a mean-based null.
* **Look-elsewhere correction.** When you scan a statistic over many hypotheses
  (periods, lags) and report the best, the per-hypothesis p-value is optimistic.
  :func:`scan_test` takes a *vector-valued* statistic and reports the multiplicity-
  corrected "is there ANY structure?" p-value (the null distribution of the max /
  of a family aggregate), alongside the per-hypothesis z-scores.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .windings import coset_preserving_shuffle

#: a null generator: given a sequence and an RNG, return a randomized copy that
#: preserves some structure and destroys the rest. Same length/element-type in, out.
NullFn = Callable[[Sequence, random.Random], Sequence]

#: a scalar statistic on a sequence (higher = more "signal", by convention).
Statistic = Callable[[Sequence], float]

#: a vector statistic: one value per hypothesis (e.g. per candidate period).
VectorStatistic = Callable[[Sequence], Sequence[float]]


def _same_type(template: Sequence, seq: list) -> Sequence:
    """Return ``seq`` shaped like ``template`` (join back to ``str`` if it was one)."""
    if isinstance(template, str):
        return "".join(seq)
    return list(seq)


# --- the null generators -----------------------------------------------------


def permutation(seq: Sequence, rng: random.Random) -> Sequence:
    """Reorder the whole sequence (preserves the exact letter multiset)."""
    out = rng.sample(list(seq), len(seq))
    return _same_type(seq, out)


def iid_empirical(seq: Sequence, rng: random.Random) -> Sequence:
    """Resample each position i.i.d. from the sequence's own letters (with replacement)."""
    pool = list(seq)
    out = [rng.choice(pool) for _ in pool]
    return _same_type(seq, out)


def iid_uniform(alphabet: Sequence | None = None) -> NullFn:
    """Null generator that resamples i.i.d. *uniformly* over ``alphabet``.

    With ``alphabet=None`` the distinct symbols observed in the sequence are used, so
    it works for A-Z text and for restricted-alphabet ciphertexts (ADFGX, etc.).
    """

    def _null(seq: Sequence, rng: random.Random) -> Sequence:
        alpha: list[Any] = list(alphabet) if alphabet is not None else list(set(seq))
        out = [rng.choice(alpha) for _ in range(len(seq))]
        return _same_type(seq, out)

    return _null


def within_coset(mod: int) -> NullFn:
    """Null generator that shuffles only *within* each residue class mod ``mod``.

    Preserves every coset's exact letter multiset (so coset-IC at ``mod`` is invariant)
    and randomizes the within-coset order — the honest null for testing whether the
    order inside a period-``mod`` arrangement is structured. Wraps
    :func:`buttcrack.windings.coset_preserving_shuffle`.
    """
    if mod < 1:
        raise ValueError(f"mod must be >= 1, got {mod}")

    def _null(seq: Sequence, rng: random.Random) -> Sequence:
        out = coset_preserving_shuffle(seq, mod, rng=rng)
        return _same_type(seq, out)

    return _null


def block_shuffle(block: int) -> NullFn:
    """Null generator that permutes whole ``block``-letter blocks (keeps within-block order).

    Preserves within-block structure and every mod-``block`` coset; use it to test
    whether a period that is a *proper multiple* of ``block`` reflects real positional
    alignment or merely the block composition. A trailing partial block is left in place.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")

    def _null(seq: Sequence, rng: random.Random) -> Sequence:
        n = len(seq)
        nb = n // block
        blocks = [list(seq[i * block : (i + 1) * block]) for i in range(nb)]
        order = list(range(nb))
        rng.shuffle(order)
        out: list = []
        for i in order:
            out.extend(blocks[i])
        out.extend(seq[nb * block :])
        return _same_type(seq, out)

    return _null


def null_is_degenerate(
    seq: Sequence,
    statistic: Statistic,
    null_fn: NullFn,
    *,
    probes: int = 8,
    rng: random.Random | None = None,
    rel_tol: float = 1e-12,
) -> bool:
    """Is ``null_fn`` invisible to ``statistic`` — i.e. the wrong null for it?

    The null-selection rule: an honest null preserves what the objective is GIVEN and
    destroys what it puts UNDER TEST. Get it backwards and the null preserves the very
    thing being tested — every draw returns the statistic's observed value bit-identically,
    the null distribution is a single point, and the observation can never beat it. The
    test then reports "not significant" forever, which reads exactly like a real negative.
    (The canonical instance: a coset-preserving shuffle under a statistic that is a
    function of coset *multisets* — merged-IC, coset-IC — which the shuffle preserves.)

    This draws ``probes`` nulls and returns True when every draw reproduces the observed
    statistic (within ``rel_tol`` relative tolerance). Run it before trusting any
    ``p ≈ 1`` from :func:`null_test`; a degenerate pairing needs a *more destructive*
    null (e.g. full :func:`permutation`), not more trials.
    """
    rng = rng or random.Random()
    obs = float(statistic(seq))
    scale = max(abs(obs), 1.0)
    for _ in range(probes):
        draw = float(statistic(null_fn(seq, rng)))
        if abs(draw - obs) > rel_tol * scale:
            return False
    return True


# --- the harness -------------------------------------------------------------


@dataclass
class NullResult:
    """Outcome of testing an observed statistic against a null distribution.

    ``p_value`` is the empirical tail probability with add-one smoothing
    ``(n_extreme + 1) / (trials + 1)`` — the honest small-sample estimate that never
    reports ``p = 0`` from a finite number of draws. ``z`` is the parametric
    ``(obs - mean) / sd`` for a quick read; trust ``p_value`` for the verdict.
    """

    obs: float
    null_mean: float
    null_sd: float
    z: float
    p_value: float
    n_extreme: int
    trials: int
    rank: int  # 1 = obs is the most extreme value seen (incl. obs itself)
    alternative: str
    null_samples: list[float] = field(default_factory=list, repr=False)
    degenerate: bool = False

    @property
    def significant(self) -> bool:
        """Convenience: ``p_value < 0.05``. Calibrate the threshold to your search size."""
        return self.p_value < 0.05

    def summary(self) -> str:
        base = (
            f"obs={self.obs:.4f} vs null {self.null_mean:.4f}±{self.null_sd:.4f} "
            f"z={self.z:+.2f} p={self.p_value:.4f} ({self.alternative}) "
            f"rank={self.rank}/{self.trials + 1}"
        )
        if self.degenerate:
            base += " [DEGENERATE null: statistic invariant under it — pick a more destructive null]"
        return base


def _tail(obs: float, samples: list[float], alternative: str) -> tuple[int, int]:
    """Count null draws at least as extreme as ``obs``; return (n_extreme, rank)."""
    if alternative == "greater":
        n_extreme = sum(1 for s in samples if s >= obs)
    elif alternative == "less":
        n_extreme = sum(1 for s in samples if s <= obs)
    elif alternative == "two-sided":
        center = sum(samples) / len(samples) if samples else obs
        d = abs(obs - center)
        n_extreme = sum(1 for s in samples if abs(s - center) >= d)
    else:
        raise ValueError(f"alternative must be greater/less/two-sided, got {alternative!r}")
    rank = n_extreme + 1  # obs ranked among {obs} ∪ samples
    return n_extreme, rank


def null_test(
    seq: Sequence,
    statistic: Statistic,
    null_fn: NullFn,
    *,
    trials: int = 1000,
    rng: random.Random | None = None,
    alternative: str = "greater",
    keep_samples: bool = False,
) -> NullResult:
    """Test ``statistic(seq)`` against ``trials`` draws from ``null_fn``.

    ``null_fn`` is one of this module's generators (e.g. ``within_coset(7)``,
    ``permutation``). Because each null draw is scored by the *same* ``statistic``, an
    honest max-over-search test needs nothing special: make ``statistic`` perform the
    search's max internally and every null draw is maxed the same way.

    ``alternative`` picks the tail: ``"greater"`` (default; the statistic is a signal
    that should exceed the null), ``"less"``, or ``"two-sided"``.

    The result carries ``degenerate=True`` when every null draw reproduced the observed
    value — the statistic is *invariant* under this null (it preserves exactly what was
    meant to be destroyed), so the ``p ≈ 1`` is an artifact of the pairing, not evidence.
    See :func:`null_is_degenerate` for the selection rule.
    """
    rng = rng or random.Random()
    obs = float(statistic(seq))
    samples = [float(statistic(null_fn(seq, rng))) for _ in range(trials)]
    scale = max(abs(obs), 1.0)
    degenerate = bool(samples) and all(abs(s - obs) <= 1e-12 * scale for s in samples)
    mean = sum(samples) / trials if trials else obs
    var = sum((s - mean) ** 2 for s in samples) / trials if trials else 0.0
    sd = math.sqrt(var)
    if sd > 0:
        z = (obs - mean) / sd
    else:
        z = 0.0 if obs == mean else math.copysign(float("inf"), obs - mean)
    n_extreme, rank = _tail(obs, samples, alternative)
    p_value = (n_extreme + 1) / (trials + 1)
    return NullResult(
        obs=obs,
        null_mean=mean,
        null_sd=sd,
        z=z,
        p_value=p_value,
        n_extreme=n_extreme,
        trials=trials,
        rank=rank,
        alternative=alternative,
        null_samples=samples if keep_samples else [],
        degenerate=degenerate,
    )


@dataclass
class ScanResult:
    """Look-elsewhere-corrected result of scanning a vector statistic over hypotheses.

    ``per_hypothesis`` pairs each hypothesis label with its raw value and its
    per-hypothesis z against the null (uncorrected — optimistic). ``scan_max_p`` is
    the honest "is there ANY structure across the scan?" p-value: the probability a
    null draw's *best* hypothesis reaches the observed best. ``family_max_p`` applies
    the same correction to a family aggregate (mean over each base period's multiples),
    the right test when the signal is spread across harmonics rather than one tooth.
    """

    hypotheses: list
    obs: list[float]
    z: list[float]
    argmax: object
    z_max: float
    scan_max_p: float
    family_max_p: float | None
    trials: int

    @property
    def per_hypothesis(self) -> list[tuple]:
        return list(zip(self.hypotheses, self.obs, self.z, strict=True))

    def summary(self) -> str:
        fam = "n/a" if self.family_max_p is None else f"{self.family_max_p:.4f}"
        return (
            f"argmax={self.argmax} z_max={self.z_max:+.2f} "
            f"scan-max p={self.scan_max_p:.4f} family-max p={fam} "
            f"(either p>0.05 ⇒ no structure across the scan)"
        )


def scan_test(
    seq: Sequence,
    stat_vec: VectorStatistic,
    null_fn: NullFn,
    hypotheses: Sequence,
    *,
    trials: int = 1000,
    rng: random.Random | None = None,
    family: bool = False,
) -> ScanResult:
    """Scan a per-hypothesis statistic and correct for the multiplicity of the search.

    ``stat_vec(seq)`` returns one value per entry of ``hypotheses`` (e.g. coset-IC at
    each candidate period). Reporting the best of those overstates significance; this
    builds the null distribution of the *maximum standardized* value across the scan
    and returns ``scan_max_p`` — the fraction of null draws whose best hypothesis
    matches or beats the observed best. Set ``family=True`` to also compute a
    ``family_max_p`` over base periods (each base scored by the mean of its multiples'
    z-scores within ``hypotheses``), the correction to use when structure spreads across
    a period and its harmonics.
    """
    rng = rng or random.Random()
    hyps = list(hypotheses)
    obs = [float(v) for v in stat_vec(seq)]
    if len(obs) != len(hyps):
        raise ValueError(f"stat_vec returned {len(obs)} values for {len(hyps)} hypotheses")
    null_rows = [[float(v) for v in stat_vec(null_fn(seq, rng))] for _ in range(trials)]
    ncol = len(hyps)
    mu = [sum(r[j] for r in null_rows) / trials if trials else obs[j] for j in range(ncol)]
    sd = [
        math.sqrt(sum((r[j] - mu[j]) ** 2 for r in null_rows) / trials) if trials else 0.0
        for j in range(ncol)
    ]

    def _z(vals: list[float]) -> list[float]:
        return [(vals[j] - mu[j]) / sd[j] if sd[j] > 0 else 0.0 for j in range(ncol)]

    z_obs = _z(obs)
    z_max = max(z_obs) if z_obs else 0.0
    argmax = hyps[z_obs.index(z_max)] if z_obs else None
    null_maxes = [max(_z(r)) for r in null_rows] if null_rows else []
    scan_ge = sum(1 for m in null_maxes if m >= z_max)
    scan_max_p = (scan_ge + 1) / (trials + 1)

    family_max_p: float | None = None
    if family:
        idx = {h: j for j, h in enumerate(hyps)}
        top = max(hyps)
        bases = [b for b in hyps if isinstance(b, int) and 2 * b <= top]

        def _fam(zrow: list[float]) -> float:
            best = 0.0
            for b in bases:
                mults = [idx[m] for m in range(b, top + 1, b) if m in idx]
                if mults:
                    best = max(best, sum(zrow[m] for m in mults) / len(mults))
            return best

        fam_obs = _fam(z_obs)
        fam_ge = sum(1 for r in null_rows if _fam(_z(r)) >= fam_obs)
        family_max_p = (fam_ge + 1) / (trials + 1)

    return ScanResult(
        hypotheses=hyps,
        obs=obs,
        z=z_obs,
        argmax=argmax,
        z_max=z_max,
        scan_max_p=scan_max_p,
        family_max_p=family_max_p,
        trials=trials,
    )
