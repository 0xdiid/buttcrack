"""Detect a linear relation among the positions of fixed n-grams.

Some ciphers expand each plaintext symbol into a fixed-length *n-gram* of ciphertext
letters, where the plaintext channel is recoverable as a fixed **linear combination** of
the n-gram's positions in a keyed-alphabet index space and the remaining positions are
free homophones / nulls.  The Delastelle three-square (**tri-square**) family is the
classic case: a plaintext letter becomes a trigraph whose deterministic intersection
letter is one fixed combination of the flanking row/column homophones.

Such a cipher hides from every standard test: each single positional stream looks flat
(IoC ~ random) because the homophones spread the frequencies, and the whole text shows no
period and no repeats — yet a specific combination ``sum_k coef[k]*index(C[k]) (mod 26)``
reconstructs the plaintext channel, whose IoC lifts toward the language floor.  This
module scans every small-integer combination, in each candidate alphabet, and
null-calibrates by shuffling each positional stream independently (which preserves the
per-position histograms but destroys any genuine cross-position relation).

A hit means: ``combine(text, coef, alphabet, n=n)`` is your recovered plaintext channel
(still possibly under a further mono / periodic / transposition layer — feed it to the
substitution, quagmire, or transsub solvers next).

Public API
----------
``scan(text, *, n=3, ...) -> dict``      rank candidate relations by null-calibrated z / p
``combine(text, coef, alphabet, n=3)``    extract the channel for a chosen relation
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

from .keysources import _alphabet
from .text import only_letters


def _index_stream(letters: str, alphabet: str) -> list[int]:
    idx = {c: i for i, c in enumerate(alphabet)}
    return [idx[c] for c in letters]


def _ioc_indices(seq: list[int]) -> float:
    n = len(seq)
    if n < 2:
        return 0.0
    counts = Counter(seq)
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def _canonical_coeffs(n: int, coeffs) -> list[tuple[int, ...]]:
    """Non-trivial coefficient vectors, deduped by overall sign.

    Negating every coefficient leaves the combined stream's IoC unchanged (negation mod
    26 is a bijection), so we keep only the lexicographically-smaller of ``(c, -c)`` and
    drop vectors with fewer than two non-zero entries (those are single positions, which
    are the ordinary — already flat — streams).
    """
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for c in itertools.product(coeffs, repeat=n):
        if sum(1 for x in c if x) < 2:
            continue
        neg = tuple(-x for x in c)
        key = min(c, neg)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def combine(text: str, coef, alphabet: str = "KRYPTOS", *, n: int | None = None) -> str:
    """Extract the linear-combination channel ``sum_k coef[k]*index(C[k]) mod 26``.

    Groups ``text`` into consecutive n-grams (``n = len(coef)`` unless given), takes each
    letter's index in ``alphabet``, applies the integer combination per group, and renders
    the result back through ``alphabet``.  Returns one letter per complete n-gram.
    """
    coef = list(coef)
    n = n or len(coef)
    if len(coef) != n:
        raise ValueError("len(coef) must equal n")
    alpha = _alphabet(alphabet)
    letters = only_letters(text)
    groups = len(letters) // n
    idx = _index_stream(letters, alpha)
    out = []
    for g in range(groups):
        base = g * n
        v = sum(coef[k] * idx[base + k] for k in range(n)) % 26
        out.append(alpha[v])
    return "".join(out)


def scan(
    text: str,
    *,
    n: int = 3,
    coeffs=(-1, 0, 1),
    alphabets=("KRYPTOS", "STD"),
    samples: int = 2000,
    seed: int = 0,
    top: int = 8,
) -> dict:
    """Scan for an elevated linear relation among the positions of fixed n-grams.

    Parameters
    ----------
    text : ciphertext (non-letters ignored); grouped into consecutive ``n``-grams.
    n : n-gram size (3 for the trigraph / tri-square family).
    coeffs : integer coefficients to try per position (default ``{-1,0,1}`` — the
        homophonic-sum case; the family's relations use unit coefficients).
    alphabets : keyed alphabets to test the index space in (name or 26-letter permutation).
    samples : shuffle-null replicates for calibration.
    seed : RNG seed (reproducible).
    top : how many ranked candidates to return.

    Returns ``{n, groups, floor, candidates: [...], verdict}`` where each candidate has
    ``{alphabet, coef, ioc, z, p}``.  ``z`` is versus the shuffle null of that combination;
    ``p`` is *search-aware* (fraction of null runs whose best-over-all-candidates IoC
    reached this candidate's IoC), so it already accounts for trying many combinations.
    A real relation shows ``p`` near 0 with ``ioc`` well above ``floor`` (the null mean).
    """
    letters = only_letters(text)
    groups = len(letters) // n
    if groups < 4:
        return {"n": n, "groups": groups, "candidates": [], "verdict": "too short"}

    rng = random.Random(seed)
    cand_coeffs = _canonical_coeffs(n, coeffs)

    # Per-alphabet positional streams (position k -> list over groups).
    per_alpha = {}
    for aname in alphabets:
        alpha = _alphabet(aname)
        idx = _index_stream(letters[: groups * n], alpha)
        pos = [[idx[g * n + k] for g in range(groups)] for k in range(n)]
        per_alpha[aname] = pos

    def combined_ioc(pos, coef) -> float:
        seq = [sum(coef[k] * pos[k][g] for k in range(n)) % 26 for g in range(groups)]
        return _ioc_indices(seq)

    # Observed IoC for every (alphabet, coef).
    observed = []
    for aname, pos in per_alpha.items():
        for coef in cand_coeffs:
            observed.append((combined_ioc(pos, coef), aname, coef))

    # Shuffle null: independently permute each positional stream (preserves per-position
    # histograms, destroys cross-position structure). Track, per replicate, the best IoC
    # over all candidates (search-aware) and accumulate per-candidate mean/var for z.
    key_index = {(a, c): i for i, (_, a, c) in enumerate(observed)}
    sums = [0.0] * len(observed)
    sumsq = [0.0] * len(observed)
    null_max = []
    for _ in range(samples):
        shuffled = {}
        for aname, pos in per_alpha.items():
            sp = []
            for stream in pos:
                s = stream[:]
                rng.shuffle(s)
                sp.append(s)
            shuffled[aname] = sp
        best = 0.0
        for aname, pos in shuffled.items():
            for coef in cand_coeffs:
                v = combined_ioc(pos, coef)
                i = key_index[(aname, coef)]
                sums[i] += v
                sumsq[i] += v * v
                if v > best:
                    best = v
        null_max.append(best)

    null_max.sort()
    floor = sum(sums) / (len(sums) * samples) if samples else 0.0

    def searchaware_p(ioc: float) -> float:
        # Matched-count empirical p: the fraction of replicates whose BEST-over-all-candidates IoC
        # reached `ioc`. This is already multiple-testing correct BY CONSTRUCTION — every replicate
        # takes the max over the *same* candidate set that was searched on the real text
        # (len(coeffs) x len(alphabets) functionals), so the null scales with the actual trial
        # count rather than a fixed correction. Laplace (+1) smoothing keeps it a proper, non-zero
        # empirical p even when
        # no shuffle reaches the observed IoC (a search that "never" beats it is p<=1/(N+1), not 0).
        if not null_max:
            return 1.0
        lo, hi = 0, len(null_max)
        while lo < hi:
            mid = (lo + hi) // 2
            if null_max[mid] < ioc:
                lo = mid + 1
            else:
                hi = mid
        return (len(null_max) - lo + 1) / (len(null_max) + 1)

    observed.sort(reverse=True)
    candidates = []
    for ioc, aname, coef in observed[:top]:
        i = key_index[(aname, coef)]
        mean = sums[i] / samples
        var = max(sumsq[i] / samples - mean * mean, 1e-12)
        z = (ioc - mean) / math.sqrt(var)
        candidates.append(
            {
                "alphabet": aname,
                "coef": list(coef),
                "ioc": round(ioc, 4),
                "z": round(z, 2),
                "p": round(searchaware_p(ioc), 5),
            }
        )

    top_cand = candidates[0] if candidates else None
    if top_cand and top_cand["p"] < 0.01 and top_cand["ioc"] > floor + 0.008:
        verdict = (
            f"relation found: {top_cand['alphabet']} coef={top_cand['coef']} "
            f"(IoC {top_cand['ioc']} vs floor {round(floor, 4)}, p={top_cand['p']}); "
            f"combine() this channel, then solve its residual sub/transposition"
        )
    else:
        verdict = "no elevated linear relation (not a linear homophonic-expansion cipher)"

    return {
        "n": n,
        "groups": groups,
        "floor": round(floor, 4),
        "candidates": candidates,
        "verdict": verdict,
    }
