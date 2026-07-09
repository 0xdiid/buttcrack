"""Joint transposition + periodic-substitution solver — a dual nested hill-climber
with simulated annealing, scoring the FINAL plaintext (entropy-normalized n-grams).

This is the AZdecrypt-style "outer climber optimizes the transposition, inner solver
fixes the substitution, score the resulting plaintext" approach. Unlike buttcrack's
``transsub`` (which anneals the *reveal-IoC* of the intermediate de-substituted stream —
a signal that only appears when the column order is already nearly perfect), this
optimizes readable English directly, so it climbs a smooth gradient toward the solution
even when no single-stage statistic isolates the transposition.

Layer orders:
  - ``inner`` (transposition OUTER, substitution INNER):  PT = desub(uncolumnar(CT))
  - ``outer`` (substitution OUTER, transposition INNER):  PT = uncolumnar(desub(CT))

The transposition is a (possibly incomplete) keyed columnar of a given width; the
substitution is a periodic shift cipher (Vigenere/Beaufort) in a keyed alphabet
(KRYPTOS / standard / custom). The inner substitution is solved optimally per outer
state by best-shift-per-residue-class, so each outer move is scored on its true best
plaintext.
"""
from __future__ import annotations

import math
import random
from collections import Counter

from .scoring import get_scorer, resolve_scorer, letter_entropy, ENGLISH_MONOGRAM_FREQ
from .text import only_letters
from .ciphers.columnar import _decode_letters as _col_decode
from .ciphers import incomplete_columnar as _icol

KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
STANDARD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EXPECT = [ENGLISH_MONOGRAM_FREQ[chr(65 + i)] for i in range(26)]


def keyed_alphabet(keyword: str) -> str:
    """Standard keyed alphabet from a keyword (dedup + remaining A-Z)."""
    seen: list[str] = []
    for ch in (keyword + STANDARD):
        if ch not in seen:
            seen.append(ch)
    return "".join(seen)


def _alphabet(spec: str) -> str:
    if spec.upper() in ("KRY", "KRYPTOS"):
        return KRYPTOS
    if spec.upper() in ("STD", "STANDARD"):
        return STANDARD
    return keyed_alphabet(spec.upper())


def _best_shifts(stream_idx: list[int], period: int, variant: str, expect: list[float]) -> list[int]:
    """Best shift per residue class i%period by min chi-squared vs English monograms.

    ``expect[j]`` is the expected English frequency of the letter at *alphabet index* j
    (i.e. reindexed for the keyed alphabet) — the de-subbed indices live in alphabet space,
    so the expected distribution must too, or shift recovery is meaningless.
    """
    classes: list[list[int]] = [[] for _ in range(period)]
    for i, v in enumerate(stream_idx):
        classes[i % period].append(v)
    shifts = []
    for cl in classes:
        m = len(cl)
        if not m:
            shifts.append(0)
            continue
        best_s, best_chi = 0, 1e18
        for s in range(26):
            cnt = [0] * 26
            for v in cl:
                j = (v - s) % 26 if variant == "vig" else (s - v) % 26
                cnt[j] += 1
            chi = 0.0
            for k in range(26):
                e = expect[k] * m
                chi += (cnt[k] - e) ** 2 / e
            if chi < best_chi:
                best_chi, best_s = chi, s
        shifts.append(best_s)
    return shifts


def _desub(stream_idx: list[int], shifts: list[int], period: int, variant: str, alph: str) -> str:
    out = []
    for i, v in enumerate(stream_idx):
        s = shifts[i % period]
        j = (v - s) % 26 if variant == "vig" else (s - v) % 26
        out.append(alph[j])
    return "".join(out)


def _decode_columnar(ct: str, order: list[int], complete: bool) -> str:
    return (_col_decode if complete else _icol._decode_letters)(ct, order)


class _Config:
    """One (width, period, alphabet, variant, layer) cell for the climber."""

    __slots__ = ("ct", "width", "period", "alph", "variant", "layer", "complete",
                 "n", "ai", "expect", "_cidx", "_outer_desub")

    def __init__(self, ct, width, period, alph, variant, layer):
        self.ct = ct
        self.width = width
        self.period = period
        self.alph = alph
        self.variant = variant
        self.layer = layer
        self.n = len(ct)
        self.complete = (self.n % width == 0)
        self.ai = {c: i for i, c in enumerate(alph)}
        # expected English frequency reindexed into the (keyed) alphabet's index space
        self.expect = [ENGLISH_MONOGRAM_FREQ[alph[j]] for j in range(26)]
        self._cidx = [self.ai[c] for c in ct]
        self._outer_desub = None
        if layer == "outer":  # de-sub is order-independent — compute once
            shifts = _best_shifts(self._cidx, period, variant, self.expect)
            self._outer_desub = _desub(self._cidx, shifts, period, variant, alph)

    def plaintext(self, order: list[int]) -> str:
        if self.layer == "inner":
            stream = _decode_columnar(self.ct, order, self.complete)
            if len(stream) != self.n:
                return ""
            sidx = [self.ai[c] for c in stream]
            shifts = _best_shifts(sidx, self.period, self.variant, self.expect)
            return _desub(sidx, shifts, self.period, self.variant, self.alph)
        else:  # outer: de-sub first (order-independent), then de-transpose
            out = _decode_columnar(self._outer_desub, order, self.complete)
            return out if len(out) == self.n else ""


def _anneal(cfg: _Config, scorer, rng, iters: int, t0: float, t1: float):
    """Simulated annealing over the columnar order; returns (best_score, best_order, best_pt)."""
    W = cfg.width
    order = list(range(W))
    rng.shuffle(order)
    pt = cfg.plaintext(order)
    cur = scorer.fitness(pt)
    best_score, best_order, best_pt = cur, order[:], pt
    if iters <= 0:
        return best_score, best_order, best_pt
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / iters)
        a, b = rng.randrange(W), rng.randrange(W)
        if a == b:
            continue
        order[a], order[b] = order[b], order[a]
        pt = cfg.plaintext(order)
        cand = scorer.fitness(pt)
        d = cand - cur
        if d > 0 or rng.random() < math.exp(d / max(T, 1e-9)):
            cur = cand
            if cand > best_score:
                best_score, best_order, best_pt = cand, order[:], pt
        else:
            order[a], order[b] = order[b], order[a]
    return best_score, best_order, best_pt


def solve_config(ct, width, period, alph_spec, variant, layer,
                 restarts=20, iters=4000, ngram="hexagrams", seed=0):
    """Run the dual climber for one config; returns dict(score, plaintext, order, ...)."""
    ct = only_letters(ct)
    alph = _alphabet(alph_spec)
    scorer = resolve_scorer(ngram)
    cfg = _Config(ct, width, period, alph, variant, layer)
    rng = random.Random(seed)
    best = (-1e18, None, None)
    t0, t1 = 0.5, 0.02
    for r in range(restarts):
        s, order, pt = _anneal(cfg, scorer, random.Random(rng.randrange(1 << 30)), iters, t0, t1)
        if s > best[0]:
            best = (s, order, pt)
    score, order, pt = best
    from .validate import solve_confidence
    conf = solve_confidence(pt, len(ct)) if pt else {"recovered": False, "word_coverage": 0.0}
    return {
        "score": score, "plaintext": pt, "order": order,
        "width": width, "period": period, "alphabet": alph_spec,
        "variant": variant, "layer": layer, "ngram": scorer.name,
        "quad": get_scorer("quadgrams").score(pt) if pt else None,
        "recovered": conf["recovered"], "word_coverage": conf["word_coverage"],
    }


def solve(ct, *, layer="inner", widths=range(6, 18), periods=range(9, 17),
          alphabets=("KRYPTOS", "STD"), variants=("vig", "beaufort"),
          restarts=20, iters=4000, ngram="hexagrams", seed=0, top=5):
    """Sweep the dual climber over configs; return the best results (sorted)."""
    results = []
    for width in widths:
        for period in periods:
            for alph in alphabets:
                for variant in variants:
                    r = solve_config(ct, width, period, alph, variant, layer,
                                     restarts=restarts, iters=iters, ngram=ngram, seed=seed)
                    results.append(r)
    results.sort(key=lambda d: -d["score"])
    return results[:top]
