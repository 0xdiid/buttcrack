"""Hill over a periodic additive — `CT = M · (P + K)` — and a keyless solver for it.

THE SHAPE
---------
A plain Hill cipher is polygraphic but monoalphabetic in effect: the same block always
encrypts the same way. Adding a periodic additive *underneath* the matrix breaks that, and
it is a natural construction to meet in the wild (it is the ACA "Hill with a running
offset", and it is what a puzzle-setter reaches for when stepping up from a polyalphabetic
to a polygraphic cipher without losing the keyword flavour):

    CT_block = M · ( P_block + K )   (mod 26),  K periodic with period p at the LETTER level

Equivalently, a per-block-parity affine offset: with an n=3 matrix and p=6, even trigraphs
take K[0:3] and odd ones K[3:6], which reads as "offset (a,b,c) on even blocks, (d,e,f) on
odd".

WHY IT IS CHEAP TO BREAK
------------------------
The two halves decouple in one direction. Apply any candidate ``M⁻¹`` to the ciphertext
blocks and you get

    M⁻¹ · CT = P + K

— the plaintext plus a *pure periodic additive*, whatever K happens to be. So the additive
never has to be searched jointly with the matrix: guess the matrix, and what is left is an
ordinary period-p Vigenere over English, which chi-square plus a quadgram polish solves in
microseconds. The matrix is then the only real unknown.

That turns a 26^(n²) problem into a bank scan. For n=3 the matrix is 9 letters read
row-major, so an ordinary dictionary of 9-letter words is a few tens of thousands of
candidates — seconds, not centuries. A structural-variant bank (circulant, companion) and
an optional annealer cover non-word matrices.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .ciphers.hill import inverse_mod26, is_invertible_mod26, matrix_from_word
from .layered import _chi2, _fast_quad_table, _freqs_for, _qscore, alphabet_header
from .scoring import NgramScorer
from .telemetry import Progress, resolve
from .text import only_letters
from .validate import long_word_coverage

_np: Any
try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


@dataclass
class HillAdditiveSolution:
    matrix: list[list[int]]
    matrix_word: str | None
    additive: list[int]
    additive_word: str | None
    period: int
    alphabet: str
    plaintext: str
    score: float

    @property
    def word_coverage(self) -> float:
        return long_word_coverage(self.plaintext)


def _blocks(idx: list[int], n: int) -> list[list[int]]:
    usable = (len(idx) // n) * n
    return [idx[i : i + n] for i in range(0, usable, n)]


def apply_inverse(cipher_idx: list[int], dmat: list[list[int]], n: int) -> list[int]:
    """``M⁻¹ · CT`` block-wise; returns index stream ``P + K``."""
    out: list[int] = []
    for blk in _blocks(cipher_idx, n):
        for i in range(n):
            out.append(sum(dmat[i][k] * blk[k] for k in range(n)) % 26)
    return out


def solve_periodic_additive(
    stream: list[int],
    header: str,
    table: list[float],
    *,
    period: int,
    freqs: dict[str, float],
    passes: int = 4,
) -> tuple[float, list[int], str]:
    """Recover a period-``p`` additive from an index stream, chi-square then quadgrams."""
    n = len(stream)
    shifts = []
    for j in range(period):
        col = stream[j::period]
        best = (1e18, 0)
        for sh in range(26):
            dec = "".join(header[(v - sh) % 26] for v in col)
            s = _chi2(dec, freqs)
            if s < best[0]:
                best = (s, sh)
        shifts.append(best[1])

    hdr_std = [ord(c) - 65 for c in header]
    buf = [0] * n

    def build() -> list[int]:
        for i in range(n):
            buf[i] = hdr_std[(stream[i] - shifts[i % period]) % 26]
        return buf

    cur = _qscore(build(), table)
    for _ in range(passes):
        moved = False
        for j in range(period):
            keep = shifts[j]
            best = (cur, keep)
            for x in range(26):
                if x == keep:
                    continue
                shifts[j] = x
                s = _qscore(build(), table)
                if s > best[0]:
                    best = (s, x)
            shifts[j] = best[1]
            if best[1] != keep:
                cur, moved = best[0], True
        if not moved:
            break
    return cur, shifts, "".join(header[(stream[i] - shifts[i % period]) % 26] for i in range(n))


def matrix_bank(
    words, n: int, alphabet: str, *, variants: bool = True
) -> list[tuple[list[list[int]], str]]:
    """Every invertible ``n x n`` matrix derivable from a word bank.

    Row-major from an ``n²``-letter word; optionally also the circulant and companion forms
    of ``n``-letter words, which are the other two conventions in common use.
    """
    from .ciphers.hill import circulant_matrix, companion_matrix

    out: list[tuple[list[list[int]], str]] = []
    seen: set[tuple[int, ...]] = set()
    for w in words:
        forms = []
        if len(w) == n * n:
            forms.append((matrix_from_word, w))
        if variants and len(w) == n:
            forms.append((circulant_matrix, w))
            forms.append((companion_matrix, w))
        for fn, word in forms:
            try:
                m = fn(word, alphabet)
            except Exception:
                continue
            if len(m) != n or not is_invertible_mod26(m):
                continue
            key = tuple(x for row in m for x in row)
            if key in seen:
                continue
            seen.add(key)
            out.append((m, word))
    return out


def crack_hill_additive(
    ciphertext: str,
    scorer: NgramScorer,
    words,
    *,
    n: int = 3,
    alphabet: str = "KRYPTOS",
    periods=(1, 2, 3, 6, 9, 12),
    language: str | None = None,
    top: int = 5,
    coverage_stop: float = 0.33,
    progress: Progress | None = None,
) -> list[HillAdditiveSolution]:
    """Keyless crack of ``CT = M·(P + K)`` by scanning matrices and solving K analytically.

    ``periods`` are letter-level additive periods to try; a period that is not a multiple
    of ``n`` is still legal and still solvable, it simply mixes across block boundaries.
    Period 1 covers a plain Hill with a constant offset, and a zero additive falls out of
    that automatically.
    """
    ct = only_letters(ciphertext).upper()
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    idx = [hpos[c] for c in ct]
    table = _fast_quad_table(scorer)
    freqs = _freqs_for(language or getattr(scorer, "lang", "english"))

    pr = resolve(progress)
    bank = matrix_bank(words, n, header)
    pr.note(
        f"hill-additive: {len(bank):,} invertible matrices x {len(periods)} periods "
        f"= {len(bank) * len(periods):,} candidate decodes"
    )
    results: list[HillAdditiveSolution] = []
    stage = pr.stage("hill-additive", units=len(bank), detail=f"n={n} alphabet={alphabet}")
    stage.__enter__()
    for m, word in bank:
        try:
            d = inverse_mod26(m)
        except Exception:
            continue
        stream = apply_inverse(idx, d, n)
        for p in periods:
            score, shifts, plain = solve_periodic_additive(
                stream, header, table, period=p, freqs=freqs
            )
            sol = HillAdditiveSolution(m, word, shifts, None, p, alphabet, plain, score)
            results.append(sol)
            if sol.word_coverage >= coverage_stop:
                pr.note(f"early accept: {word} coverage {sol.word_coverage:.2f}")
                results.sort(key=lambda r: r.score, reverse=True)
                stage.__exit__(None, None, None)
                return results[:top]
        pr.tick()
        if len(results) > 20 * top:
            results.sort(key=lambda r: r.score, reverse=True)
            del results[top:]
    stage.__exit__(None, None, None)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top]


def additive_word(shifts: list[int], alphabet: str = "KRYPTOS") -> str:
    """Render a recovered additive back as letters of the keyed alphabet."""
    header = alphabet_header(alphabet)
    return "".join(header[s % 26] for s in shifts)


def anneal_matrix(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    n: int = 3,
    alphabet: str = "KRYPTOS",
    period: int = 6,
    restarts: int = 12,
    iters: int = 20000,
    language: str | None = None,
    rng: random.Random | None = None,
) -> HillAdditiveSolution | None:
    """Fallback for non-word matrices: anneal ``M⁻¹`` directly, solving K at every step.

    Perturbs the DECRYPTION matrix, since that is what the objective sees; invertibility is
    re-checked on every move because most random neighbours are singular mod 26.
    """
    ct = only_letters(ciphertext).upper()
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    idx = [hpos[c] for c in ct]
    table = _fast_quad_table(scorer)
    freqs = _freqs_for(language or getattr(scorer, "lang", "english"))
    rng = rng or random.Random(0)
    best: HillAdditiveSolution | None = None

    def evaluate(d):
        stream = apply_inverse(idx, d, n)
        return solve_periodic_additive(stream, header, table, period=period, freqs=freqs)

    for _ in range(restarts):
        while True:
            d = [[rng.randrange(26) for _ in range(n)] for _ in range(n)]
            if is_invertible_mod26(d):
                break
        cur = evaluate(d)[0]
        for it in range(iters):
            temp = max(0.02, 5.0 * (1.0 - it / iters))
            i, j = rng.randrange(n), rng.randrange(n)
            keep = d[i][j]
            d[i][j] = rng.randrange(26)
            if not is_invertible_mod26(d):
                d[i][j] = keep
                continue
            s, sh, pl = evaluate(d)
            if s >= cur or rng.random() < math.exp((s - cur) / temp):
                cur = s
                if best is None or s > best.score:
                    best = HillAdditiveSolution(
                        [row[:] for row in d], None, sh, None, period, alphabet, pl, s
                    )
            else:
                d[i][j] = keep
    return best
