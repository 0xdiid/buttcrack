"""Layered stacks: a periodic substitution over N columnar transpositions.

WHY THIS MODULE EXISTS
----------------------
:mod:`buttcrack.layered` cracks the two-layer shape (one periodic substitution over one
columnar) by searching the substitution and the transposition *together* — for each
candidate column order it re-derives the shifts. That couples two independent problems and
caps the practical depth at one transposition.

The decoupling that makes arbitrary depth cheap:

    **A transposition preserves monogram frequencies.**

So when a periodic substitution sits OUTSIDE a stack of transpositions, the substituted
stream still has the plaintext's letter distribution, merely permuted in position. The
per-coset shifts are therefore recoverable from the raw ciphertext by monogram chi-square
**without touching the transposition at all** — which is exactly how the ACA solves these by
hand, and why a published solution can say "the period-45 key is visible in the raw CT".

Peel the substitution first and what remains is a pure transposition problem, at whatever
depth. One instrument then covers the whole family:

    layers=0  periodic substitution only          (Vigenere / Quagmire / Beaufort ...)
    layers=1  substitution over one columnar
    layers=2  substitution over a double columnar
    period=1  pure transposition, no substitution

Complete (flush) rectangles only: ``n % width == 0``. Incomplete columnars are a different
geometry and belong to :mod:`buttcrack.ciphers.incomplete_columnar`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .layered import (
    _chi2,
    _fast_quad_table,
    _freqs_for,
    _qscore,
    alphabet_header,
    detect_periods,
)
from .scoring import NgramScorer
from .telemetry import Progress, resolve
from .text import only_letters
from .validate import long_word_coverage

CONVENTIONS = ("vigenere", "beaufort", "variant-beaufort")


# -- transposition geometry ----------------------------------------------------


def columnar_inverse_index(n: int, width: int, order: list[int]) -> list[int]:
    """``idx`` such that ``plaintext[k] = ciphertext[idx[k]]`` for a complete columnar.

    Encryption writes the text into ``n/width`` rows of ``width`` and reads the columns out
    in ``order`` (``order[j]`` is the column emitted j-th), so column ``c`` occupies block
    ``inv[c]`` of the ciphertext.
    """
    if n % width:
        raise ValueError(f"complete columnar needs width | n; {width} does not divide {n}")
    rows = n // width
    inv = [0] * width
    for j, c in enumerate(order):
        inv[c] = j
    idx = [0] * n
    for i in range(rows):
        base = i * width
        for c in range(width):
            idx[base + c] = inv[c] * rows + i
    return idx


def compose_index(outer: list[int], inner: list[int]) -> list[int]:
    """Index map of applying ``inner`` to the result of ``outer``.

    If ``outer`` inverts the last-applied transposition and ``inner`` the one before it,
    ``compose_index(outer, inner)`` inverts both in one gather.
    """
    return [outer[k] for k in inner]


def _gather(src: list[int], idx: list[int]) -> list[int]:
    return [src[i] for i in idx]


# -- peeling the outer periodic substitution -----------------------------------


@dataclass
class Peel:
    """The recovered outer substitution and the stream left underneath it."""

    period: int
    shifts: list[int]
    convention: str
    alphabet: str
    stream: list[int]  # standard A-Z indices, still transposed
    chi2: float

    @property
    def text(self) -> str:
        return "".join(chr(65 + x) for x in self.stream)


def _apply_convention(cidx: int, shift: int, convention: str) -> int:
    if convention == "vigenere":
        return (cidx - shift) % 26
    if convention == "beaufort":
        return (shift - cidx) % 26
    if convention == "variant-beaufort":
        return (cidx + shift) % 26
    raise ValueError(f"unknown convention {convention!r}")


def peel_periodic(
    ciphertext: str,
    *,
    period: int,
    alphabet: str = "KRYPTOS",
    convention: str = "vigenere",
    language: str = "english",
) -> Peel:
    """Recover the outer periodic substitution by per-coset monogram chi-square.

    Valid whenever everything *below* the substitution preserves monogram frequencies —
    i.e. any stack of transpositions, at any depth. Cost is O(26·n) and independent of the
    transposition, which is the whole point.
    """
    ct = only_letters(ciphertext).upper()
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    hdr_std = [ord(ch) - 65 for ch in header]
    freqs = _freqs_for(language)
    n = len(ct)

    shifts: list[int] = []
    total = 0.0
    for j in range(period):
        col = [hpos[c] for c in ct[j::period]]
        best = (1e18, 0)
        for sh in range(26):
            dec = "".join(chr(65 + hdr_std[_apply_convention(c, sh, convention)]) for c in col)
            s = _chi2(dec, freqs)
            if s < best[0]:
                best = (s, sh)
        shifts.append(best[1])
        total += best[0]

    stream = [
        hdr_std[_apply_convention(hpos[ct[i]], shifts[i % period], convention)] for i in range(n)
    ]
    return Peel(period, shifts, convention, alphabet, stream, total / max(period, 1))


# -- solving the transposition stack -------------------------------------------


@dataclass
class StackSolution:
    score: float
    widths: list[int] = field(default_factory=list)
    orders: list[list[int]] = field(default_factory=list)
    plain: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(chr(65 + x) for x in self.plain)


def _brute_single(stream: list[int], width: int, table: list[float]) -> StackSolution:
    """Exhaust every read-order for one complete columnar."""
    from itertools import permutations

    n = len(stream)
    best = StackSolution(-1e18)
    for order in permutations(range(width)):
        idx = columnar_inverse_index(n, width, list(order))
        cand = _gather(stream, idx)
        s = _qscore(cand, table)
        if s > best.score:
            best = StackSolution(s, [width], [list(order)], cand)
    return best


def _anneal_double(
    stream: list[int],
    w1: int,
    w2: int,
    table: list[float],
    *,
    restarts: int = 8,
    iters: int = 6000,
    rng: random.Random | None = None,
) -> StackSolution:
    """Simulated annealing over the PAIR of read-orders of a double columnar.

    ``w1`` is the transposition applied first (innermost, closest to the plaintext), ``w2``
    the one applied last. Inverting means undoing ``w2`` then ``w1``, so the composed gather
    is ``compose_index(inv_w2, inv_w1)``.
    """
    rng = rng or random.Random(0)
    n = len(stream)
    best = StackSolution(-1e18)

    def score_of(o1: list[int], o2: list[int]) -> tuple[float, list[int]]:
        idx = compose_index(columnar_inverse_index(n, w2, o2), columnar_inverse_index(n, w1, o1))
        cand = _gather(stream, idx)
        return _qscore(cand, table), cand

    for _ in range(restarts):
        o1 = list(range(w1))
        o2 = list(range(w2))
        rng.shuffle(o1)
        rng.shuffle(o2)
        cur, cand = score_of(o1, o2)
        temp = 8.0
        for it in range(iters):
            temp = max(0.05, 8.0 * (1.0 - it / iters))
            which = o1 if rng.random() < 0.5 else o2
            a, b = rng.randrange(len(which)), rng.randrange(len(which))
            if a == b:
                continue
            which[a], which[b] = which[b], which[a]
            s, c2 = score_of(o1, o2)
            if s > cur or rng.random() < pow(2.718281828, (s - cur) / max(temp, 1e-9)):
                cur, cand = s, c2
            else:
                which[a], which[b] = which[b], which[a]
        if cur > best.score:
            best = StackSolution(cur, [w1, w2], [list(o1), list(o2)], cand)
    return best


def _anneal_double_np(
    stream: list[int],
    w1: int,
    w2: int,
    scorer: NgramScorer,
    *,
    restarts: int = 24,
    iters: int = 60000,
    rng: random.Random | None = None,
):
    """Numpy simulated annealing over the PAIR of read-orders of a double columnar.

    Double transposition does not decompose: undoing only the outer layer leaves a single
    columnar of English, which has English monograms and no positional signal, so there is
    nothing to score an intermediate against. Both orders have to move together, which puts
    the space at ``w1! * w2!`` (1.3e11 at width 9) and rules out enumeration.

    What makes annealing viable anyway is that the objective is cheap and smooth under
    single column swaps: one swap relocates a whole column of the grid, so a partially
    correct order already scores above a random one. Scoring is vectorised — a candidate
    costs one gather and four array ops — so a long schedule is affordable.
    """
    import numpy as np

    from .ciphers import _quagmire_solver as qs

    table, ngram = qs._fast_table(scorer)
    if ngram != 4:
        raise ValueError("double-columnar annealing requires a quadgram scorer")
    tab = np.asarray(table, dtype=np.float32)
    arr = np.asarray(stream, dtype=np.int64)
    n = arr.size
    rng = rng or random.Random(0)

    inv_cache1 = {}
    inv_cache2 = {}

    def idx_for(o1, o2):
        k1, k2 = tuple(o1), tuple(o2)
        i1 = inv_cache1.get(k1)
        if i1 is None:
            i1 = inv_cache1[k1] = np.asarray(columnar_inverse_index(n, w1, list(o1)))
        i2 = inv_cache2.get(k2)
        if i2 is None:
            i2 = inv_cache2[k2] = np.asarray(columnar_inverse_index(n, w2, list(o2)))
        return i2[i1]

    def score(o1, o2):
        cand = arr[idx_for(o1, o2)]
        code = ((cand[:-3] * 26 + cand[1:-2]) * 26 + cand[2:-1]) * 26 + cand[3:]
        return float(tab[code].sum()), cand

    best = StackSolution(-1e18)
    for _ in range(restarts):
        o1 = list(range(w1))
        o2 = list(range(w2))
        rng.shuffle(o1)
        rng.shuffle(o2)
        cur, cand = score(o1, o2)
        for it in range(iters):
            temp = max(0.02, 6.0 * (1.0 - it / iters))
            which = o1 if rng.random() < 0.5 else o2
            a, b = rng.randrange(len(which)), rng.randrange(len(which))
            if a == b:
                continue
            which[a], which[b] = which[b], which[a]
            s, c2 = score(o1, o2)
            if s >= cur or rng.random() < 2.718281828 ** ((s - cur) / temp):
                cur, cand = s, c2
            else:
                which[a], which[b] = which[b], which[a]
            if cur > best.score:
                best = StackSolution(cur, [w1, w2], [list(o1), list(o2)], cand.tolist())
    return best


def solve_transposition_stack(
    stream: list[int],
    table: list[float],
    *,
    scorer: NgramScorer | None = None,
    layers: int = 1,
    widths=None,
    brute_max_width: int = 9,
    restarts: int = 8,
    iters: int = 6000,
    rng: random.Random | None = None,
) -> StackSolution:
    """Undo ``layers`` complete columnars beneath an already-peeled substitution."""
    n = len(stream)
    if layers == 0:
        return StackSolution(_qscore(stream, table), [], [], list(stream))
    cand_widths = [w for w in (widths or range(3, 13)) if 2 <= w <= n and n % w == 0]
    best = StackSolution(-1e18)
    if layers == 1:
        for w in cand_widths:
            if w > brute_max_width:
                continue
            s = _brute_single(stream, w, table)
            if s.score > best.score:
                best = s
        return best
    if layers == 2:
        # Two-stage: a cheap anneal over every admissible width pair to find the geometry,
        # then a full-budget re-anneal of the winner only. Annealing all pairs at full
        # budget is quadratic in the divisor count and dominated by hopeless pairs.
        scan: list[tuple[float, int, int]] = []
        for w1 in cand_widths:
            for w2 in cand_widths:
                q = (
                    _anneal_double_np(
                        stream,
                        w1,
                        w2,
                        scorer,
                        restarts=max(2, restarts // 8),
                        iters=max(4000, iters // 8),
                        rng=rng,
                    )
                    if scorer is not None
                    else _anneal_double(stream, w1, w2, table, restarts=2, iters=4000, rng=rng)
                )
                scan.append((q.score, w1, w2))
                if q.score > best.score:
                    best = q
        scan.sort(reverse=True)
        for _, w1, w2 in scan[:2]:
            s = (
                _anneal_double_np(stream, w1, w2, scorer, restarts=restarts, iters=iters, rng=rng)
                if scorer is not None
                else _anneal_double(stream, w1, w2, table, restarts=restarts, iters=iters, rng=rng)
            )
            if s.score > best.score:
                best = s
        return best
    raise ValueError("layers must be 0, 1 or 2")


def _stack_index(n: int, widths: list[int], orders: list[list[int]]) -> list[int]:
    """The single gather that inverts a whole stack (outermost layer listed last)."""
    idx = list(range(n))
    for w, o in zip(reversed(widths), reversed(orders), strict=True):
        idx = compose_index(idx, columnar_inverse_index(n, w, o))
    return idx


def peel_with(ct: str, period: int, shifts: list[int], alphabet: str, convention: str) -> list[int]:
    """Apply known shifts; returns the still-transposed stream as A-Z indices."""
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    hdr_std = [ord(ch) - 65 for ch in header]
    return [
        hdr_std[_apply_convention(hpos[c], shifts[i % period], convention)]
        for i, c in enumerate(ct)
    ]


# -- joint refinement ----------------------------------------------------------


def refine_shifts(
    ct: str,
    idx: list[int],
    table: list[float],
    *,
    period: int,
    alphabet: str,
    convention: str,
    seed: list[int],
    passes: int = 6,
    restarts: int = 0,
    rng: random.Random | None = None,
) -> tuple[float, list[int], list[int]]:
    """Coordinate-ascent the per-coset shifts against QUADGRAMS of the FINAL plaintext.

    The chi-square peel is a monogram estimate, and a monogram estimate on a short coset is
    noisy: at period 45 over 224 letters each coset holds 5 letters, which is nowhere near
    enough to pick a shift. Quadgrams can only be used once the transposition is undone, so
    the two halves have to be solved alternately rather than in one pass — seed with
    chi-square, solve the transposition, then re-fit the shifts through the recovered
    geometry, and repeat until neither moves.

    ``idx`` is the gather that inverts the whole transposition stack.
    """
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    hdr_std = [ord(ch) - 65 for ch in header]
    ctn = [hpos[c] for c in ct]
    n = len(ctn)
    shifts = list(seed)
    buf = [0] * n

    def build() -> list[int]:
        for k in range(n):
            i = idx[k]
            buf[k] = hdr_std[_apply_convention(ctn[i], shifts[i % period], convention)]
        return buf

    def climb(init: list[int]) -> tuple[float, list[int]]:
        nonlocal shifts
        shifts = list(init)
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
                    sc = _qscore(build(), table)
                    if sc > best[0]:
                        best = (sc, x)
                shifts[j] = best[1]
                if best[1] != keep:
                    cur, moved = best[0], True
            if not moved:
                break
        return cur, list(shifts)

    best_s, best_sh = climb(seed)
    rng = rng or random.Random(0)
    for _ in range(restarts):
        s2, sh2 = climb([rng.randrange(26) for _ in range(period)])
        if s2 > best_s:
            best_s, best_sh = s2, sh2
    shifts = list(best_sh)
    return best_s, best_sh, list(build())


# -- the whole stack -----------------------------------------------------------


def crack_stack(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "KRYPTOS",
    periods: list[int] | None = None,
    conventions: tuple[str, ...] = ("vigenere",),
    layers: int | tuple[int, ...] = (0, 1),
    widths=None,
    language: str | None = None,
    brute_max_width: int = 9,
    restarts: int = 8,
    iters: int = 6000,
    coverage_stop: float = 0.45,
    refine_rounds: int = 4,
    shift_restarts: int = 6,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Crack `periodic substitution over N columnars` by peeling then solving.

    Returns the best candidate across the requested periods, conventions and depths, with a
    machine-readable ``structure`` describing every layer. Stops early once a candidate
    reads as clean English (``coverage_stop``).
    """
    ct = only_letters(ciphertext).upper()
    table = _fast_quad_table(scorer)
    if not language:
        language = getattr(scorer, "lang", "english")
    rng = rng or random.Random(0)
    depths = (layers,) if isinstance(layers, int) else tuple(layers)
    if periods is None:
        periods = [1, *detect_periods(ct)]

    best: dict[str, Any] | None = None
    for period in periods:
        for convention in conventions:
            peel = (
                Peel(1, [0], convention, alphabet, [ord(c) - 65 for c in ct], 0.0)
                if period == 1
                else peel_periodic(
                    ct,
                    period=period,
                    alphabet=alphabet,
                    convention=convention,
                    language=language,
                )
            )
            for depth in depths:
                sol = solve_transposition_stack(
                    peel.stream,
                    table,
                    scorer=scorer,
                    layers=depth,
                    widths=widths,
                    brute_max_width=brute_max_width,
                    restarts=restarts,
                    iters=iters,
                    rng=rng,
                )
                shifts = peel.shifts
                if period > 1:
                    # alternate: refit the shifts through the recovered geometry, then
                    # re-solve the geometry under the better shifts, until neither moves.
                    for _ in range(refine_rounds):
                        idx = _stack_index(len(ct), sol.widths, sol.orders)
                        sc2, shifts2, plain2 = refine_shifts(
                            ct,
                            idx,
                            table,
                            period=period,
                            alphabet=alphabet,
                            convention=convention,
                            seed=shifts,
                            restarts=shift_restarts,
                            rng=rng,
                        )
                        if shifts2 == shifts and sc2 <= sol.score:
                            break
                        shifts = shifts2
                        sol = StackSolution(sc2, sol.widths, sol.orders, plain2)
                        again = solve_transposition_stack(
                            peel_with(ct, period, shifts, alphabet, convention),
                            table,
                            scorer=scorer,
                            layers=depth,
                            widths=widths,
                            brute_max_width=brute_max_width,
                            restarts=restarts,
                            iters=iters,
                            rng=rng,
                        )
                        if again.score > sol.score:
                            sol = again
                        else:
                            break
                cov = long_word_coverage(sol.text)
                cand = {
                    "structure": {
                        "layer_order": "substitution-over-transposition",
                        "substitution": ("none" if period == 1 else f"{convention}/{alphabet}"),
                        "period": period,
                        "shifts": shifts,
                        "transposition_layers": depth,
                        "widths": sol.widths,
                        "orders": sol.orders,
                    },
                    "plaintext": sol.text,
                    "score": sol.score,
                    "word_coverage": round(cov, 3),
                }
                if best is None or sol.score > best["score"]:
                    best = cand
                if cov >= coverage_stop:
                    return cand
    assert best is not None
    return best


# -- the combined entry point --------------------------------------------------


def _as_result(cands):
    """Adapt a cipher's Candidate list to the dict shape the stack solvers return."""
    if not cands:
        return None
    c = cands[0]
    return {
        "structure": {"layer_order": "substitution-only", "key": getattr(c, "key", None)},
        "plaintext": c.plaintext,
        "score": c.score,
    }


def get_registered(name: str):
    from .registry import get  # noqa: PLC0415

    return get(name)


def crack_layered_fn(*a, **k):
    from .layered import crack_layered  # noqa: PLC0415

    return crack_layered(*a, **k)


def _deep_path(ct, scorer, *, alphabet, conventions, divisors, language, rng):
    """Depth-2 via the exact-inner solver: search the outer order, Held-Karp the inner."""
    best = None
    for convention in conventions:
        for period in [1, *detect_periods(ct)]:
            peel = (
                Peel(1, [0], convention, alphabet, [ord(c) - 65 for c in ct], 0.0)
                if period == 1
                else peel_periodic(
                    ct,
                    period=period,
                    alphabet=alphabet,
                    convention=convention,
                    language=language or "english",
                )
            )
            for w in divisors:
                sol = solve_stack_deep(
                    peel.stream, scorer, widths=[w, w], restarts=10, iters=1500, rng=rng
                )
                cov = long_word_coverage(sol.text)
                if best is None or cov > best[0]:
                    best = (
                        cov,
                        {
                            "structure": {
                                "layer_order": "substitution-over-transposition",
                                "substitution": "none"
                                if period == 1
                                else f"{convention}/{alphabet}",
                                "period": period,
                                "shifts": peel.shifts,
                                "transposition_layers": 2,
                                "widths": sol.widths,
                                "orders": sol.orders,
                            },
                            "plaintext": sol.text,
                            "score": sol.score,
                        },
                    )
    return best[1] if best else None


def crack_any_stack(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "KRYPTOS",
    conventions: tuple[str, ...] = ("vigenere",),
    max_layers: int = 2,
    widths=None,
    language: str | None = None,
    restarts: int = 30,
    iters: int = 60000,
    accept: float = 0.58,
    skip_expensive: float = 0.45,
    progress: Progress | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Try the layered paths cheapest-first and return the first that reads as English.

    Three solvers, because no one of them dominates:

    * **depth 0** — a plain periodic substitution. Delegated to the dedicated Quagmire
      solver, which re-derives shifts against quadgrams and beats a monogram peel when the
      period is large relative to the text (period 40 over 280 letters leaves 7 letters per
      coset, where chi-square is guesswork).
    * **depth 1** — substitution over one columnar. Delegated to :func:`layered.crack_layered`,
      which searches order and shifts *jointly*; for short cosets that coupling is what wins.
    * **depth 2** — substitution over a double columnar. Only this module does it: peel the
      substitution off the raw ciphertext, then anneal the pair of read-orders.

    Ranked by long-word coverage, which is comparable across paths in a way raw quadgram
    score is not.
    """
    ct = only_letters(ciphertext).upper()
    rng = rng or random.Random(0)
    out: list[dict[str, Any]] = []
    errors: list[str] = []
    pr = resolve(progress)
    n = len(ct)
    divisors = [w for w in range(3, 13) if n % w == 0]
    pr.note(
        f"crack_any_stack: n={n}, complete-rectangle widths {divisors}, max_layers={max_layers}"
    )

    def record(r, solver):
        r.setdefault("structure", {})["solver"] = solver
        r["word_coverage"] = round(long_word_coverage(r["plaintext"]), 3)
        out.append(r)
        return r["word_coverage"] >= accept

    # Cheapest first, and stop as soon as one reads as English. Running all four and
    # ranking costs the sum of their runtimes on every input; the peel-based paths are
    # milliseconds and cover the two commonest shapes, so ordering by cost is most of the
    # speed. Only fall through to the expensive joint searches when the cheap ones fail.
    def _peel_paths():
        return crack_stack(
            ct,
            scorer,
            alphabet=alphabet,
            conventions=conventions,
            layers=tuple(d for d in (0, 1) if d <= max_layers),
            widths=divisors or None,
            language=language,
            rng=rng,
        )

    stages: list[tuple[str, Any]] = [("stack/peel", _peel_paths)]
    if max_layers >= 2:
        stages.append(
            (
                "stack/deep",
                lambda: _deep_path(
                    ct,
                    scorer,
                    alphabet=alphabet,
                    conventions=conventions,
                    divisors=divisors,
                    language=language,
                    rng=rng,
                ),
            )
        )
    stages.append(
        (
            "quagmire3",
            lambda: _as_result(get_registered("quagmire3").crack(ct, scorer, top=1, rng=rng)),
        )
    )
    if max_layers >= 1:
        stages.append(
            (
                "layered",
                lambda: crack_layered_fn(
                    ct,
                    scorer,
                    alphabet=alphabet,
                    language=language,
                    widths=divisors or range(4, 9),
                    rng=rng,
                ),
            )
        )

    # `layered` is the joint substitution+order search: it is the only stage that can crack
    # a period so long its cosets are too short to peel (period 45 over 224 letters leaves
    # 5 letters per coset), and it costs a width-sweep of order-brutes to do it. Its
    # HYPOTHESIS CLASS, though — one periodic substitution over one columnar — is already
    # covered by stack/peel at depth 1, just less powerfully. So once any earlier stage has
    # produced something that reads as English, running it buys almost nothing and costs
    # minutes. Skip it then, and keep it for the case it uniquely serves.
    for solver, fn in stages:
        best_cov = max((r.get("word_coverage", 0.0) for r in out), default=0.0)
        if solver == "layered" and out and best_cov >= skip_expensive:
            errors.append("layered: skipped (an earlier stage already reads as English)")
            pr.note("layered: SKIPPED — an earlier stage already reads as English")
            continue
        with pr.stage(f"solver:{solver}"):
            try:
                r = fn()
            except Exception as exc:  # must not sink the others -- nor hide them
                errors.append(f"{solver}: {type(exc).__name__}: {exc}")
                pr.note(f"{solver} FAILED: {type(exc).__name__}: {exc}")
                continue
        if r is None:
            pr.note(f"{solver}: no candidate")
            continue
        cov_now = round(long_word_coverage(r["plaintext"]), 3)
        pr.note(f"{solver}: coverage {cov_now:.3f} (accept at {accept})")
        if record(r, solver):
            r["solver_errors"] = errors or None
            r["structure"]["tried"] = [x["structure"].get("solver") for x in out]
            return r

    if not out:
        raise RuntimeError("no layered solver produced a candidate: " + "; ".join(errors))
    out.sort(key=lambda r: (r.get("word_coverage", 0.0), r.get("score", -1e18)), reverse=True)
    best = out[0]
    best.setdefault("structure", {})["tried"] = [r["structure"].get("solver") for r in out]
    if errors:
        best["solver_errors"] = errors
    return best


# -- arbitrary depth -----------------------------------------------------------


def solve_stack_deep(
    stream: list[int],
    scorer: NgramScorer,
    *,
    widths: list[int],
    restarts: int = 12,
    iters: int = 4000,
    rng: random.Random | None = None,
) -> StackSolution:
    """Undo ``len(widths)`` columnars by searching all but ONE of them.

    The innermost transposition (the one applied first, closest to the plaintext) is the
    only layer whose columns are *plaintext* columns, so it is the only one whose order can
    be scored pairwise — and therefore the only one recoverable EXACTLY, by Held-Karp
    column matching. Every outer layer has to be searched.

    So: anneal the outer ``N-1`` orders, and at each step undo them and solve the innermost
    exactly. That divides the search space by ``w!`` — 362,880-fold at width 9 — and it
    also makes the landscape far smoother, because every candidate is evaluated with its
    best possible inner layer rather than a random one.

    Depth 1 is pure Held-Karp with no search at all. Depth 2 searches one layer. Depth 3
    searches two. The cost is ``(w!)^(N-1)`` rather than ``(w!)^N``, which is what makes
    depth 3 reachable at all.
    """
    from .columnar_exact import _bigram_table, column_adjacency, held_karp_path

    n = len(stream)
    inner_w, outer_ws = widths[0], list(widths[1:])
    tab = _bigram_table(scorer)
    rows = n // inner_w
    rng = rng or random.Random(0)

    def solve_inner(seq: list[int]) -> tuple[float, list[int], list[int]]:
        blocks = [seq[j * rows : (j + 1) * rows] for j in range(inner_w)]
        sc, path = held_karp_path(column_adjacency(blocks, tab))
        plain = [0] * n
        for c, blk in enumerate(path):
            for i in range(rows):
                plain[i * inner_w + c] = blocks[blk][i]
        order = [0] * inner_w
        for col, blk in enumerate(path):
            order[blk] = col
        return sc, order, plain

    if not outer_ws:
        sc, order, plain = solve_inner(list(stream))
        return StackSolution(sc, [inner_w], [order], plain)

    def peel_outers(orders: list[list[int]]) -> list[int]:
        idx = list(range(n))
        for w, o in zip(reversed(outer_ws), reversed(orders), strict=True):
            idx = compose_index(idx, columnar_inverse_index(n, w, o))
        return _gather(stream, idx)

    best = StackSolution(-1e18)
    for _ in range(restarts):
        orders = []
        for w in outer_ws:
            o = list(range(w))
            rng.shuffle(o)
            orders.append(o)
        cur, _, _ = solve_inner(peel_outers(orders))
        for it in range(iters):
            temp = max(0.02, 5.0 * (1.0 - it / iters))
            li = rng.randrange(len(orders))
            o = orders[li]
            a, b = rng.randrange(len(o)), rng.randrange(len(o))
            if a == b:
                continue
            o[a], o[b] = o[b], o[a]
            sc, iorder, plain = solve_inner(peel_outers(orders))
            if sc >= cur or rng.random() < 2.718281828 ** ((sc - cur) / temp):
                cur = sc
                if sc > best.score:
                    best = StackSolution(
                        sc,
                        [inner_w, *outer_ws],
                        [iorder, *[list(x) for x in orders]],
                        plain,
                    )
            else:
                o[a], o[b] = o[b], o[a]
    return best


# -- chained keys: a running key supplied from outside ------------------------


def crack_with_keystream(
    ciphertext: str,
    keystream: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "KRYPTOS",
    convention: str = "vigenere",
    layers: int = 1,
    widths=None,
    max_offset: int = 0,
    directions: tuple[str, ...] = ("forward",),
) -> dict[str, Any]:
    """Crack ``CT = runningkey(transposition(PT))`` when the keystream is KNOWN.

    A running key is not a search problem — with the keystream in hand the substitution is
    a deterministic subtraction, and only the transposition underneath is unknown. What it
    needs is an INPUT, because the keystream comes from outside the ciphertext (a book, a
    previous message, an earlier puzzle's plaintext). Serial puzzles chain like this
    routinely, and no amount of solver work substitutes for being able to say what the
    keystream is.

    ``max_offset`` sweeps a start offset into the keystream, and ``directions`` may include
    ``"reverse"`` — both are the usual conventions when the keystream is a longer text than
    the message.
    """
    from .columnar_exact import solve_columnar_widths

    ct = only_letters(ciphertext).upper()
    ks_full = only_letters(keystream).upper()
    if not ks_full:
        raise ValueError("keystream has no letters")
    header = alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    n = len(ct)
    divisors = [w for w in (widths or range(2, 21)) if 2 <= w < n and n % w == 0]

    best: tuple[float, dict[str, Any]] | None = None
    for direction in directions:
        ks_base = ks_full if direction == "forward" else ks_full[::-1]
        for off in range(max_offset + 1):
            ks = ks_base[off:] + ks_base[:off] if off else ks_base
            stream = [
                hpos[header[_apply_convention(hpos[ct[i]], hpos[ks[i % len(ks)]], convention)]]
                for i in range(n)
            ]
            text = "".join(header[v] for v in stream)
            if layers == 0:
                cands = [(scorer.score(text) / max(n, 1), None, text)]
            else:
                std = [ord(header[v]) - 65 for v in stream]
                cands = [
                    (scorer.score(s.plaintext) / max(n, 1), s, s.plaintext)
                    for s in solve_columnar_widths(
                        "".join(chr(65 + x) for x in std), scorer=scorer, widths=divisors, top=3
                    )
                ]
            for sc_val, sol, plain in cands:
                cov = long_word_coverage(plain)
                if best is None or sc_val > best[0]:
                    best = (
                        sc_val,
                        {
                            "structure": {
                                "layer_order": "runningkey-over-transposition",
                                "substitution": f"{convention}/{alphabet} (supplied keystream)",
                                "keystream_offset": off,
                                "keystream_direction": direction,
                                "columnar_width": getattr(sol, "width", None),
                                "columnar_order": getattr(sol, "order", None),
                                "solver": "stack/keystream",
                            },
                            "plaintext": plain,
                            "score": sc_val,
                            "word_coverage": round(cov, 3),
                        },
                    )
    assert best is not None
    return best[1]
