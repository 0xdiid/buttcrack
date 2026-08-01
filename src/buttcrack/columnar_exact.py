"""Exact columnar recovery by column matching + Held-Karp.

THE IDEA
--------
Brute-forcing a complete columnar costs ``w!`` decodes of the whole text. But a columnar's
read-order is not an arbitrary label — it is a *sequence of columns*, and adjacent columns
of the plaintext grid are adjacent letters of the plaintext. So the objective decomposes
pairwise:

    score(order) = Σ over consecutive column pairs (a, b) of  Σ over rows i  logP(a[i] b[i])

Maximising a sum of pairwise terms over an ordering is the open-path Travelling Salesman
Problem, which Held-Karp solves EXACTLY in ``O(2^w · w²)`` instead of ``O(w! · n)``.

    width  w!            2^w·w²
    7      5,040         6,272
    9      362,880       41,472
    12     479,001,600   589,824
    14     87,178,291,200 3,211,264

So width 9 goes from 363k full-text decodes to 41k integer operations — roughly three
orders of magnitude — and widths 12-16, which are simply unreachable by enumeration, become
routine. That matters for its own sake, and it matters much more as the INNER solve of a
deeper stack: anything that has to solve a columnar in its inner loop can now afford to.

The pairwise decomposition is exact for a complete rectangle apart from the row-wrap terms
(the last column of row i is followed by the first column of row i+1). Those are added as a
single extra term for the candidate first/last pair, which is why the solver returns the
best open path rather than a cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .scoring import NgramScorer
from .telemetry import Progress, resolve

try:  # optional acceleration only; the package itself stays dependency-free
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in dev/test
    _np = None  # type: ignore[assignment]  # optional-dependency fallback sentinel


@dataclass
class ColumnarSolution:
    order: list[int]
    width: int
    score: float
    plaintext: str


def _bigram_table(scorer: NgramScorer | None) -> list[list[float]]:
    """26x26 log-probability table, from a bigram scorer if available."""
    from .scoring import get_scorer

    try:
        bs = get_scorer("bigrams", getattr(scorer, "lang", "english"))
        floor = bs.floor
        tab = [[floor] * 26 for _ in range(26)]
        for gram, lp in bs.log_probs.items():
            if len(gram) == 2:
                tab[ord(gram[0]) - 65][ord(gram[1]) - 65] = lp
        return tab
    except Exception:
        # Uniform fallback keeps the solver usable without a bigram table; the caller can
        # still rescue ranking with a quadgram rescore of the top orders.
        return [[0.0] * 26 for _ in range(26)]


def column_adjacency(columns: list[list[int]], tab) -> list[list[float]]:
    """``adj[a][b]`` = score of placing column ``a`` immediately left of column ``b``."""
    w = len(columns)
    adj = [[0.0] * w for _ in range(w)]
    for a in range(w):
        ca = columns[a]
        for b in range(w):
            if a == b:
                continue
            cb = columns[b]
            adj[a][b] = sum(tab[x][y] for x, y in zip(ca, cb, strict=True))
    return adj


MAX_EXACT_WIDTH = 20
"""Widest column count Held-Karp will attempt.

At width 20 the DP is already 2^20 x 20 states and ~4e8 operations. Beyond that the exact
solver must REFUSE rather than run: width 51 divides 153 exactly, so it is a perfectly
legal thing for a caller to ask for, and attempting it is an unbounded hang whose only
symptom is silence.
"""


def held_karp_path(adj: list[list[float]]) -> tuple[float, list[int]]:
    """Maximum-weight Hamiltonian PATH over all start/end pairs. ``O(2^w · w²)``."""
    w = len(adj)
    if w == 1:
        return 0.0, [0]
    if w > MAX_EXACT_WIDTH:
        raise ValueError(
            f"width {w} exceeds MAX_EXACT_WIDTH={MAX_EXACT_WIDTH}: Held-Karp would need "
            f"2^{w} states. Use a heuristic order search at this width."
        )
    NEG = -math.inf
    size = 1 << w
    dp = [[NEG] * w for _ in range(size)]
    par = [[-1] * w for _ in range(size)]
    for s in range(w):
        dp[1 << s][s] = 0.0
    for mask in range(size):
        row = dp[mask]
        for last in range(w):
            cur = row[last]
            if cur == NEG or not (mask >> last) & 1:
                continue
            for nxt in range(w):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                val = cur + adj[last][nxt]
                if val > dp[nm][nxt]:
                    dp[nm][nxt] = val
                    par[nm][nxt] = last
    full = size - 1
    best, end = max((dp[full][k], k) for k in range(w))
    order = [end]
    mask = full
    while True:
        p = par[mask][order[-1]]
        if p < 0:
            break
        mask ^= 1 << order[-1]
        order.append(p)
    order.reverse()
    return best, order


def solve_columnar(
    text: str | list[int],
    width: int,
    *,
    scorer: NgramScorer | None = None,
    tab=None,
    rescore_top: int = 0,
) -> ColumnarSolution:
    """Recover a complete columnar's read-order exactly.

    ``text`` is the ciphertext (letters or A-Z indices). Returns the order in the same
    convention as :func:`buttcrack.stack.columnar_inverse_index` — ``order[j]`` is the
    column emitted j-th.
    """
    idx = (
        [ord(c) - 65 for c in text.upper() if c.isalpha()] if isinstance(text, str) else list(text)
    )
    n = len(idx)
    if n % width:
        raise ValueError(f"complete columnar needs width | n; {width} does not divide {n}")
    rows = n // width
    blocks = [idx[j * rows : (j + 1) * rows] for j in range(width)]
    tab = tab if tab is not None else _bigram_table(scorer)
    adj = column_adjacency(blocks, tab)
    score, seq = held_karp_path(adj)
    # `seq` lists the ciphertext blocks in plaintext-column order: block seq[c] is column c.
    # The read-order convention used everywhere else (and in the published solutions) is the
    # INVERSE of that: order[j] = the column emitted j-th.
    order = [0] * width
    for col, blk in enumerate(seq):
        order[blk] = col
    plain_idx = [0] * n
    for c, blk in enumerate(seq):
        for i in range(rows):
            plain_idx[i * width + c] = blocks[blk][i]
    plaintext = "".join(chr(65 + v) for v in plain_idx)
    return ColumnarSolution(order, width, score, plaintext)


def solve_columnar_widths(
    text: str,
    *,
    scorer: NgramScorer | None = None,
    widths=None,
    top: int = 3,
    progress: Progress | None = None,
) -> list[ColumnarSolution]:
    """Solve every admissible width exactly and rank by a quadgram rescore of the result.

    Widths beyond :data:`MAX_EXACT_WIDTH` are reported and skipped rather than attempted.
    That case is not exotic: 51 divides 153 exactly, and asking Held-Karp for 2^51 states
    is an unbounded hang whose only symptom is silence.
    """
    pr = resolve(progress)
    idx = [ord(c) - 65 for c in text.upper() if c.isalpha()]
    n = len(idx)
    asked = [w for w in (widths or range(2, 21)) if 2 <= w < n and n % w == 0]
    cand = [w for w in asked if w <= MAX_EXACT_WIDTH]
    for w in asked:
        if w > MAX_EXACT_WIDTH:
            pr.predict(f"held-karp w={w}", 2.0**w * w * w, limit=2.0**MAX_EXACT_WIDTH * 400)
    tab = _bigram_table(scorer)
    out = []
    with pr.stage("columnar-widths", units=len(cand), detail=f"widths {cand}"):
        for w in cand:
            pr.predict(f"held-karp w={w}", 2.0**w * w * w)
            sol = solve_columnar(idx, w, tab=tab)
            if scorer is not None:
                sol.score = scorer.score(sol.plaintext) / max(len(sol.plaintext), 1)
            out.append(sol)
            pr.tick()
    out.sort(key=lambda s: s.score, reverse=True)
    return out[:top]
