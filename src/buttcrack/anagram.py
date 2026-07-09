"""Multiple-anagramming / pure columnar transposition solver.

Cipher model: ``CT = columnar(width W)(English PT)`` with NO substitution, so the
ciphertext keeps the English letter distribution (IoC ~0.066). The unknown is the
column *read-order* (a permutation of ``0..W-1``); recovering it un-shuffles the
columns and reads the plaintext back row-by-row.

The classic attack on a pure transposition is *multiple anagramming*: try to place
the physical columns side-by-side so that the letters that end up adjacent within
each grid row spell English. We score a candidate left-to-right column arrangement
by the bigram fit between vertically-stacked adjacent columns (a fast, factored
proxy that has a strong gradient under single-swap moves) and refine / rank the
survivors by the hexagram fitness of the fully assembled plaintext.

Both *complete* columnar (``W | N``) and *incomplete* columnar (ragged final row)
are handled: the column heights are computed exactly as in
``buttcrack.ciphers.incomplete_columnar`` (the first ``N mod W`` physical columns
are one row taller), and the complete case falls out as the ``N mod W == 0``
special case of the same code.

Public API
----------
``solve(ct, widths=range(4, 18)) -> dict(score, plaintext, width, order)``

``order`` is the recovered read-order: ``order[k]`` is the physical column read
k-th, identical to the convention of ``buttcrack.ciphers.columnar._read_order`` /
``incomplete_columnar._read_order`` so it feeds straight back into ``decode``.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from importlib import resources

from . import search
from .ciphers.incomplete_columnar import _column_heights
from .scoring import resolve_scorer
from .text import only_letters

# --------------------------------------------------------------------------- #
# Bigram adjacency model (cheap, factored, strong swap-gradient).
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _bigram_logp() -> tuple[dict[str, float], float]:
    """Load English bigram log10-probabilities plus an unseen-bigram floor.

    Returned as ``(table, floor)`` where ``table[XY]`` is ``log10(P(XY))`` and
    ``floor`` is the log-prob assigned to a bigram missing from the table.
    """
    raw = (
        resources.files("buttcrack.data")
        .joinpath("english_bigrams.txt")
        .read_text(encoding="ascii")
    )
    counts: dict[str, int] = {}
    total = 0
    for line in raw.splitlines():
        if not line:
            continue
        gram, _, cnt = line.partition(" ")
        c = int(cnt)
        counts[gram] = c
        total += c
    total = max(total, 1)
    table = {g: math.log10(c / total) for g, c in counts.items()}
    floor = math.log10(0.01 / total)
    return table, floor


def _physical_columns(ct: str, width: int, order: list[int]) -> list[str]:
    """Place the transmitted ciphertext blocks into their physical columns.

    Given a candidate read-order (``order[k]`` = physical column read k-th), peel the
    transmitted stream into blocks whose lengths are the physical column heights —
    exactly the split used by ``incomplete_columnar._decode_letters`` — and return the
    blocks indexed by physical column. Correct for complete and incomplete grids: in
    the incomplete case the per-block long/short split is determined by ``order``,
    which is why the split must be recomputed for each candidate rather than once.
    """
    n = len(ct)
    heights = _column_heights(n, width)
    cols: list[str] = [""] * width
    idx = 0
    for c in order:
        cols[c] = ct[idx : idx + heights[c]]
        idx += heights[c]
    return cols


def _adjacency_of_columns(cols: list[str], table: dict[str, float], floor: float) -> float:
    """Total mean-bigram adjacency over physically-consecutive column pairs.

    For each adjacent physical pair ``(c, c+1)`` average the row-wise bigram
    log-probabilities over the overlapping height, then sum across the grid. This is
    the multiple-anagramming objective evaluated on a concrete physical layout, so it
    is valid for ragged (incomplete) grids too.
    """
    total = 0.0
    for c in range(len(cols) - 1):
        left, right = cols[c], cols[c + 1]
        h = min(len(left), len(right))
        if h == 0:
            continue
        s = 0.0
        for r in range(h):
            s += table.get(left[r] + right[r], floor)
        total += s / h
    return total


def _assemble(cols: list[str], n: int) -> str:
    """Read a physical-column grid back row-by-row into plaintext.

    Skips exhausted short columns (the ragged final row of an incomplete grid),
    the exact inverse of columnar encoding.
    """
    width = len(cols)
    heights = [len(c) for c in cols]
    pos = [0] * width
    out: list[str] = []
    while len(out) < n:
        for c in range(width):
            if pos[c] < heights[c]:
                out.append(cols[c][pos[c]])
                pos[c] += 1
    return "".join(out)


def _split_complete_blocks(ct: str, width: int) -> list[str]:
    """Split a complete-grid ciphertext into its equal-height transmitted blocks."""
    h = len(ct) // width
    return [ct[k * h : (k + 1) * h] for k in range(width)]


def _pair_matrix(
    blocks: list[str], table: dict[str, float], floor: float
) -> list[list[float]]:
    """``M[a][b]`` = mean bigram log-prob of stacking block ``a`` left of block ``b``.

    Precomputed once for the complete-grid case so a full arrangement scores in
    O(width) (sum of consecutive-pair entries) instead of O(n) per evaluation.
    """
    width = len(blocks)
    m = [[0.0] * width for _ in range(width)]
    for a in range(width):
        ba = blocks[a]
        for b in range(width):
            if a == b:
                continue
            bb = blocks[b]
            h = min(len(ba), len(bb))
            if h == 0:
                continue
            s = 0.0
            for r in range(h):
                s += table.get(ba[r] + bb[r], floor)
            m[a][b] = s / h
    return m


def _hill_climb(order: list[int], fitness) -> list[int]:
    """Deterministic best-improvement pairwise-swap polish to a local optimum.

    Repeatedly applies the single swap that most improves ``fitness`` until none does.
    Cleans up the residual local-optimum errors that annealing leaves behind.
    """
    width = len(order)
    cur = list(order)
    cur_score = fitness(cur)
    improved = True
    while improved:
        improved = False
        best_swap = None
        best_gain = 1e-12
        for i in range(width - 1):
            for j in range(i + 1, width):
                cand = list(cur)
                cand[i], cand[j] = cand[j], cand[i]
                g = fitness(cand) - cur_score
                if g > best_gain:
                    best_gain, best_swap = g, (i, j)
        if best_swap is not None:
            i, j = best_swap
            cur[i], cur[j] = cur[j], cur[i]
            cur_score = fitness(cur)
            improved = True
    return cur


# --------------------------------------------------------------------------- #
# Per-width solve.
# --------------------------------------------------------------------------- #


def _solve_width(
    ct: str,
    width: int,
    scorer,
    rng: random.Random,
    restarts: int,
) -> dict | None:
    """Recover the best read-order for a single ``width`` via SA on adjacency fit.

    The search variable is the read-order itself; each candidate is materialised into
    physical columns (the block split depends on the order for incomplete grids) and
    scored by the factored bigram-adjacency proxy, which has a strong single-swap
    gradient. The annealing endpoints (and their reversal, since the row-direction is
    a symmetry of the proxy) are re-ranked by the real hexagram fitness of the
    assembled plaintext. Returns ``dict(score, plaintext, width, order)`` or ``None``.
    """
    n = len(ct)
    if width < 2 or width > n:
        return None
    table, floor = _bigram_logp()

    # Complete grid (W | N): every transmitted block is equal length, so the
    # block-to-physical split is order-independent. Precompute the full physical
    # adjacency matrix M[a][b] once and score arrangements in O(width); this lets us
    # afford many more restarts/iters where the search is hardest. Incomplete grids
    # need a per-candidate split, so they fall back to the direct (O(n)) objective.
    if n % width == 0:
        blocks = _split_complete_blocks(ct, width)
        m = _pair_matrix(blocks, table, floor)

        def fitness(order: list[int]) -> float:
            # order[k] = physical column of block k  =>  physical col c holds block
            # order.index(c); arrange blocks left-to-right by physical position.
            inv = [0] * width
            for k, c in enumerate(order):
                inv[c] = k
            return sum(m[inv[c]][inv[c + 1]] for c in range(width - 1))
    else:

        def fitness(order: list[int]) -> float:
            cols = _physical_columns(ct, width, order)
            return _adjacency_of_columns(cols, table, floor)

    def init_order() -> list[int]:
        return search.shuffled(list(range(width)), rng)

    # The bigram-adjacency proxy is fast and well-shaped but its single global optimum
    # is not always the true order (a near-tie can flip it). So we don't trust one
    # winner: we anneal+hill-climb from many random starts and collect a POOL of
    # distinct high-adjacency local optima, then re-rank the whole pool by the real
    # hexagram fitness of the assembled plaintext (which separates true English from
    # a proxy-optimal-but-scrambled near-miss by a wide margin).
    pool: dict[tuple[int, ...], float] = {}

    def consider(order: list[int]) -> None:
        polished = _hill_climb(order, fitness)
        for o in (polished, [width - 1 - p for p in polished]):
            pool[tuple(o)] = fitness(o)

    iters = max(240, 30 * width)
    for _ in range(restarts):
        cand, _ = search.anneal(
            init=init_order,
            neighbour=search.swap_neighbour,
            score=fitness,
            rng=rng,
            restarts=1,
            iters_per_temp=iters,
            temp0=0.45,
            cooling=0.9,
            min_temp=0.015,
        )
        consider(cand)

    # Re-rank the top adjacency optima (and their row-reversals) by hexagram fitness.
    top = sorted(pool, key=lambda o: pool[o], reverse=True)[:24]
    best: dict | None = None
    for order in (list(o) for o in top):
        cols = _physical_columns(ct, width, order)
        plain = _assemble(cols, n)
        sc = scorer.fitness(plain)
        if best is None or sc > best["score"]:
            best = {
                "score": sc,
                "plaintext": plain,
                "width": width,
                "order": order,
            }
    return best


def solve(ct: str, widths=range(4, 18)) -> dict:
    """Solve a pure columnar transposition by multiple anagramming.

    Parameters
    ----------
    ct:
        Ciphertext (non-letters ignored; case-folded to A-Z).
    widths:
        Candidate grid widths to try (default ``range(4, 18)``).

    Returns
    -------
    ``dict`` with keys ``score`` (hexagram fitness of the winning plaintext),
    ``plaintext``, ``width``, and ``order`` (the recovered read-order, ready for
    ``columnar.decode`` / ``incomplete_columnar.decode``).
    """
    letters = only_letters(ct)
    scorer = resolve_scorer("hexagrams")

    n = len(letters)
    best: dict | None = None
    for width in widths:
        if width < 2 or width > n:
            continue
        # Independent, deterministically-seeded RNG per width so a width's search is
        # reproducible and unaffected by how many widths preceded it (a shared stream
        # would make width-12's restarts depend on the surrounding sweep range).
        rng = random.Random(0xC0FFEE ^ (width * 0x9E3779B1))
        # Wider keys are a harder permutation search; give them more restarts. The
        # complete-grid fast path makes each evaluation O(width), so we can afford a
        # generous budget; incomplete grids are O(n) per eval but rarer/here narrower.
        complete = (n % width) == 0
        if complete:
            restarts = 40 if width <= 12 else (120 if width <= 15 else 250)
        else:
            restarts = 16 if width <= 10 else (40 if width <= 13 else 90)
        cand = _solve_width(letters, width, scorer, rng, restarts)
        if cand is None:
            continue
        if best is None or cand["score"] > best["score"]:
            best = cand
    if best is None:
        return {"score": float("-inf"), "plaintext": letters, "width": 0, "order": []}
    return best


# --------------------------------------------------------------------------- #
# Self-test: plant a width-12 complete columnar of English, assert recovery.
# --------------------------------------------------------------------------- #


def _char_match(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    hits = sum(1 for i in range(n) if a[i] == b[i])
    return hits / max(len(a), len(b))


def _self_test() -> bool:
    from .ciphers.columnar import _encode_letters as _enc_complete

    # >= 288 letters of plain English (no substitution), width 12, W | N.
    pt = (
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG WHILE THE EARLY MORNING SUN "
        "ROSE SLOWLY OVER THE QUIET VILLAGE AND THE PEOPLE WENT ABOUT THEIR WORK "
        "WITH A STEADY AND FAMILIAR RHYTHM THAT HAD NOT CHANGED IN MANY YEARS AND "
        "THE OLD CLOCK IN THE SQUARE STILL KEPT THE SAME PATIENT TIME AS EVER IT "
        "HAD WHEN THE FIRST SETTLERS BUILT THEIR HOMES ALONG THE WINDING RIVER BANK"
    )
    letters = only_letters(pt)
    width = 12
    # Pad to a multiple of width so the plant is a *complete* columnar.
    if len(letters) % width:
        letters = letters[: len(letters) - (len(letters) % width)]
    assert len(letters) >= 288, f"plant too short: {len(letters)}"
    assert len(letters) % width == 0, "plant must be complete columnar"

    rng = random.Random(42)
    true_order = rng.sample(range(width), width)
    ct = _enc_complete(letters, true_order)

    res = solve(ct, widths=range(8, 15))
    recovered = res["plaintext"]
    match = _char_match(letters, recovered)

    print("=== anagram.py self-test (pure columnar, multiple anagramming) ===")
    print(f"plaintext length : {len(letters)}")
    print(f"true width       : {width}   recovered width: {res['width']}")
    print(f"true read-order  : {true_order}")
    print(f"recovered order  : {res['order']}")
    print(f"hexagram fitness : {res['score']:.4f}")
    print(f"char-match       : {match * 100:.1f}%")
    print(f"plaintext  (head): {letters[:72]}")
    print(f"recovered  (head): {recovered[:72]}")
    ok = match >= 0.95
    print(f"RESULT           : {'PASS' if ok else 'FAIL'} (threshold 95%)")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_test() else 1)
