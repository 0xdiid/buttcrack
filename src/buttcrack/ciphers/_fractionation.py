"""Shared blind two-phase crack for the ADFGX/ADFGVX fractionation ciphers.

Both ciphers are *fractionation then columnar transposition*, and the two layers
peel in order without a crib:

1. **Transposition**, by a mapping-independent statistic. When the columns are
   un-transposed correctly, consecutive symbol pairs reconstitute the original
   digraphs, whose frequency profile is a 1:1 substitution of English letters — so
   the **digraph index-of-coincidence ≈ 0.066**; a wrong order mixes the halves of
   different letters and flattens it. Annealing the column order to maximise
   digraph-IoC recovers the transposition *without knowing the square* (IoC is
   invariant to the digraph→letter relabelling).
2. **Square**, as a simple substitution. The recovered digraph stream is a
   monoalphabetic substitution over the square's symbols; anneal the digraph→symbol
   map on the quadgram score.

Nested and hard: reliable only on long messages and not guaranteed.
"""

from __future__ import annotations

import time

from .. import search
from ..result import Candidate
from ..scoring import NgramScorer
from ..text import reflow


def _column_lengths(n: int, width: int) -> list[int]:
    """Per-column row counts for an ``n``-char message in ``width`` columns.

    Earlier columns (lower index) take the remainder when ``n`` doesn't divide evenly.
    """
    full_rows, extra = divmod(n, width)
    return [full_rows + (1 if c < extra else 0) for c in range(width)]


def untranspose(cipher: str, order: list[int]) -> str:
    """Undo a columnar transposition given the column read-order."""
    width = len(order)
    n = len(cipher)
    lengths = _column_lengths(n, width)
    columns: list[str] = [""] * width
    idx = 0
    for c in order:
        columns[c] = cipher[idx : idx + lengths[c]]
        idx += lengths[c]
    pos = [0] * width
    out: list[str] = []
    for i in range(n):
        c = i % width
        out.append(columns[c][pos[c]])
        pos[c] += 1
    return "".join(out)


def _block_ioc(stream: str, block: int) -> float:
    """Index of coincidence of consecutive non-overlapping ``block``-char chunks."""
    counts: dict[str, int] = {}
    n = 0
    for i in range(0, len(stream) - block + 1, block):
        chunk = stream[i : i + block]
        counts[chunk] = counts.get(chunk, 0) + 1
        n += 1
    if n < 2:
        return 0.0
    return sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))


def digraph_ioc(stream: str) -> float:
    """Index of coincidence of consecutive non-overlapping digraphs.

    ~0.066 when the stream is correctly un-transposed (digraphs = a 1:1 substitution
    of English letters); flatter when the transposition is wrong. Mapping-independent.
    """
    return _block_ioc(stream, 2)


def _transposition_score(stream: str) -> float:
    """Fitness for the un-transposition: digraph-IoC + a reading-order tiebreak.

    Digraph-IoC pins the column *grouping* but is sequence-invariant, so a wrong
    column *order* with the same digraph multiset ties the true one. The tetragraph
    (4-char block = digraph-bigram) IoC is sequence-sensitive and still
    mapping-independent — it's higher in the true reading order because consecutive
    digraphs reflect English letter-bigram lumpiness, which a scramble flattens. The
    digraph term dominates (its spread across groupings is ~10x the tetra term), so
    the tetra term only decides among otherwise-equal groupings.
    """
    return digraph_ioc(stream) + 0.1 * _block_ioc(stream, 4)


def _anneal_transposition(
    stream: str, width: int, rng, deadline: float | None
) -> tuple[list[int], float]:
    """Recover a column read-order maximising the transposition score."""

    def fitness(order: list[int]) -> float:
        return _transposition_score(untranspose(stream, order))

    def init_order() -> list[int]:
        return search.shuffled(list(range(width)), rng)

    best, _ = search.anneal(
        init=init_order,
        neighbour=search.swap_neighbour,
        score=fitness,
        rng=rng,
        restarts=8,
        iters_per_temp=160,
        temp0=0.02,
        cooling=0.9,
        min_temp=0.0005,
        deadline=deadline,
    )
    return best, digraph_ioc(untranspose(stream, best))


def _candidate_orders(
    stream: str, width: int, rng, deadline: float | None, k: int
) -> list[list[int]]:
    """Up to ``k`` distinct column orders, best transposition-score first.

    Digraph-IoC pins the grouping but ties on reading order; the tetragraph tiebreak
    usually but not always settles it, so we keep several top candidates and let
    phase 2's quadgram score (the ground truth) choose the readable one.
    """
    seen: dict[tuple[int, ...], float] = {}
    runs = max(1, k)
    sub = deadline
    for r in range(runs):
        if deadline is not None:
            now = time.monotonic()
            if now > deadline:
                break
            sub = now + (deadline - now) / (runs - r)
        order, _ = _anneal_transposition(stream, width, rng, sub)
        seen[tuple(order)] = _transposition_score(untranspose(stream, order))
    ranked = sorted(seen, key=lambda o: seen[o], reverse=True)
    return [list(o) for o in ranked[:k]]


def _solve_substitution(
    digraph_stream: str,
    scorer: NgramScorer,
    rng,
    deadline: float | None,
    target_symbols: str,
) -> tuple[str, dict[str, str]]:
    """Solve the recovered digraph stream as a simple substitution (the square).

    Each distinct digraph maps to one plaintext symbol; anneal that bijection on the
    quadgram score. Returns (plaintext, digraph->symbol map).
    """
    pairs = [digraph_stream[i : i + 2] for i in range(0, len(digraph_stream) - 1, 2)]
    appear = sorted(set(pairs))
    k = len(appear)
    if k == 0:
        return "", {}
    idx = {d: i for i, d in enumerate(appear)}
    coded = [idx[p] for p in pairs]
    # Plaintext is overwhelmingly letters; search only the letters of the square's
    # symbol set unless more distinct digraphs appear than there are letters (a
    # digit-heavy plaintext). A wider pool just dilutes the search.
    letters = [s for s in target_symbols if "A" <= s <= "Z"]
    pool = list(target_symbols) if k > len(letters) else letters

    def decode(state: list[str]) -> str:
        return "".join(state[c] for c in coded)

    def fitness(state: list[str]) -> float:
        return scorer.score(decode(state))

    def init_state() -> list[str]:
        return search.shuffled(pool, rng)[:k]

    def neighbour(state: list[str], r) -> list[str]:
        out = list(state)
        # When fewer symbols are assigned than the pool holds, half the moves swap an
        # assigned symbol for an unused one so every pool symbol stays reachable; the
        # rest swap two assignments (the standard simple-substitution move).
        if k < len(pool) and r.random() < 0.5:
            unused = [s for s in pool if s not in out]
            out[r.randrange(k)] = r.choice(unused)
        else:
            i, j = r.randrange(k), r.randrange(k)
            out[i], out[j] = out[j], out[i]
        return out

    best, _ = search.anneal(
        init=init_state,
        neighbour=neighbour,
        score=fitness,
        rng=rng,
        restarts=12,
        iters_per_temp=300,
        temp0=8.0,
        cooling=0.92,
        min_temp=0.05,
        deadline=deadline,
    )
    mapping = {appear[i]: best[i] for i in range(k)}
    return decode(best), mapping


def two_phase_crack(
    text: str,
    stream: str,
    *,
    scorer: NgramScorer,
    rng,
    deadline: float | None,
    name: str,
    target_symbols: str,
    all_digraphs: list[str],
    min_symbols: int,
    opts: dict,
) -> list[Candidate]:
    """Blind two-phase (transposition then square) recovery; ``[]`` if too short."""
    n = len(stream)
    if n < min_symbols or n % 2:
        return []

    forced = opts.get("width")
    max_width = int(opts.get("max_width", 12))
    widths = [int(forced)] if forced else range(2, max_width + 1)
    widths = [w for w in widths if 2 <= w <= n // 2]
    if not widths:
        return []

    # Budget: ~40% to find the width/grouping, ~60% to nail the order + square.
    p1_deadline = deadline
    if deadline is not None:
        now = time.monotonic()
        p1_deadline = now + 0.4 * (deadline - now)

    # Phase 1a: pick the width by digraph-IoC (grouping quality), one anneal per width.
    best_trans: tuple[float, int, list[int]] | None = None  # (ioc, width, order)
    sa_left = len(widths)
    for width in widths:
        if p1_deadline and time.monotonic() > p1_deadline:
            break
        sub = p1_deadline
        if p1_deadline is not None and sa_left > 0:
            sub = time.monotonic() + (p1_deadline - time.monotonic()) / sa_left
        sa_left -= 1
        order, ioc = _anneal_transposition(stream, width, rng, sub)
        if best_trans is None or ioc > best_trans[0]:
            best_trans = (ioc, width, order)
    if best_trans is None:
        return []
    best_ioc, width, _ = best_trans

    # Phase 1b/2: gather a few candidate column orders for that width and solve the
    # square for each, keeping the best by quadgram and stopping once one reads
    # clearly as English (resolves the reading-order tie digraph-IoC can't).
    half = time.monotonic() + 0.5 * (deadline - time.monotonic()) if deadline else None
    orders = _candidate_orders(stream, width, rng, half, k=4) or [best_trans[2]]
    best_cand: Candidate | None = None
    for i, order in enumerate(orders):
        sub2 = deadline
        if deadline is not None:
            now = time.monotonic()
            if now > deadline and best_cand is not None:
                break
            sub2 = now + (deadline - now) / (len(orders) - i)
        plain, mapping = _solve_substitution(
            untranspose(stream, order), scorer, rng, sub2, target_symbols
        )
        if not plain:
            continue
        cand = Candidate(
            plaintext=reflow(text, plain),
            cipher=name,
            key=None,  # recovered (square, order) isn't expressible as a keyword pair
            score=scorer.score(plain),
            confidence=scorer.confidence(plain),
            meta={
                "width": width,
                "read_order": order,
                "digraph_ioc": round(best_ioc, 4),
                "square": "".join(mapping.get(d, "?") for d in all_digraphs),
                "method": "two-phase-anneal",
            },
        )
        if best_cand is None or cand.score > best_cand.score:
            best_cand = cand
        if cand.confidence >= 0.85:  # clearly readable -> reading order is right
            break
    return [best_cand] if best_cand else []
