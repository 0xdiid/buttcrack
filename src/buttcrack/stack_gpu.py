"""Batched Held-Karp on the GPU, and the population search it makes possible.

WHY BATCHING AND NOT MICRO-OPTIMISATION
---------------------------------------
Profiled on a width-9 depth-2 stack over 315 letters, one candidate evaluation costs:

    peel outer layers   0.02 ms   ( 1.5%)
    build adjacency     0.10 ms   ( 7.8%)
    Held-Karp           1.13 ms   (90.6%)
                        -------
    total               1.25 ms   ->  ~800 evaluations/second

So making the adjacency build incremental — the obvious optimisation, and the one this
module was originally going to be — is worth at most 8%. The DP is the wall.

Held-Karp is hard to speed up *per instance*: it is 2^w sequential mask steps, each tiny.
But those steps are identical across candidates, so the whole DP vectorises across a BATCH.
One kernel launch per mask handles thousands of candidates at once, and the 2^w launch
overhead is paid once for the entire batch instead of once per candidate. That converts a
latency problem into a throughput problem, which is the shape a GPU wants.

Only scores are computed in batch; the winning path is reconstructed once, on the CPU,
where it costs nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DeepResult:
    score: float
    orders: list[list[int]]
    plaintext: str
    evaluations: int
    seconds: float


def bigram_tensor(tab: list[list[float]], device: str):
    return torch.tensor(tab, dtype=torch.float32, device=device)


def batch_adjacency(blocks: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """``blocks`` [B, w, rows] of letter indices -> ``adj`` [B, w, w].

    ``adj[b, a, c]`` scores column ``a`` immediately left of column ``c``: the summed
    bigram log-probability of the ``rows`` letter pairs they form.
    """
    B, w, rows = blocks.shape
    left = blocks[:, :, None, :].expand(B, w, w, rows)
    right = blocks[:, None, :, :].expand(B, w, w, rows)
    return tab[left.reshape(-1), right.reshape(-1)].view(B, w, w, rows).sum(-1)


def held_karp_batch(adj: torch.Tensor) -> torch.Tensor:
    """Max-weight Hamiltonian path score for every instance in the batch. [B, w, w] -> [B].

    Same recurrence as the scalar solver, but ``dp`` carries a batch dimension so each of
    the 2^w mask steps is one kernel launch covering the whole batch.
    """
    B, w, _ = adj.shape
    size = 1 << w
    NEG = torch.finfo(torch.float32).min / 4
    dp = torch.full((B, size, w), NEG, device=adj.device, dtype=torch.float32)
    starts = torch.tensor([1 << s for s in range(w)], device=adj.device)
    dp[:, starts, torch.arange(w, device=adj.device)] = 0.0

    for mask in range(size):
        cur = dp[:, mask, :]  # [B, w]
        if not bool((cur > NEG / 2).any()):
            continue
        cand = cur[:, :, None] + adj  # [B, last, nxt]
        bits = [k for k in range(w) if not (mask >> k) & 1]
        if not bits:
            continue
        inmask = torch.tensor(
            [k for k in range(w) if (mask >> k) & 1], device=adj.device, dtype=torch.long
        )
        if inmask.numel() == 0:
            continue
        sub = cand[:, inmask, :]  # [B, |mask|, nxt]
        best_next = sub.max(dim=1).values  # [B, nxt]
        for k in bits:
            nm = mask | (1 << k)
            torch.maximum(dp[:, nm, k], best_next[:, k], out=dp[:, nm, k])
    return dp[:, size - 1, :].max(dim=1).values


def _inverse_index_np(n: int, width: int, order: np.ndarray) -> np.ndarray:
    """Vectorised columnar inverse index for a BATCH of orders. [B, w] -> [B, n]."""
    B = order.shape[0]
    rows = n // width
    inv = np.argsort(order, axis=1)  # [B, w]
    r = np.arange(rows)[None, :, None]  # [1, rows, 1]
    idx = inv[:, None, :] * rows + r  # [B, rows, w]
    return idx.reshape(B, n)


def solve_deep_gpu(
    stream: list[int],
    tab: list[list[float]],
    *,
    widths: list[int],
    population: int = 4096,
    generations: int = 40,
    elite: int = 64,
    device: str | None = None,
    rng: random.Random | None = None,
    log=None,
) -> DeepResult:
    """Population search over the OUTER orders, innermost layer solved exactly per candidate.

    Each generation evaluates ``population`` candidates in one batched Held-Karp call, keeps
    the top ``elite``, and repopulates by mutating them (random column swaps) plus a
    fraction of fresh random restarts to keep the pool from collapsing.
    """
    if torch is None:
        raise RuntimeError("solve_deep_gpu requires torch")
    import time

    dev = _device(device)
    rng = rng or random.Random(0)
    t0 = time.time()
    n = len(stream)
    inner_w, outer_ws = widths[0], list(widths[1:])
    if len(outer_ws) != 1 and len(outer_ws) != 2:
        raise ValueError("solve_deep_gpu handles depth 2 and 3 (one or two outer layers)")
    rows = n // inner_w
    st = torch.tensor(stream, dtype=torch.long, device=dev)
    tabt = bigram_tensor(tab, dev)

    def evaluate(pop: list[np.ndarray]) -> torch.Tensor:
        """pop is one [B, w] array per outer layer, innermost-outward."""
        B = pop[0].shape[0]
        idx = np.tile(np.arange(n), (B, 1))
        for w, orders in zip(reversed(outer_ws), reversed(pop), strict=True):
            step = _inverse_index_np(n, w, orders)
            # compose as idx o step (gather the running map THROUGH the new one), matching
            # stack.compose_index. The transposed form idx[step] silently produces a stream
            # that barely depends on the orders at all -- which is what a score plateauing
            # from generation 10, identical across two independent runs, looks like.
            idx = np.take_along_axis(idx, step, axis=1)
        gathered = st[torch.tensor(idx, device=dev)]
        # The innermost columnar's ciphertext is CONTIGUOUS blocks of `rows` letters, one
        # per column -- not a row-major grid. Reshaping the other way round silently yields
        # a well-formed but meaningless adjacency, and the search then optimises noise.
        blocks = gathered.view(B, inner_w, rows)
        return held_karp_batch(batch_adjacency(blocks, tabt))

    def random_pop(B: int) -> list[np.ndarray]:
        return [
            np.array([rng.sample(range(w), w) for _ in range(B)], dtype=np.int64) for w in outer_ws
        ]

    pop = random_pop(population)
    best_score, best_orders, evals = -1e18, None, 0
    for gen in range(generations):
        scores = evaluate(pop).cpu().numpy()
        evals += population
        top = np.argsort(-scores)[:elite]
        if scores[top[0]] > best_score:
            best_score = float(scores[top[0]])
            best_orders = [p[top[0]].tolist() for p in pop]
        if log and gen % 10 == 0:
            log(
                f"    gen {gen:>3}  best {best_score:>10.1f}  ({evals:,} evals, "
                f"{evals / max(time.time() - t0, 1e-9):,.0f}/s)"
            )
        fresh = population // 8
        newpop = []
        for li, w in enumerate(outer_ws):
            parents = pop[li][top]
            reps = np.repeat(parents, max(1, (population - fresh) // elite), axis=0)
            reps = reps[: population - fresh].copy()
            a = np.random.randint(0, w, size=reps.shape[0])
            b = np.random.randint(0, w, size=reps.shape[0])
            r = np.arange(reps.shape[0])
            reps[r, a], reps[r, b] = reps[r, b], reps[r, a]
            rand = np.array([rng.sample(range(w), w) for _ in range(fresh)], dtype=np.int64)
            newpop.append(np.concatenate([reps, rand], axis=0))
        pop = newpop

    # reconstruct the winning plaintext on the CPU
    from .columnar_exact import column_adjacency, held_karp_path
    from .stack import _gather, columnar_inverse_index, compose_index

    idx = list(range(n))
    for w, o in zip(reversed(outer_ws), reversed(best_orders), strict=True):
        idx = compose_index(idx, columnar_inverse_index(n, w, o))
    peeled = _gather(stream, idx)
    blocks = [peeled[j * rows : (j + 1) * rows] for j in range(inner_w)]
    _, path = held_karp_path(column_adjacency(blocks, tab))
    plain = [0] * n
    for c, blk in enumerate(path):
        for i in range(rows):
            plain[i * inner_w + c] = blocks[blk][i]
    inner_order = [0] * inner_w
    for col, blk in enumerate(path):
        inner_order[blk] = col
    return DeepResult(
        best_score,
        [inner_order, *best_orders],
        "".join(chr(65 + v) for v in plain),
        evals,
        time.time() - t0,
    )
