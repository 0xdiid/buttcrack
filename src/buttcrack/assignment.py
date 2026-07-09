"""Maximum-weight assignment (Hungarian / Kuhn-Munkres) — pick one column per row.

A reusable combinatorial primitive that turns up whenever a cryptanalysis step must match
`n` things to `m` slots under a per-pair score without enumerating the `n!` orderings — e.g.
assigning ciphertext rows to key phases under a unigram likelihood (a jigsaw / seriation
step), or matching recovered fragments to positions. It finds the assignment maximising the
total score in `O(n^3)`; the greedy "best column per row" it replaces can be arbitrarily far
from optimal when two rows want the same column.
"""

from __future__ import annotations

import math


def hungarian_max(matrix: list[list[float]]) -> tuple[float, list[int]]:
    """Maximum-weight assignment of each row to a distinct column.

    ``matrix[i][j]`` is the score of assigning row ``i`` to column ``j``. Requires
    ``n = len(matrix) <= m = len(matrix[0])`` (at least as many columns as rows). Returns
    ``(total_score, assign)`` where ``assign[i]`` is the column chosen for row ``i`` (all
    distinct) and ``total_score`` is their sum. This is the standard `O(n^3)` Hungarian
    algorithm run on negated scores (so a minimiser maximises), with the Jonker-Volgenant
    potential update; ties are broken by the first column reaching the minimum.
    """
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    if n == 0 or n > m:
        raise ValueError("assignment needs a non-empty matrix with columns >= rows")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = row assigned to column j (1-indexed; 0 = free)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = -matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assign[p[j] - 1] = j - 1
    total = sum(matrix[i][assign[i]] for i in range(n))
    return total, assign
