"""Known-plaintext (crib) attack on the Hill cipher, and the mod-26 linear solver it rests on.

The Hill cipher is linear, so a little known plaintext breaks it outright: an ``n x n``
key needs only ``n`` independent plaintext/ciphertext blocks to pin every entry. This
module solves that system over ``Z_26`` — which is a *ring*, not a field, so ordinary
Gaussian elimination fails on the zero-divisors ``2`` and ``13``. The fix is the Chinese
Remainder Theorem: solve mod 2 and mod 13 (both fields) independently and recombine.
Because a mod-2 system is often rank-deficient, we enumerate its whole solution space, so
the caller gets *every* key consistent with the crib rather than one arbitrary choice.

On top of that primitive:

* :func:`recover_matrix` — the classic KPA: recover the ``n x n`` matrix from aligned
  known plaintext (the crib need not be at the very start; give its block offset).
* :func:`recover_affine` — the affine generalisation ``c = K p + s[block mod q]`` with an
  unknown period-``q`` additive schedule (a shape that turns up in real layered puzzles),
  all in a chosen index alphabet (plain or keyed).

Public API
----------
``solve_mod26(A, b) -> list[list[int]]``       every ``x`` with ``A x = b (mod 26)``
``recover_matrix(plain, cipher, n, *, alphabet, offset) -> list[matrix]``
``crib_drag(crib, cipher, n, *, alphabet, scorer, top) -> list[{offset, matrix, plaintext, score}]``
``recover_affine(plain, cipher, *, n, q, alphabet, offset) -> list[(matrix, schedule)]``
"""

from __future__ import annotations

import itertools

from .keysources import _alphabet
from .text import only_letters


def _solve_prime(A: list[list[int]], b: list[int], p: int) -> list[list[int]] | None:
    """Every solution of ``A x = b (mod p)`` for prime ``p`` (Gaussian elimination).

    Returns a list of solution vectors (``p ** free`` of them), or ``None`` if the system
    is inconsistent. Rank-deficient systems enumerate the free variables.
    """
    rows = [[x % p for x in row] + [bi % p] for row, bi in zip(A, b)]
    ncols = len(A[0])
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(rows)) if rows[i][c] % p), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = pow(rows[r][c], p - 2, p)
        rows[r] = [(v * inv) % p for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [(rows[i][k] - f * rows[r][k]) % p for k in range(ncols + 1)]
        pivots.append(c)
        r += 1
        if r == len(rows):
            break
    # consistency: any all-zero coefficient row must have zero RHS
    for i in range(len(rows)):
        if not any(rows[i][:ncols]) and rows[i][ncols]:
            return None
    free = [c for c in range(ncols) if c not in pivots]
    piv_row = {c: i for i, c in enumerate(pivots)}
    sols: list[list[int]] = []
    for combo in itertools.product(range(p), repeat=len(free)):
        x = [0] * ncols
        for c, v in zip(free, combo):
            x[c] = v
        for c in pivots:
            row = rows[piv_row[c]]
            x[c] = (row[ncols] - sum(row[k] * x[k] for k in free)) % p
        sols.append(x)
    return sols


def solve_mod26(A: list[list[int]], b: list[int], *, max_solutions: int = 4096) -> list[list[int]]:
    """Every ``x`` with ``A x = b (mod 26)``, via CRT over mod 2 and mod 13.

    ``A`` is an ``m x n`` coefficient matrix (``m`` equations, ``n`` unknowns) and ``b`` an
    ``m``-vector, all integers. Returns the full (possibly empty) solution set, capped at
    ``max_solutions``. Because 26 = 2 x 13 with 2 and 13 prime, ``x`` solves mod 26 iff it
    solves mod 2 and mod 13; we combine each pair of component solutions with CRT
    (``x = 13*x2 + 14*x13 mod 26``, since 13 = 1 mod 2 / 0 mod 13 and 14 = 0 mod 2 / 1 mod 13).
    """
    if not A or not A[0]:
        return []
    s2 = _solve_prime(A, b, 2)
    s13 = _solve_prime(A, b, 13)
    if s2 is None or s13 is None:
        return []
    n = len(A[0])
    out: list[list[int]] = []
    for x2 in s2:
        for x13 in s13:
            out.append([(13 * x2[k] + 14 * x13[k]) % 26 for k in range(n)])
            if len(out) >= max_solutions:
                return out
    return out


def _det_coprime(matrix: list[list[int]], n: int) -> bool:
    import math

    if n == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return math.gcd(det % 26, 26) == 1


def _indices(text: str, alphabet: str) -> list[int]:
    pos = {ch: i for i, ch in enumerate(alphabet)}
    return [pos[ch] for ch in only_letters(text).upper()]


def recover_matrix(
    plain: str,
    cipher: str,
    n: int,
    *,
    alphabet: str = "STD",
    offset: int = 0,
) -> list[list[list[int]]]:
    """Recover every invertible ``n x n`` encryption matrix consistent with a crib.

    ``plain`` is known plaintext aligned to ``cipher`` starting at block ``offset`` (so the
    crib covers ciphertext blocks ``offset, offset+1, ...``). Encryption is ``c = K p`` on
    column blocks in the index of ``alphabet`` (``"STD"``, ``"KRYPTOS"``, or a permutation).
    Returns the invertible key matrices that reproduce the crib (usually one; more if the
    crib is short enough to leave the key underdetermined).
    """
    alpha = _alphabet(alphabet)
    P = _indices(plain, alpha)
    C = _indices(cipher, alpha)
    kb = len(P) // n
    if kb < n:
        raise ValueError(f"need at least {n} full crib blocks for an {n}x{n} key; got {kb}")
    pblocks = [P[n * i : n * i + n] for i in range(kb)]
    cblocks = [C[n * (offset + i) : n * (offset + i) + n] for i in range(kb)]
    if len(cblocks[-1]) < n:
        raise ValueError("crib runs past the end of the ciphertext at that offset")
    # Each key ROW r is independent: sum_j K[r][j] p[i][j] = c[i][r] over crib blocks i.
    row_solutions: list[list[list[int]]] = []
    for r in range(n):
        A = [pblocks[i] for i in range(kb)]
        b = [cblocks[i][r] for i in range(kb)]
        sols = solve_mod26(A, b)
        if not sols:
            return []
        row_solutions.append(sols)
    out: list[list[list[int]]] = []
    seen: set[tuple] = set()
    for rows in itertools.product(*row_solutions):
        matrix = [list(row) for row in rows]
        key = tuple(map(tuple, matrix))
        if key in seen or not _det_coprime(matrix, n):
            continue
        seen.add(key)
        out.append(matrix)
    return out


def crib_drag(
    crib: str,
    cipher: str,
    n: int,
    *,
    alphabet: str = "STD",
    scorer=None,
    top: int = 10,
) -> list[dict]:
    """Slide a crib across every block-aligned offset, recover the Hill key at each, decrypt the
    whole message, and rank the results.

    Use when you know a probable plaintext word/phrase but NOT where it sits. This is
    :func:`recover_matrix` (which needs the offset) wrapped in an offset sweep + full decrypt +
    ranking. The crib must be at least ``n`` full blocks (``n*n`` letters) to pin an ``n x n`` key.

    ``scorer`` is any ``callable(str) -> float`` (higher = better); it defaults to English
    quadgram fitness, but pass your own to recognise a **non-English payload** (e.g. gzip
    compressibility, a coordinate/keyword token detector, minimum letter entropy) — the lesson
    being that the right decryption of a route/key/coordinate plaintext will not score as English.

    Returns up to ``top`` dicts ``{offset, matrix, plaintext, score}`` sorted best-first (one per
    distinct recovered key). Empty if the crib is too short or never yields an invertible key.
    """
    from .ciphers.hill import _inverse  # local import avoids any import-order coupling

    if scorer is None:
        from .scoring import get_scorer

        _sc = get_scorer("quadgrams", "english")
        scorer = _sc.score
    alpha = _alphabet(alphabet)
    idx = {ch: i for i, ch in enumerate(alpha)}
    crib_l = only_letters(crib)
    cipher_l = only_letters(cipher)
    crib_blocks = len(crib_l) // n
    total_blocks = len(cipher_l) // n
    if crib_blocks < n:
        raise ValueError(f"crib needs >= {n} full blocks ({n * n} letters) for an {n}x{n} key")
    C = [idx[c] for c in cipher_l]
    results: list[dict] = []
    seen: set[tuple] = set()
    for offset in range(0, total_blocks - crib_blocks + 1):
        for K in recover_matrix(crib_l, cipher_l, n, alphabet=alphabet, offset=offset):
            key = tuple(map(tuple, K))
            if key in seen:
                continue
            seen.add(key)
            try:
                inv = _inverse(K, n)
            except ValueError:
                continue
            out: list[str] = []
            for start in range(0, len(C) - n + 1, n):
                block = C[start : start + n]
                for row in inv:
                    out.append(alpha[sum(row[k] * block[k] for k in range(n)) % 26])
            pt = "".join(out)
            results.append(
                {"offset": offset, "matrix": K, "plaintext": pt, "score": scorer(pt)}
            )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top]


def recover_affine(
    plain: str,
    cipher: str,
    *,
    n: int = 3,
    q: int = 1,
    alphabet: str = "STD",
    offset: int = 0,
) -> list[tuple[list[list[int]], list[list[int]]]]:
    """Recover ``(K, schedule)`` for ``c = K p + s[block mod q]`` from a crib.

    Generalises :func:`recover_matrix` with an unknown period-``q`` additive keystream on
    the *ciphertext* blocks (``schedule[t]`` is the length-``n`` offset for blocks
    ``t mod q``). Each key row plus its ``q`` schedule entries is one linear system in
    ``n + q`` unknowns, so the crib must span enough blocks per class. Returns a list of
    ``(matrix, schedule)`` pairs with invertible ``K``.
    """
    alpha = _alphabet(alphabet)
    P = _indices(plain, alpha)
    C = _indices(cipher, alpha)
    kb = len(P) // n
    pblocks = [P[n * i : n * i + n] for i in range(kb)]
    cblocks = [C[n * (offset + i) : n * (offset + i) + n] for i in range(kb)]
    if any(len(cb) < n for cb in cblocks):
        raise ValueError("crib runs past the end of the ciphertext at that offset")
    if kb < n + q:
        raise ValueError(f"need at least {n + q} crib blocks for n={n}, q={q}; got {kb}")
    row_solutions: list[list[list[int]]] = []
    for r in range(n):
        # unknowns: K[r][0..n-1], then schedule[0..q-1][r]; block i uses class (offset+i)%q
        A: list[list[int]] = []
        b: list[int] = []
        for i in range(kb):
            eq = list(pblocks[i]) + [0] * q
            eq[n + (offset + i) % q] = 1
            A.append(eq)
            b.append(cblocks[i][r])
        sols = solve_mod26(A, b)
        if not sols:
            return []
        row_solutions.append(sols)
    out: list[tuple[list[list[int]], list[list[int]]]] = []
    seen: set[tuple] = set()
    for rows in itertools.product(*row_solutions):
        matrix = [list(row[:n]) for row in rows]
        if not _det_coprime(matrix, n):
            continue
        schedule = [[rows[r][n + t] for r in range(n)] for t in range(q)]
        key = (tuple(map(tuple, matrix)), tuple(map(tuple, schedule)))
        if key in seen:
            continue
        seen.add(key)
        out.append((matrix, schedule))
    return out
