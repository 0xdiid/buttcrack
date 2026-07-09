"""Nihilist transposition cipher.

A double (columns-then-rows) keyed transposition on a square ``n x n`` grid,
where the **same** numeric key permutes both the columns and the rows. The key
is a permutation of ``1..n``; the plaintext is written into the grid by rows
(padded to a perfect square), the columns are permuted so their key labels are
ascending, then the rows are permuted the same way, and the ciphertext is read
off the resulting square. Take-off is by columns by default (the ACA "by
columns" convention from *Elcy*); ``takeoff="rows"`` is also supported.

Reference: American Cryptogram Association, *The ACA and You*,
``NihilistTransposition.pdf``, citing Elcy ch. IV p.18.
"""

from __future__ import annotations

import time
from itertools import permutations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

PAD = "X"


def _parse_key(key) -> list[int]:
    """Parse a key into a 0-based permutation of ``range(n)``.

    Accepts a digit string like ``"2134"``, a separated list like ``"2,1,3,4"``,
    or anything iterable of integers. The key is a permutation of ``1..n``;
    it is returned shifted to ``0..n-1`` (so the smallest label becomes 0).
    """
    s = str(key).strip()
    if s and all(ch.isdigit() or ch in ", " for ch in s):
        if "," in s or " " in s:
            nums = [int(x) for x in s.replace(",", " ").split()]
        else:
            nums = [int(ch) for ch in s]
    else:
        raise ValueError("nihilist-transposition key must be a numeric permutation")
    if not nums:
        raise ValueError("nihilist-transposition key is empty")
    lo = min(nums)
    zeroed = [x - lo for x in nums]
    if sorted(zeroed) != list(range(len(zeroed))):
        raise ValueError(f"key must be a permutation of {lo}..{lo + len(nums) - 1}")
    return zeroed


def _pad_to(letters: str, n: int) -> str:
    """Pad ``letters`` to fill an ``n x n`` grid (length ``n*n``)."""
    if len(letters) > n * n:
        raise ValueError(
            f"text has {len(letters)} letters, too many for an {n}x{n} grid (max {n * n})"
        )
    return letters.ljust(n * n, PAD)


def _encode_grid(letters: str, key: list[int], takeoff: str) -> str:
    n = len(key)
    grid = [list(letters[i * n : (i + 1) * n]) for i in range(n)]
    # new column position p holds the original column whose key label ranks p-th
    col_order = sorted(range(n), key=lambda j: key[j])
    sq2 = [[grid[r][col_order[c]] for c in range(n)] for r in range(n)]
    row_order = sorted(range(n), key=lambda i: key[i])
    sq3 = [sq2[row_order[i]] for i in range(n)]
    if takeoff == "rows":
        return "".join("".join(r) for r in sq3)
    return "".join(sq3[r][c] for c in range(n) for r in range(n))


def _decode_grid(cipher: str, key: list[int], takeoff: str) -> str:
    n = len(key)
    sq3: list[list[str]] = [[""] * n for _ in range(n)]
    k = 0
    if takeoff == "rows":
        for r in range(n):
            for c in range(n):
                sq3[r][c] = cipher[k]
                k += 1
    else:
        for c in range(n):
            for r in range(n):
                sq3[r][c] = cipher[k]
                k += 1
    row_order = sorted(range(n), key=lambda i: key[i])
    sq2: list[list[str]] = [[""] * n for _ in range(n)]
    for i in range(n):
        sq2[row_order[i]] = sq3[i]
    col_order = sorted(range(n), key=lambda j: key[j])
    grid: list[list[str]] = [[""] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            grid[r][col_order[c]] = sq2[r][c]
    return "".join("".join(r) for r in grid)


class NihilistTransposition(Cipher):
    name = "nihilist-transposition"
    aliases = ("nihilist-trans", "nihtrans")
    description = (
        "Double (columns-then-rows) keyed transposition on a square grid; "
        "one numeric key permutes both columns and rows."
    )
    key_format = "numeric permutation of 1..n (n = grid side), e.g. 2,1,3,4,5,6,7"
    key_example = "2,1,3,4,5,6,7"
    complexity = 3

    # Transposition only reorders letters, so it cannot preserve word spacing;
    # encode/decode operate on a clean uppercase letter stream (no reflow, which
    # would leak the plaintext's word lengths into the ciphertext).
    def encode(self, text: str, key: str) -> str:
        perm = _parse_key(key)
        n = len(perm)
        letters = _pad_to(only_letters(text), n)
        return _encode_grid(letters, perm, "columns")

    def decode(self, text: str, key: str) -> str:
        perm = _parse_key(key)
        letters = only_letters(text)
        n = len(perm)
        if len(letters) != n * n:
            raise ValueError(f"ciphertext length {len(letters)} is not {n}x{n} for key of size {n}")
        return _decode_grid(letters, perm, "columns")

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Keyless crack.

        The period ``n`` is essentially fixed by the ciphertext length
        (``n = ceil(sqrt(len))``), so we only search the single ``n``-permutation
        key. For small ``n`` (``<= 6``) we exhaust ``n!``; for larger ``n`` we
        hill-climb over random restarts with key-element swaps. Both take-off
        conventions ("columns" and "rows") are tried.
        """
        import random

        letters = only_letters(text)
        if len(letters) < 4:
            return []
        # Only square ciphertexts are decodable; pad-on-encode means cracked
        # text is generally an exact square, so require that.
        n = 1
        while n * n < len(letters):
            n += 1
        if n * n != len(letters):
            return []
        if n > 10:
            return []

        rng = rng or random.Random(0xC1A551C)
        takeoffs = ("columns", "rows")
        deadline = (time.monotonic() + timeout) if timeout else None
        seen: dict[tuple, Candidate] = {}

        def consider(perm: list[int], takeoff: str) -> None:
            plain = _decode_grid(letters, perm, takeoff)
            human_key = ",".join(str(x + 1) for x in perm)
            seen[(human_key, takeoff)] = Candidate(
                plaintext=plain,
                cipher=self.name,
                key=human_key,
                score=scorer.score(plain),
                confidence=scorer.confidence(plain),
                meta={"n": n, "takeoff": takeoff},
            )

        if n <= 6:
            for perm_t in permutations(range(n)):
                if deadline and time.monotonic() > deadline:
                    break
                for takeoff in takeoffs:
                    consider(list(perm_t), takeoff)
        else:
            restarts = int(opts.get("restarts", 30))
            for _ in range(restarts):
                if deadline and time.monotonic() > deadline:
                    break
                for takeoff in takeoffs:
                    perm = list(range(n))
                    rng.shuffle(perm)
                    best_score = scorer.score(_decode_grid(letters, perm, takeoff))
                    improved = True
                    while improved:
                        if deadline and time.monotonic() > deadline:
                            break
                        improved = False
                        for i in range(n):
                            for j in range(i + 1, n):
                                cand = perm[:]
                                cand[i], cand[j] = cand[j], cand[i]
                                sc = scorer.score(_decode_grid(letters, cand, takeoff))
                                if sc > best_score:
                                    best_score, perm, improved = sc, cand, True
                    consider(perm, takeoff)

        candidates = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return candidates[:top]
