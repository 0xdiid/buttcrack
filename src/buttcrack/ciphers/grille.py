"""Turning (Fleissner) grille transposition.

A square ``N x N`` grille (``N`` even) has ``N*N/4`` holes punched in it. With
the grille laid over an empty grid, the first quarter of the message is written
through the holes in reading order (top-to-bottom, then left-to-right). The
grille is then turned 90 degrees clockwise and the next quarter written in;
after four turns every cell has been filled exactly once. Removing the grille
and reading the grid horizontally gives the ciphertext.

Key format
----------
The key is the set of hole positions as 1-indexed cell numbers, numbered
across each row in turn (cell ``r*N + c + 1`` for the cell at row ``r``,
column ``c``), exactly as the ACA reports a grille solution. They may be
separated by spaces or commas, e.g. ``"1 8 10 12"`` (the ACA 4x4 example) or
``"1,8,10,12"``. The grid size ``N`` is inferred from the hole count
(``N = sqrt(4 * holes)``). An optional ``width N;`` prefix may pin ``N``
explicitly, e.g. ``"width 4; 1 8 10 12"``.

For the holes to form a valid grille, the four rotations of the hole set must
tile the whole grid without overlap; the parser rejects keys that do not.

Published vector (ACA cipher sheet, aca.info/ciphers/Grille.pdf): plaintext
``THE TURNING GRILLE`` with the 4x4 grille reported as ``1 8 10 12`` encodes to
``TILUN RGHGE LTENI R`` (``TILUNRGHGELTENIR``).
"""

from __future__ import annotations

import math
import time
from itertools import product

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

#: pad letter used to complete the final grille block
PAD = "X"


def _rot90_cw(r: int, c: int, n: int) -> tuple[int, int]:
    """Where cell ``(r, c)`` lands after turning the grille 90 degrees clockwise."""
    return (c, n - 1 - r)


def _infer_size(num_holes: int) -> int:
    n = int(round(math.sqrt(4 * num_holes)))
    if n <= 0 or n % 2 != 0 or n * n != 4 * num_holes:
        raise ValueError(f"grille hole count {num_holes} is not N*N/4 for any even N")
    return n


def _parse_key(key: str) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(N, holes)`` from a key of 1-indexed cell numbers.

    ``holes`` is the Position-1 hole set as ``(row, col)`` pairs. An optional
    leading ``width N;`` (or ``w N;``) token pins the grid size.
    """
    s = str(key).strip()
    if not s:
        raise ValueError("grille key must list the hole cell numbers")
    pinned: int | None = None
    # Split off an optional 'width N' / 'w N' prefix segment.
    head, sep, tail = s.partition(";")
    if sep:
        toks = head.replace(",", " ").split()
        if len(toks) == 2 and toks[0].lower() in ("width", "w") and toks[1].isdigit():
            pinned = int(toks[1])
            s = tail.strip()
    nums = [int(tok) for tok in s.replace(",", " ").split() if tok]
    if not nums:
        raise ValueError("grille key must list the hole cell numbers")
    if len(set(nums)) != len(nums):
        raise ValueError("grille key has duplicate hole numbers")
    n = pinned if pinned is not None else _infer_size(len(nums))
    if n <= 0 or n % 2 != 0:
        raise ValueError("grille grid size N must be a positive even integer")
    if len(nums) != n * n // 4:
        raise ValueError(f"a {n}x{n} grille needs exactly {n * n // 4} holes, got {len(nums)}")
    holes: list[tuple[int, int]] = []
    for cell in nums:
        if not (1 <= cell <= n * n):
            raise ValueError(f"hole {cell} out of range for a {n}x{n} grid")
        idx = cell - 1
        holes.append((idx // n, idx % n))
    _validate_grille(holes, n)
    return n, holes


def _validate_grille(holes: list[tuple[int, int]], n: int) -> None:
    """Ensure the four rotations of ``holes`` tile the grid exactly once."""
    covered: set[tuple[int, int]] = set()
    current = list(holes)
    for _ in range(4):
        for r, c in current:
            if (r, c) in covered:
                raise ValueError("grille holes overlap under rotation; not a valid grille")
            covered.add((r, c))
        current = [_rot90_cw(r, c, n) for r, c in current]
    if len(covered) != n * n:
        raise ValueError("grille holes do not cover the whole grid under rotation")


def _block_size(n: int) -> int:
    return n * n


def _encode_block(letters: str, holes: list[tuple[int, int]], n: int) -> str:
    """Encode exactly ``n*n`` letters through one grille block."""
    grid: list[list[str]] = [["" for _ in range(n)] for _ in range(n)]
    current = list(holes)
    idx = 0
    for _ in range(4):
        for r, c in sorted(current):
            grid[r][c] = letters[idx]
            idx += 1
        current = [_rot90_cw(r, c, n) for r, c in current]
    return "".join(grid[r][c] for r in range(n) for c in range(n))


def _decode_block(cipher: str, holes: list[tuple[int, int]], n: int) -> str:
    """Decode exactly ``n*n`` ciphertext letters through one grille block."""
    grid = [[cipher[r * n + c] for c in range(n)] for r in range(n)]
    current = list(holes)
    out: list[str] = []
    for _ in range(4):
        for r, c in sorted(current):
            out.append(grid[r][c])
        current = [_rot90_cw(r, c, n) for r, c in current]
    return "".join(out)


def _encode_letters(letters: str, holes: list[tuple[int, int]], n: int) -> str:
    block = _block_size(n)
    if not letters:
        return ""
    blocks = math.ceil(len(letters) / block)
    padded = letters + PAD * (blocks * block - len(letters))
    return "".join(
        _encode_block(padded[i * block : (i + 1) * block], holes, n) for i in range(blocks)
    )


def _decode_letters(cipher: str, holes: list[tuple[int, int]], n: int) -> str:
    block = _block_size(n)
    if not cipher:
        return ""
    blocks = math.ceil(len(cipher) / block)
    padded = cipher + PAD * (blocks * block - len(cipher))
    return "".join(
        _decode_block(padded[i * block : (i + 1) * block], holes, n) for i in range(blocks)
    )


def _orbits(n: int) -> list[list[tuple[int, int]]]:
    """Partition the grid cells into rotation orbits (each of size 4)."""
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = []
            p = (r, c)
            for _ in range(4):
                orbit.append(p)
                seen.add(p)
                p = _rot90_cw(p[0], p[1], n)
            orbits.append(orbit)
    return orbits


def _holes_to_key(holes: list[tuple[int, int]], n: int) -> str:
    return " ".join(str(r * n + c + 1) for r, c in sorted(holes))


class Grille(Cipher):
    """Turning (Fleissner) grille transposition.

    The key is the grille's hole positions as 1-indexed cell numbers (ACA
    convention), e.g. ``"1 8 10 12"`` for the 4x4 grille.
    """

    name = "grille"
    aliases = ("turninggrille", "fleissner")
    description = "Turning (Fleissner) grille transposition; key is the hole cell numbers."
    key_format = "1-indexed hole cell numbers (space/comma-separated), e.g. 1 8 10 12"
    key_example = "1 8 10 12"
    complexity = 4

    # A transposition only reorders letters, so encode/decode operate on a clean
    # uppercase letter stream (no reflow, which would leak plaintext word lengths).
    def encode(self, text: str, key: str) -> str:
        n, holes = _parse_key(key)
        return _encode_letters(only_letters(text), holes, n)

    def decode(self, text: str, key: str) -> str:
        n, holes = _parse_key(key)
        return _decode_letters(only_letters(text), holes, n)

    def crack(
        self,
        text: str,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Brute force the grille for small grids.

        A valid ``N x N`` grille picks one cell from each of the ``N*N/4``
        rotation orbits, giving ``4**(N*N/4)`` grilles. That is tiny for ``N=4``
        (256) and tractable for ``N=6`` (4096), so for each candidate grid size
        we enumerate every valid grille, decode, and rank by the n-gram scorer.
        Larger grids blow up combinatorially and are skipped.
        """
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        # Candidate grid sizes: even N whose block divides the text length, so
        # there are no pad artefacts. Cap N to keep 4**(N*N/4) tractable.
        if opts.get("width"):
            sizes = [int(opts["width"])]
        else:
            max_n = int(opts.get("max_size", 6))
            sizes = [n for n in range(2, max_n + 1, 2) if len(letters) % (n * n) == 0]
        deadline = (time.monotonic() + timeout) if timeout else None

        candidates: list[Candidate] = []
        truncated = False
        for n in sizes:
            if n % 2 != 0 or len(letters) % (n * n) != 0:
                continue
            orbits = _orbits(n)
            for choice in product(*orbits):
                if deadline and time.monotonic() > deadline:
                    truncated = True
                    break
                holes = list(choice)
                plain = _decode_letters(letters, holes, n)
                clean = plain.rstrip(PAD) or plain
                key = _holes_to_key(holes, n)
                candidates.append(
                    Candidate(
                        plaintext=clean,
                        cipher=self.name,
                        key=key,
                        score=scorer.score(clean),
                        confidence=scorer.confidence(clean),
                        meta={"size": n},
                    )
                )
            if truncated:
                break
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated and out:
            out[-1].meta["timeout_truncated"] = True
        return out
