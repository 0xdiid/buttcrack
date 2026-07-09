"""Redefence: a rail fence whose rails are read in a permuted order, with offset.

The plaintext is written in the usual rail-fence zigzag, but two extra degrees
of freedom are added on top of plain rail fence:

* an **offset** ``O`` that shifts where in the up-down cycle the first letter
  lands (the starting phase of the triangular wave, period ``2R-2``); and
* a **read order** -- instead of concatenating the rails top-rail-first, the
  rows are emitted in a key-given permutation of the rails.

Plain rail fence is the special case ``offset=0`` with the identity read order.

Key format
----------
``"R:O:p1,p2,...,pR"`` where ``R`` is the rail count, ``O`` the offset
(``0..2R-3``), and ``p1..pR`` a permutation of ``1..R`` giving, for each rail in
natural order, the position at which it is read out (1 = read first). For
example ``"3:0:3,1,2"`` means rail1 is read 3rd, rail2 1st, rail3 2nd, so the
ciphertext is ``rail2 + rail3 + rail1``. A bare permutation (e.g. ``"3,1,2"``)
is also accepted and assumes ``offset=0`` with ``R`` inferred from its length.
"""

from __future__ import annotations

import time
from itertools import permutations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher


def _zigzag_pattern(n: int, rails: int, offset: int) -> list[int]:
    """Rail index (0-based) for each of ``n`` positions, given offset phase."""
    if rails < 2:
        return [0] * n
    period = 2 * rails - 2
    out = []
    for i in range(n):
        phase = (i + offset) % period
        out.append(phase if phase < rails else period - phase)
    return out


def _read_positions(order: list[int]) -> list[int]:
    """``order[r]`` (1-based read position per rail) -> rail indices in read order."""
    rails = len(order)
    # rail r is read at position order[r]; invert to "which rail at each position".
    by_pos = sorted(range(rails), key=lambda r: order[r])
    return by_pos


def _encode_letters(letters: str, rails: int, offset: int, order: list[int]) -> str:
    pattern = _zigzag_pattern(len(letters), rails, offset)
    rows: list[list[str]] = [[] for _ in range(max(rails, 1))]
    for ch, r in zip(letters, pattern, strict=True):
        rows[r].append(ch)
    read_order = _read_positions(order)
    return "".join("".join(rows[r]) for r in read_order)


def _decode_letters(cipher: str, rails: int, offset: int, order: list[int]) -> str:
    n = len(cipher)
    pattern = _zigzag_pattern(n, rails, offset)
    lengths = [pattern.count(r) for r in range(max(rails, 1))]
    read_order = _read_positions(order)
    # Cut the ciphertext into strips following the read order, assign to rails.
    rows: list[str] = [""] * max(rails, 1)
    idx = 0
    for r in read_order:
        rows[r] = cipher[idx : idx + lengths[r]]
        idx += lengths[r]
    # Walk the zigzag, pulling the next letter from the appropriate rail.
    pos = [0] * max(rails, 1)
    out = []
    for r in pattern:
        out.append(rows[r][pos[r]])
        pos[r] += 1
    return "".join(out)


def _parse_key(key: str) -> tuple[int, int, list[int]]:
    """Parse ``"R:O:perm"`` (or a bare permutation) into (rails, offset, order)."""
    s = str(key).strip()
    parts = s.split(":")
    if len(parts) == 3:
        rails = int(parts[0])
        offset = int(parts[1])
        order = [int(x) for x in parts[2].replace(",", " ").split()]
    elif len(parts) == 1:
        order = [int(x) for x in parts[0].replace(",", " ").split()]
        rails = len(order)
        offset = 0
    else:
        raise ValueError("redefence key must be 'R:O:perm' or a bare permutation")
    if rails < 2:
        raise ValueError("redefence needs at least 2 rails")
    if len(order) != rails:
        raise ValueError(f"redefence read order must have {rails} entries")
    if sorted(order) != list(range(1, rails + 1)):
        raise ValueError(f"redefence read order must be a permutation of 1..{rails}")
    if not (0 <= offset <= 2 * rails - 3):
        raise ValueError(f"redefence offset must be in 0..{2 * rails - 3}")
    return rails, offset, order


class Redefence(Cipher):
    name = "redefence"
    aliases = ("redef",)
    description = "Rail fence with a permuted rail read-order and a phase offset."
    key_format = (
        "R:O:perm (rails:offset:1..R permutation), e.g. 3:0:3,1,2; bare perm assumes offset 0"
    )
    key_example = "3:0:3,1,2"
    complexity = 3

    # Transposition only reorders letters, so it cannot preserve word spacing;
    # encode/decode operate on a clean uppercase letter stream (no reflow, which
    # would leak the plaintext's word lengths into the ciphertext).
    def encode(self, text: str, key: str) -> str:
        rails, offset, order = _parse_key(key)
        return _encode_letters(only_letters(text), rails, offset, order)

    def decode(self, text: str, key: str) -> str:
        rails, offset, order = _parse_key(key)
        return _decode_letters(only_letters(text), rails, offset, order)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        min_rails = int(opts.get("min_rails", 3))
        max_rails = int(opts.get("max_rails", min(7, len(letters) - 1)))
        deadline = (time.monotonic() + timeout) if timeout else None

        candidates: list[Candidate] = []
        truncated = False
        for rails in range(min_rails, max_rails + 1):
            for offset in range(0, 2 * rails - 2):
                for perm in permutations(range(1, rails + 1)):
                    if deadline and time.monotonic() > deadline:
                        truncated = True
                        break
                    order = list(perm)
                    plain = _decode_letters(letters, rails, offset, order)
                    candidates.append(
                        Candidate(
                            plaintext=plain,
                            cipher=self.name,
                            key=f"{rails}:{offset}:{','.join(map(str, order))}",
                            score=scorer.score(plain),
                            confidence=scorer.confidence(plain),
                            meta={"rails": rails, "offset": offset},
                        )
                    )
                if truncated:
                    break
            if truncated:
                break
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated and out:
            out[-1].meta["timeout_truncated"] = True
        return out
