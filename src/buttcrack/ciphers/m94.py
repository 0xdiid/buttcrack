"""M-94 / M-138 — the Jefferson cylinder: a bank of scrambled alphabet disks.

The nearest relative of the strip machinery already in this package, and the one
rotor-family member whose classical attack is genuinely tractable rather than a brute
force. The US Army fielded it from 1922 to 1943.

ALGORITHM
---------
Twenty-five aluminium disks sit on a spindle; each carries the 26 letters in its own
scrambled order and is identified by the letter following ``A``. The *order of the
disks on the spindle* is the key. To encipher, the operator turns each disk until the
25 plaintext letters read across one row (the "generatrix"), then copies off any other
row. The offset between the two rows is the second half of the key.

So for disk ``d`` holding alphabet ``A_d``, plaintext letter ``p`` at position ``j``::

    ciphertext = A_d[(A_d.index(p) + offset) % 26]

Messages longer than the disk count wrap: position ``j`` and position ``j + 25`` go
through the same disk, which is precisely what makes the cipher solvable.

M-138 is the same machine built from sliding strips instead of disks — a subset of a
larger strip set, in a chosen order. Give a shorter disk order to model it.

KEY FORMAT
----------
``ORDER/OFFSET`` — comma-separated 1-based disk numbers and the row offset 1..25,
e.g. ``17,4,9,22,1/6``. Positions cycle through the listed disks, so any number of
disks from 2 to 25 works.

Disk alphabets: the standard 25-disk M-94 set (disk 17 famously begins
``ARMYOFTHEUS``). Sourced from the CrypTool 2 ``CylinderCipher`` tables.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from importlib import resources

from ..assignment import hungarian_max
from ..result import Candidate
from ..scoring import ENGLISH_MONOGRAM_FREQ, NgramScorer
from ..text import only_letters
from .base import Cipher


@lru_cache(maxsize=1)
def m94_disks() -> tuple[str, ...]:
    """The 25 standard M-94 disk alphabets, in disk-number order."""
    raw = resources.files("buttcrack.data").joinpath("m94_disks.txt").read_text()
    disks = tuple(line.strip() for line in raw.split() if line.strip())
    if len(disks) != 25 or any(len(set(d)) != 26 for d in disks):
        raise ValueError("m94_disks.txt must hold 25 alphabets of 26 distinct letters")
    return disks


def _parse_key(key: str) -> tuple[list[int], int]:
    order_part, sep, offset_part = str(key).rpartition("/")
    if not sep or not offset_part.strip().lstrip("-").isdigit():
        raise ValueError("m94 key must be 'ORDER/OFFSET', e.g. '17,4,9,22,1/6'")
    order = [int(x) - 1 for x in order_part.replace(",", " ").split()]
    if not order or any(not 0 <= d < 25 for d in order):
        raise ValueError("m94 disk numbers must be 1..25")
    if len(set(order)) != len(order):
        raise ValueError("m94 disk order must not repeat a disk")
    return order, int(offset_part) % 26


def _run(letters: str, order: list[int], offset: int) -> str:
    disks = m94_disks()
    out = []
    for j, ch in enumerate(letters):
        alphabet = disks[order[j % len(order)]]
        out.append(alphabet[(alphabet.index(ch) + offset) % 26])
    return "".join(out)


class M94(Cipher):
    name = "m94"
    aliases = ("cylinder", "jefferson", "m138", "strip-cipher")
    description = "Jefferson cylinder: scrambled alphabet disks in a keyed order, read off-row."
    key_format = "disk order and row offset, 'ORDER/OFFSET' (1-based disk numbers)"
    key_example = "17,4,9,22,1/6"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        order, offset = _parse_key(key)
        return _run(only_letters(text).upper(), order, offset)

    def decode(self, text: str, key: str) -> str:
        order, offset = _parse_key(key)
        return _run(only_letters(text).upper(), order, -offset % 26)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Solve the disk order as an assignment problem, then refine on n-grams.

        The order is a permutation of 25 disks — 15.5 septillion arrangements, so no
        brute force. But because position ``j`` and position ``j + width`` share a disk,
        the ciphertext splits into ``width`` independent columns, and for a fixed offset
        each column decrypts under each disk in exactly one way. That makes "which disk
        sits in which position" a maximum-weight bipartite assignment, solved exactly by
        Hungarian rather than searched.

        Monogram fit is all a single column can supply, and a column only holds
        ``n / width`` letters, so the assignment alone is unreliable on short messages.
        A pairwise-swap climb on the quadgram score of the *whole* reconstructed
        plaintext follows it — that is the step that uses cross-column context, which no
        per-column score can see.

        ``width`` defaults to 25 (a full M-94); pass it for an M-138 strip subset.
        """
        letters = only_letters(text).upper()
        widths = opts.get("widths") or [int(opts.get("width", 25))]
        if len(letters) < 2 * max(widths):
            return []
        deadline = (time.monotonic() + timeout) if timeout else None
        disks = m94_disks()
        offsets = opts.get("offsets") or range(1, 26)

        candidates: list[Candidate] = []
        for width in widths:
            if width < 2 or width > 25:
                continue
            columns = [letters[c::width] for c in range(width)]
            for offset in offsets:
                if deadline and time.monotonic() > deadline:
                    break
                back = -offset % 26
                # decoded[col][disk] = that column's plaintext under that disk.
                decoded = [
                    [_decode_column(col, disks[d], back) for d in range(25)] for col in columns
                ]
                weights = [[_monogram_fit(decoded[c][d]) for d in range(25)] for c in range(width)]
                _, assignment = hungarian_max(weights)
                assignment = _climb(assignment, decoded, letters, width, scorer, deadline)
                plain = _reassemble(decoded, assignment, len(letters), width)
                order = ",".join(str(d + 1) for d in assignment)
                candidates.append(
                    Candidate(
                        plaintext=plain,
                        cipher=self.name,
                        key=f"{order}/{offset}",
                        score=scorer.score(plain),
                        confidence=scorer.confidence(plain),
                        meta={
                            "width": width,
                            "offset": offset,
                            "disk_order": [d + 1 for d in assignment],
                        },
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]


def _decode_column(column: str, alphabet: str, back: int) -> str:
    return "".join(alphabet[(alphabet.index(ch) + back) % 26] for ch in column)


_LOG_FREQ = {
    ch: math.log(max(ENGLISH_MONOGRAM_FREQ[ch], 1e-6)) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
}


def _monogram_fit(letters: str) -> float:
    """Log-likelihood of ``letters`` under English monograms — the per-cell weight."""
    return sum(_LOG_FREQ[ch] for ch in letters)


def _reassemble(decoded: list[list[str]], assignment: list[int], n: int, width: int) -> str:
    cols = [decoded[c][assignment[c]] for c in range(width)]
    return "".join(cols[i % width][i // width] for i in range(n))


def _climb(
    assignment: list[int],
    decoded: list[list[str]],
    letters: str,
    width: int,
    scorer: NgramScorer,
    deadline: float | None,
) -> list[int]:
    """Swap pairs of column->disk assignments while the quadgram score improves.

    Swapping keeps the assignment a permutation (no disk used twice), so the climb
    never leaves the feasible set the Hungarian seed started in.
    """
    best = list(assignment)
    best_score = scorer.score(_reassemble(decoded, best, len(letters), width))
    improved = True
    while improved:
        improved = False
        for i in range(width):
            if deadline and time.monotonic() > deadline:
                return best
            for j in range(i + 1, width):
                trial = list(best)
                trial[i], trial[j] = trial[j], trial[i]
                score = scorer.score(_reassemble(decoded, trial, len(letters), width))
                if score > best_score:
                    best, best_score, improved = trial, score, True
    return best
