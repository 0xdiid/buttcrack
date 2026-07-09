"""Phillips cipher: eight related 5x5 squares with a diagonal substitution.

The Phillips cipher (ACA / CryptoCrack) is a polyalphabetic substitution built
from a SINGLE keyed 5x5 Polybius square (J merged into I, filled row-by-row).
From that base square eight working squares are derived by sinking individual
rows; the plaintext is processed in groups of five letters, each group using the
next working square in sequence, so the overall period is 40 (8 squares x 5).

SQUARE GENERATION (Row variant)
-------------------------------
Number the base square's rows ``r0 r1 r2 r3 r4``. The eight squares used, by row
order, are::

    #1  r0 r1 r2 r3 r4   (the base square)
    #2  r1 r0 r2 r3 r4   (row 1 sunk one position; rows above shift up)
    #3  r1 r2 r0 r3 r4
    #4  r1 r2 r3 r0 r4
    #5  r0 r1 r2 r3 r4   (== #1; row 1 has cycled back to the top)
    #6  r0 r2 r1 r3 r4   (now row 2 sinks)
    #7  r0 r2 r3 r1 r4
    #8  r0 r2 r3 r4 r1

Squares #1-#4 are exactly the squares used in the published CryptoCrack
worked example (keyword PATIENCE); #5 coincides with #1 as the row cycle wraps.

SUBSTITUTION
------------
For each plaintext letter, find it in the current working square and take the
letter DIAGONALLY DOWN-AND-RIGHT (row + 1, col + 1) with wraparound: the bottom
row wraps to the top, the right column wraps to the left. Decryption takes the
letter diagonally UP-AND-LEFT (row - 1, col - 1) with the same wraparound.
Encryption and decryption are therefore NOT the same operation.

KEY FORMAT
----------
A single keyword that builds the base 5x5 square (duplicates dropped, remaining
letters of A-Z appended, J->I, filled row-by-row). Example: ``PATIENCE`` ->
keyed alphabet ``PATIENCBDFGHKLMOQRSUVWXYZ``. The period is fixed at 40.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher
from .squares import ALPHABET_5, PolybiusSquare

#: number of working squares (group cycle length is GROUP_SIZE * N_SQUARES = 40)
N_SQUARES = 8
GROUP_SIZE = 5


def _rows(grid: list[str]) -> list[list[str]]:
    return [grid[i * 5 : (i + 1) * 5] for i in range(5)]


def _flat(rows: list[list[str]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        out.extend(row)
    return out


def _working_squares(base: list[str]) -> list[list[str]]:
    """Derive the eight working squares from the base square's rows."""
    rb = _rows(base)
    out: list[list[str]] = []
    # Phase A: row r0 sinks through positions 0..3 (rows r1,r2,r3 shift up; r4 fixed).
    rest_a = [rb[1], rb[2], rb[3]]
    for k in range(4):
        out.append(_flat(rest_a[:k] + [rb[0]] + rest_a[k:] + [rb[4]]))
    # Square #5 coincides with #1 (the row cycle has wrapped back to the top).
    out.append(list(base))
    # Phase B: row r1 sinks through positions 2..4 (r0 fixed at top).
    rest_b = [rb[2], rb[3], rb[4]]
    for k in range(1, 4):
        out.append(_flat([rb[0]] + rest_b[:k] + [rb[1]] + rest_b[k:]))
    return out


def _down_right(grid: list[str], pos: dict[str, int], ch: str) -> str:
    idx = pos[ch]
    r, c = divmod(idx, 5)
    return grid[((r + 1) % 5) * 5 + (c + 1) % 5]


def _up_left(grid: list[str], pos: dict[str, int], ch: str) -> str:
    idx = pos[ch]
    r, c = divmod(idx, 5)
    return grid[((r - 1) % 5) * 5 + (c - 1) % 5]


def _build_base(key: str) -> list[str]:
    """Base 5x5 keyed square as a flat 25-letter list (J->I, row-by-row)."""
    return list(PolybiusSquare(key).grid)


def _transform(letters: str, base: list[str], *, decrypt: bool) -> str:
    squares = _working_squares(base)
    positions = [{ch: i for i, ch in enumerate(sq)} for sq in squares]
    step = _up_left if decrypt else _down_right
    out: list[str] = []
    for i, ch in enumerate(letters):
        s = (i // GROUP_SIZE) % N_SQUARES
        out.append(step(squares[s], positions[s], ch))
    return "".join(out)


def _prepare(text: str) -> str:
    """Letters only, uppercased, J merged into I (the 5x5 square alphabet)."""
    return only_letters(text).replace("J", "I")


class Phillips(Cipher):
    """Phillips cipher: eight related 5x5 squares, diagonal down-right substitution.

    KEY FORMAT: a single keyword building the base 5x5 square (duplicates dropped,
    remaining A-Z appended, J->I, filled row-by-row), e.g. ``PATIENCE``. Plaintext
    is processed in groups of five letters; each group uses the next of eight
    squares derived by row-sinking, giving a fixed period of 40. Each letter is
    replaced by the letter diagonally down-and-right (wrapping); decryption takes
    the letter diagonally up-and-left. Encrypt and decrypt differ.
    """

    name = "phillips"
    description = "Eight related 5x5 squares; diagonal down-right substitution (period 40)."
    key_format = "keyword (letters; builds the base 5x5 square, J->I; period fixed at 40)"
    key_example = "PATIENCE"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        return _transform(_prepare(text), _build_base(key), decrypt=False)

    def decode(self, text: str, key: str) -> str:
        return _transform(_prepare(text), _build_base(key), decrypt=True)

    def crack(
        self,
        text: str,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng: random.Random | None = None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Best-effort: simulated annealing on the single 25-letter base square.

        All eight working squares derive from ONE base square, so we hill-climb /
        anneal directly on the 25-letter base, decrypting and scoring with quadgram
        fitness. The period is fixed at 40, so no period search is needed. Returns
        the best decrypt; ``[]`` for inputs too short to fingerprint.
        """
        letters = _prepare(text)
        if len(letters) < 40:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        restarts = int(opts.get("restarts", 4))
        temp0 = float(opts.get("temp", 10.0))
        step = float(opts.get("temp_step", 0.4))
        iters = int(opts.get("iters", 2000))
        base_alphabet = list(ALPHABET_5)

        def decrypt_with(square: list[str]) -> str:
            return _transform(letters, square, decrypt=True)

        best_sq = base_alphabet[:]
        best_score = float("-inf")
        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = base_alphabet[:]
            rng.shuffle(parent)
            cur = scorer.score(decrypt_with(parent))
            temp = temp0
            while temp > 0:
                if deadline and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    i, j = rng.randrange(25), rng.randrange(25)
                    child = parent[:]
                    child[i], child[j] = child[j], child[i]
                    s = scorer.score(decrypt_with(child))
                    delta = s - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        parent, cur = child, s
                        if s > best_score:
                            best_sq, best_score = child[:], s
                temp -= step

        sq_str = "".join(best_sq)
        plain = decrypt_with(best_sq)
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=sq_str,
                score=best_score,
                confidence=scorer.confidence(plain),
                meta={"square": sq_str, "period": GROUP_SIZE * N_SQUARES},
            )
        ]
