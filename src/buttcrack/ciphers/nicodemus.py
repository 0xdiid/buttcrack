"""Nicodemus cipher: keyed columnar transposition combined with a per-column
Vigenere, taken off the block five rows at a time (ACA standard).

Devised by Harold Berkley and described in *The Cryptogram* (Aug-Sep 1949), the
Nicodemus combines three steps under a single keyword:

  1. **Columnar transposition.** Write the plaintext row by row beneath the
     keyword, then reorder the columns into the keyword's alphabetical rank
     order (ties broken left-to-right by position), exactly as in a columnar
     transposition. No null padding is used, so a short final row simply leaves
     the rightmost columns one letter shorter (an *incomplete* rectangle).
  2. **Vigenere encipherment with the same key.** Each (now alphabetically
     ordered) column is enciphered with a Vigenere shift; the shift for the
     k-th column is the k-th letter of the *sorted* keyword. So with keyword
     ``CAT`` the columns are headed ``A C T`` and shifted by 0, 2, 19.
  3. **Take-off, five rows at a time.** The enciphered block is read off in
     consecutive horizontal bands of five rows; within each band the columns
     are read in order (left to right). The final band may be shorter than five
     rows, in which case its remaining letters are taken off column by column.

KEY FORMAT
----------
Normally a single keyword, e.g. ``MONARCH``. The keyword's letters supply both
the column rank order (the transposition) and — once sorted — the per-column
Vigenere shifts. The take-off band height defaults to 5 (the ACA convention);
to override it append ``/N``, e.g. ``MONARCH/5``.

``crack`` recovers an arbitrary (column order, per-column shift) pair that need
not correspond to any real keyword, so it reports the explicit form
``=READORDER|SHIFTS`` (e.g. ``=3,5,6,0,2,1,4|A,C,H,M,N,O,R``) which ``decode``
also accepts: ``READORDER`` is the comma-separated physical column read order
and ``SHIFTS`` the per-output-column shift letters.

Worked vector (ACA): pt ``THE EARLY BIRD GETS THE WORM`` key ``CAT`` ->
``HAYRE VGNKI XKUWM TWMUG TAH``.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from itertools import permutations

from ..result import Candidate
from ..scoring import ENGLISH_MONOGRAM_FREQ, NgramScorer
from ..text import only_letters
from .base import Cipher

_DEFAULT_BLOCK = 5
_MONO = [ENGLISH_MONOGRAM_FREQ[chr(65 + i)] for i in range(26)]


def _parse_key(key: str) -> tuple[list[int], list[int], int]:
    """Return (read_order, shifts, block).

    ``read_order[j]`` is the physical column read j-th; ``shifts[j]`` is the
    Vigenere shift (0-25) for output column j. Accepts a keyword (optionally
    ``KEYWORD/BLOCK``) or the explicit ``=READORDER|SHIFTS`` crack form.
    """
    s = str(key).strip()
    if s.startswith("="):
        order_part, _, shift_part = s[1:].partition("|")
        if not shift_part:
            raise ValueError("nicodemus explicit key must be '=READORDER|SHIFTS'")
        order = [int(x) for x in order_part.replace(",", " ").split()]
        shifts: list[int] = []
        for tok in shift_part.replace(",", " ").split():
            tok = tok.strip().upper()
            shifts.append((ord(tok) - 65) % 26 if tok.isalpha() else int(tok) % 26)
        if sorted(order) != list(range(len(order))) or len(shifts) != len(order):
            raise ValueError("nicodemus explicit key order must be a permutation matching SHIFTS")
        return order, shifts, _DEFAULT_BLOCK

    block = _DEFAULT_BLOCK
    if "/" in s:
        kw, _, blk = s.rpartition("/")
        blk = blk.strip()
        if not blk.isdigit() or int(blk) < 1:
            raise ValueError("nicodemus block size must be a positive integer (key 'KEYWORD/N')")
        s, block = kw, int(blk)
    letters = only_letters(s)
    if not letters:
        raise ValueError("nicodemus key must be a keyword (optionally 'KEYWORD/N')")
    # Column read order = alphabetical rank of the keyword letters (ties L->R).
    order = [idx for _, idx in sorted((ch, i) for i, ch in enumerate(letters))]
    # The k-th output column is shifted by the k-th letter of the sorted keyword.
    shifts = [ord(ch) - 65 for ch in sorted(letters)]
    return order, shifts, block


def _column_heights(n: int, width: int) -> list[int]:
    """Physical column heights for an incomplete (unpadded) ``n``-letter grid."""
    if n == 0:
        return [0] * width
    rows = (n + width - 1) // width
    r = n % width
    if r == 0:
        return [rows] * width
    return [rows if c < r else rows - 1 for c in range(width)]


def _encode_letters(letters: str, order: list[int], shifts: list[int], block: int) -> str:
    width = len(order)
    n = len(letters)
    rows = (n + width - 1) // width if n else 0
    # Physical columns, filled row by row across the rectangle.
    cols: list[list[str]] = [[] for _ in range(width)]
    for i, ch in enumerate(letters):
        cols[i % width].append(ch)
    # Output column j = physical column order[j], Vigenere-shifted by shifts[j].
    out_cols: list[str] = []
    for j in range(width):
        shift = shifts[j]
        src = cols[order[j]]
        out_cols.append("".join(chr((ord(ch) - 65 + shift) % 26 + 65) for ch in src))
    # Take off in horizontal bands of `block` rows, columns left-to-right.
    out: list[str] = []
    for r0 in range(0, rows, block):
        for col in out_cols:
            out.append(col[r0 : r0 + block])
    return "".join(out)


def _decode_letters(cipher: str, order: list[int], shifts: list[int], block: int) -> str:
    width = len(order)
    n = len(cipher)
    if n == 0:
        return ""
    heights = _column_heights(n, width)
    rows = (n + width - 1) // width
    out_heights = [heights[order[j]] for j in range(width)]
    # Rebuild each enciphered output column from the band-interleaved ciphertext.
    out_cols: list[str] = [""] * width
    idx = 0
    for r0 in range(0, rows, block):
        band = min(block, rows - r0)
        for j in range(width):
            take = min(out_heights[j], r0 + band) - r0 if out_heights[j] > r0 else 0
            out_cols[j] += cipher[idx : idx + take]
            idx += take
    # Reverse the Vigenere, mapping each output column back to its physical column.
    plain_cols: list[str] = [""] * width
    for j in range(width):
        shift = shifts[j]
        plain_cols[order[j]] = "".join(chr((ord(ch) - 65 - shift) % 26 + 65) for ch in out_cols[j])
    # Read the rectangle back row by row across the physical columns.
    pos = [0] * width
    out: list[str] = []
    while len(out) < n:
        for c in range(width):
            if pos[c] < len(plain_cols[c]):
                out.append(plain_cols[c][pos[c]])
                pos[c] += 1
    return "".join(out)


def _best_shift(col: str) -> int:
    """Chi-squared best Caesar shift for a single column against English."""
    cnt = len(col)
    best_s, best_chi = 0, math.inf
    for s in range(26):
        counts = Counter((ord(ch) - 65 - s) % 26 for ch in col)
        chi = 0.0
        for i in range(26):
            exp = _MONO[i] * cnt / 100.0
            if exp > 0:
                diff = counts.get(i, 0) - exp
                chi += diff * diff / exp
        if chi < best_chi:
            best_chi, best_s = chi, s
    return best_s


class Nicodemus(Cipher):
    """Nicodemus: keyed columnar transposition + per-column Vigenere, taken off
    five rows at a time (ACA standard).

    KEY FORMAT: a keyword such as ``MONARCH`` (optionally ``KEYWORD/N`` to set the
    take-off band height, default 5). The keyword supplies both the column rank
    order and — once its letters are sorted — the per-column Vigenere shifts.
    ``crack`` may instead report the explicit ``=READORDER|SHIFTS`` form, which
    ``decode`` also accepts.
    """

    name = "nicodemus"
    aliases = ("nicodemous",)
    description = "Keyed columnar transposition + per-column Vigenere (ACA), key is a keyword."
    key_format = "keyword (letters), optional /N take-off band height, e.g. MONARCH or MONARCH/5"
    key_example = "MONARCH"
    complexity = 6

    # Transposition cannot preserve word spacing; operate on a clean uppercase
    # letter stream (no reflow, which would leak plaintext word lengths).
    def encode(self, text: str, key: str) -> str:
        order, shifts, block = _parse_key(key)
        return _encode_letters(only_letters(text), order, shifts, block)

    def decode(self, text: str, key: str) -> str:
        order, shifts, block = _parse_key(key)
        return _decode_letters(only_letters(text), order, shifts, block)

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Best-effort keyless recovery by simulated annealing per period.

        For each candidate width the column read order and the per-column
        Vigenere shifts are searched jointly: swaps reorder the columns, while
        shift mutations slide a column's Caesar key, both scored by quadgram
        fitness of the decrypt. Single-column frequency analysis is too weak on
        the short interleaved columns to solve the shifts in isolation, so the
        joint annealing (seeded from a chi-squared shift guess) is what makes the
        recovery tractable. Returns the best decrypt per width, ranked; ``[]``
        for inputs too short to fingerprint.
        """
        letters = only_letters(text)
        if len(letters) < 60:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        max_width = int(opts.get("max_width", 8))
        widths = [int(opts["width"])] if opts.get("width") else range(2, max_width + 1)
        restarts = int(opts.get("restarts", 4))
        iters = int(opts.get("iters", 2500))
        temp0 = float(opts.get("temp", 8.0))
        cooling = float(opts.get("cooling", 0.85))

        n = len(letters)
        block = _DEFAULT_BLOCK

        def seed_shifts(width: int, order: list[int]) -> list[int]:
            heights = _column_heights(n, width)
            out_heights = [heights[order[j]] for j in range(width)]
            rows = (n + width - 1) // width
            out_cols: list[str] = [""] * width
            idx = 0
            for r0 in range(0, rows, block):
                band = min(block, rows - r0)
                for j in range(width):
                    take = min(out_heights[j], r0 + band) - r0 if out_heights[j] > r0 else 0
                    out_cols[j] += letters[idx : idx + take]
                    idx += take
            return [_best_shift(col) for col in out_cols]

        candidates: list[Candidate] = []
        for width in widths:
            if width < 2 or width > n:
                continue
            if deadline and time.monotonic() > deadline:
                break
            best_score = float("-inf")
            best_order: list[int] = list(range(width))
            best_shifts: list[int] = [0] * width
            for restart in range(restarts):
                if deadline and time.monotonic() > deadline:
                    break
                order = list(range(width))
                rng.shuffle(order)
                # Seed shifts from a per-column chi-squared guess on the first
                # restart, random thereafter, to diversify the search.
                shifts = (
                    seed_shifts(width, order)
                    if restart == 0
                    else [rng.randrange(26) for _ in range(width)]
                )
                cur = scorer.score(_decode_letters(letters, order, shifts, block))
                temp = temp0
                while temp > 0.3:
                    if deadline and time.monotonic() > deadline:
                        break
                    for _ in range(iters):
                        new_order = order[:]
                        new_shifts = shifts[:]
                        if rng.random() < 0.5:
                            i, j = rng.randrange(width), rng.randrange(width)
                            new_order[i], new_order[j] = new_order[j], new_order[i]
                        else:
                            k = rng.randrange(width)
                            new_shifts[k] = (new_shifts[k] + rng.randrange(1, 26)) % 26
                        score = scorer.score(_decode_letters(letters, new_order, new_shifts, block))
                        delta = score - cur
                        if delta > 0 or rng.random() < math.exp(delta / temp):
                            order, shifts, cur = new_order, new_shifts, score
                            if score > best_score:
                                best_score = score
                                best_order = order[:]
                                best_shifts = shifts[:]
                    temp *= cooling

            plain = _decode_letters(letters, best_order, best_shifts, block)
            key_str = "={}|{}".format(
                ",".join(map(str, best_order)),
                ",".join(chr(65 + s) for s in best_shifts),
            )
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=key_str,
                    score=best_score,
                    confidence=scorer.confidence(plain),
                    meta={"width": width},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]


def _exhaustive_small(
    letters: str, scorer: NgramScorer, width: int, block: int
) -> tuple[float, list[int], list[int]]:
    """Brute-force every column order for a small width, chi-squared shifts.

    Unused by ``crack`` (the annealer subsumes it) but kept as a clear reference
    for the column-permutation search on tiny periods.
    """
    n = len(letters)
    heights = _column_heights(n, width)
    rows = (n + width - 1) // width
    best = (float("-inf"), list(range(width)), [0] * width)
    for perm in permutations(range(width)):
        order = list(perm)
        out_heights = [heights[order[j]] for j in range(width)]
        out_cols: list[str] = [""] * width
        idx = 0
        for r0 in range(0, rows, block):
            band = min(block, rows - r0)
            for col_idx in range(width):
                take = min(out_heights[col_idx], r0 + band) - r0 if out_heights[col_idx] > r0 else 0
                out_cols[col_idx] += letters[idx : idx + take]
                idx += take
        shifts = [_best_shift(col) for col in out_cols]
        plain = _decode_letters(letters, order, shifts, block)
        score = scorer.score(plain)
        if score > best[0]:
            best = (score, order, shifts)
    return best
