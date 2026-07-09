"""Bazeries cipher (Etienne Bazeries, ~1890s): a combined transposition and
keyed substitution driven by a single secret number.

The key is one integer below 1,000,000. That number does three jobs:

1. **Right square** — the number is *spelled out in words* (e.g. ``7352`` ->
   ``SEVENTHOUSANDTHREEHUNDREDANDFIFTYTWO``); repeated letters are dropped
   (``SEVNTHOUADRFIYW``) and the remaining alphabet letters are appended in
   order (``SEVNTHOUADRFIYWBCGJKLMPQXZ``). With J merged into I this 25-letter
   mixed alphabet is laid into a 5x5 grid **by rows**.
2. **Left square** — the straight alphabet (J->I) laid into a 5x5 grid **by
   columns** (read down: ``A F L Q V`` is the first column, so row 0 is
   ``A F L Q V``).
3. **Transposition cycle** — the digits of the number give a repeating cycle of
   group lengths (``7352`` -> 7, 3, 5, 2, 7, 3, 5, 2, ...).

ENCRYPT: split the plaintext into successive groups whose lengths follow the
digit cycle and reverse each group; then substitute every letter by finding it
in the LEFT square and reading the letter at the SAME (row, column) in the
RIGHT square.

DECRYPT: substitute RIGHT->LEFT, then re-group by the same digit cycle and
reverse each group again. Encrypt and decrypt are NOT identical.

KEY FORMAT
----------
A single integer ``1 <= key <= 999999`` (digits ``0`` are ignored as group
lengths, so a useful key has no zero digits). Example: ``7352``.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

ALPHABET_5 = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 25 letters, no J

_ONES = (
    "",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
)
_TENS = ("", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY")


def _spell_two(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + _ONES[ones]


def _spell_three(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    out = ""
    if hundreds:
        out += _ONES[hundreds] + "HUNDRED"
    if rest:
        if hundreds:
            out += "AND"
        out += _spell_two(rest)
    return out


def spell_number(n: int) -> str:
    """Spell a 1..999_999 integer in (British-style) words, letters only."""
    if n <= 0:
        return "ZERO"
    thousands, rest = divmod(n, 1000)
    out = ""
    if thousands:
        out += _spell_three(thousands) + "THOUSAND"
    if rest:
        if thousands and rest < 100:
            out += "AND"
        out += _spell_three(rest)
    return out


def _merge(ch: str) -> str:
    return "I" if ch == "J" else ch


def build_right_alphabet(n: int) -> str:
    """The 25-letter keyed alphabet from the spelled-out number (J->I)."""
    seq: list[str] = []
    for ch in spell_number(n):
        m = _merge(ch)
        if m not in seq:
            seq.append(m)
    for ch in ALPHABET_5:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _build_squares(n: int) -> tuple[dict[str, int], list[str]]:
    """Return (left_pos: letter -> flat index, right_grid: flat list).

    The LEFT square is the straight alphabet filled by COLUMNS; the RIGHT square
    is the keyed alphabet filled by ROWS. Substitution maps a left-square cell to
    the right-square cell at the same (row, col), so we only need the left
    letter->index map and the right index->letter list.
    """
    left_grid = [""] * 25
    idx = 0
    for col in range(5):
        for row in range(5):
            left_grid[row * 5 + col] = ALPHABET_5[idx]
            idx += 1
    left_pos = {ch: i for i, ch in enumerate(left_grid)}
    right_grid = list(build_right_alphabet(n))
    return left_pos, right_grid


def _digit_cycle(n: int) -> list[int]:
    """Group lengths from the digits of ``n`` (drop any 0 digit)."""
    digits = [int(d) for d in str(n) if d != "0"]
    if not digits:
        raise ValueError("bazeries key needs at least one nonzero digit")
    return digits


def _transpose(letters: str, cycle: list[int]) -> str:
    """Split into groups whose lengths follow ``cycle`` and reverse each group."""
    out: list[str] = []
    i = 0
    di = 0
    n = len(letters)
    while i < n:
        glen = cycle[di % len(cycle)]
        out.append(letters[i : i + glen][::-1])
        i += glen
        di += 1
    return "".join(out)


def _parse_key(key: str) -> int:
    s = str(key).strip()
    if not s.isdigit():
        raise ValueError("bazeries key must be a positive integer, e.g. 7352")
    n = int(s)
    if not 1 <= n <= 999_999:
        raise ValueError("bazeries key must be in 1..999999")
    return n


def _encode_letters(letters: str, n: int) -> str:
    left_pos, right_grid = _build_squares(n)
    transposed = _transpose(letters, _digit_cycle(n))
    return "".join(right_grid[left_pos[_merge(ch)]] for ch in transposed)


def _decode_letters(letters: str, n: int) -> str:
    left_pos, right_grid = _build_squares(n)
    # invert the right square: letter -> flat index, then look up left grid.
    right_pos = {ch: i for i, ch in enumerate(right_grid)}
    left_grid = [""] * 25
    for ch, i in left_pos.items():
        left_grid[i] = ch
    substituted = "".join(left_grid[right_pos[_merge(ch)]] for ch in letters)
    # reversing each group is its own inverse, so re-apply the same cycle.
    return _transpose(substituted, _digit_cycle(n))


class Bazeries(Cipher):
    """Bazeries cipher: digit-cycle group reversal + keyed 5x5 substitution.

    KEY FORMAT: a single integer ``1..999999`` (e.g. ``7352``). The number is
    spelled out to build the keyed RIGHT 5x5 square (J->I, filled by rows); the
    straight alphabet fills the LEFT square by columns; and the number's digits
    give the repeating cycle of group lengths reversed during transposition.
    Encode/decode operate on the letters-only uppercase stream (J merged into I).
    """

    name = "bazeries"
    description = "Bazeries: digit-cycle group reversal plus keyed 5x5 substitution; integer key."
    key_format = "single integer 1..999999 (digits should be nonzero)"
    key_example = "7352"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        n = _parse_key(key)
        letters = only_letters(text).replace("J", "I")
        return _encode_letters(letters, n)

    def decode(self, text: str, key: str) -> str:
        n = _parse_key(key)
        letters = only_letters(text).replace("J", "I")
        return _decode_letters(letters, n)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Brute-force the key number against quadgram fitness.

        The key fully determines both the keyed alphabet (via its spelling) and
        the transposition cycle (via its digits), so the entire keyspace is just
        the integers ``1..max_key``. We scan that range (default capped to a
        budget that fits the timeout), decrypt each, and keep the best-scoring
        decrypts. Best-effort: with no timeout the default cap is 200000.
        """
        letters = only_letters(text).replace("J", "I")
        if len(letters) < 8:
            return []
        _ = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None
        max_key = int(opts.get("max_key", 200_000))
        max_key = min(max_key, 999_999)

        results: list[tuple[float, int, str]] = []
        for n in range(1, max_key + 1):
            if deadline and (n & 0x3FF) == 0 and time.monotonic() > deadline:
                break
            if "0" in str(n):  # zero digits give no group length; skip
                continue
            try:
                plain = _decode_letters(letters, n)
            except (ValueError, KeyError):
                continue
            results.append((scorer.score(plain), n, plain))

        results.sort(key=lambda r: r[0], reverse=True)
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for score, n, plain in results:
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=str(n),
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"key": n},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
