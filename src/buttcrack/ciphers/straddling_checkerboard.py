"""Straddling Checkerboard: a fractionating substitution that emits digits.

A 10-column board encodes the eight most common plaintext letters as a SINGLE
digit (the column they sit in on the top, un-prefixed row) and every other
character as a TWO-digit code (a row-prefix digit then a column digit). Because
the short codes never collide with the start of a long code, the resulting
digit stream decodes unambiguously left to right -- letters "straddle" the
column boundary, hence the name. It is the substitution core of the VIC cipher.

LAYOUT
    The top row carries 8 letters in 8 of the 10 columns; the 2 remaining
    columns are left blank and their column digits become the prefixes for two
    further rows that hold the other 18 letters plus the figure-shift ``/`` and
    a full stop ``.`` (28 cells total = 26 letters + ``/`` + ``.``). The classic
    Wikipedia board (blank columns 2 and 6)::

            0 1 2 3 4 5 6 7 8 9
              E T   A O N   R I S
        2 |   B C D F G H J K L M
        6 |   P Q / U V W X Y Z .

    so E=1, T=2, A=4, ... B=20, C=21, ... P=60, Q=61, ``/``=62, etc. A bare
    digit in the plaintext is enciphered as the ``/`` code followed by the digit
    repeated three times (the ACA/VIC figure convention); decoding reverses it.

KEY FORMAT
    ``"BOARD/BLANKS"`` where ``BLANKS`` are the two (optionally three) blank
    column digits, e.g. ``"26"``. ``BOARD`` is either

      * a KEYWORD/phrase -- deduplicated, then the remaining A-Z appended, then
        ``/`` and ``.`` -- laid into the open cells in reading order, OR
      * an explicit 28-character sequence (a permutation of A-Z plus ``/`` and
        ``.``) read into the open cells in order, for reproducing a published
        board exactly.

    Examples::

        "ETAONRISBCDFGHJKLMPQ/UVWXYZ./26"   # the Wikipedia board, explicit
        "KEYWORD/26"                         # keyword-built board

    A bare key with no ``/`` is treated as a keyword with default blanks ``26``.
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXTRAS = "/."  # figure-shift and full stop fill the last two cells
FULL = LETTERS + EXTRAS  # 28 cells
DEFAULT_BLANKS = "26"


def _parse_key(key: str) -> tuple[str, str]:
    """Return (board_spec, blanks) from ``"BOARD/BLANKS"``.

    The blank field is the trailing slash-separated token consisting only of
    distinct digits; everything before it is the board spec (which may itself
    contain a ``/`` cell when given as an explicit 28-char sequence).
    """
    s = str(key).strip()
    if "/" not in s:
        return s, DEFAULT_BLANKS
    head, _, tail = s.rpartition("/")
    if tail and all(c.isdigit() for c in tail) and len(set(tail)) == len(tail):
        return head, tail
    # No trailing digit field -> whole thing is the board, default blanks.
    return s, DEFAULT_BLANKS


def _board_sequence(spec: str) -> str:
    """28-cell ordering from a keyword or an explicit 28-char permutation."""
    spec_up = spec.upper()
    explicit = [c for c in spec_up if c in FULL]
    if len(explicit) == 28 and len(set(explicit)) == 28:
        return "".join(explicit)
    # Keyword build: dedup keyword letters, then remaining A-Z, then "/.".
    seq: list[str] = []
    for ch in spec_up + LETTERS:
        if ch in LETTERS and ch not in seq:
            seq.append(ch)
    seq.extend(EXTRAS)
    if len(seq) != 28:
        raise ValueError(f"checkerboard needs 28 cells, built {len(seq)}")
    return "".join(seq)


class _Board:
    """A built straddling checkerboard: char<->code lookups."""

    def __init__(self, key: str):
        spec, blanks = _parse_key(key)
        if len(blanks) < 2 or len(set(blanks)) != len(blanks):
            raise ValueError("blanks must be >=2 distinct digit columns")
        self.blanks = blanks
        seq = _board_sequence(spec)

        open_cols = [str(d) for d in range(10) if str(d) not in blanks]
        n_top = len(open_cols)  # 8 for two blanks, 7 for three
        self.encode_map: dict[str, str] = {}
        self.decode_map: dict[str, str] = {}

        cells = list(seq)
        # Top (un-prefixed) row: the open columns, single-digit codes.
        for col, ch in zip(open_cols, cells[:n_top], strict=True):
            self.encode_map[ch] = col
            self.decode_map[col] = ch
        # One full 10-column row per blank, prefixed by the blank's digit.
        rest = cells[n_top:]
        for prefix in blanks:
            row, rest = rest[:10], rest[10:]
            for ci, ch in enumerate(row):
                code = prefix + str(ci)
                self.encode_map[ch] = code
                self.decode_map[code] = ch
        if rest:
            raise ValueError("board spec has too many cells for this blank count")
        self.figure = self.encode_map.get("/")


def _prepare(text: str) -> str:
    """Keep A-Z (uppercased) and digits; drop everything else."""
    return "".join(c for c in text.upper() if c.isalpha() or c.isdigit())


class StraddlingCheckerboard(Cipher):
    name = "straddling-checkerboard"
    aliases = ("straddling", "checkerboard-straddling")
    description = "Fractionating substitution to a digit stream (VIC cipher core)."
    key_format = (
        "board/blanks (board=keyword or 28-char A-Z/. sequence; blanks=2 distinct digit columns)"
    )
    key_example = "KEYWORD/26"
    complexity = 4

    def encode(self, text: str, key: str) -> str:
        board = _Board(key)
        out: list[str] = []
        for ch in _prepare(text):
            if ch.isdigit():
                # Figure shift, then the digit thrice (ACA/VIC convention).
                if board.figure is None:
                    raise ValueError("board has no '/' cell for digit encoding")
                out.append(board.figure + ch * 3)
            else:
                code = board.encode_map.get(ch)
                if code is None:
                    raise ValueError(f"no board cell for {ch!r}")
                out.append(code)
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        board = _Board(key)
        digits = "".join(c for c in str(text) if c.isdigit())
        out: list[str] = []
        i = 0
        n = len(digits)
        while i < n:
            d = digits[i]
            if d in board.blanks:
                code = digits[i : i + 2]
                ch = board.decode_map.get(code, "")
                i += 2
            else:
                ch = board.decode_map.get(d, "")
                i += 1
            if ch == "/":
                # Figure shift: next three identical digits are one plaintext digit.
                out.append(digits[i] if i < n else "")
                i += 3
            else:
                out.append(ch)
        return "".join(out)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Best-effort keyless crack via random-restart hill climbing.

        The board has C(10,2)=45 blank-column choices and a 28-cell permutation,
        so exhaustive search is intractable. For each plausible blank-column pair
        we run repeated random restarts, each climbing by swapping two board
        cells and keeping improvements to the quadgram fitness of the decode. We
        return the best decodes found before ``timeout``.

        Full recovery is NOT guaranteed -- the fractionated digit stream gives a
        weak, deceptive fitness landscape with many local optima.
        """
        import random

        digits = "".join(c for c in str(text) if c.isdigit())
        if len(digits) < 8:
            return []
        rng = rng or random.Random(0)
        deadline = None if timeout is None else time.monotonic() + timeout
        blank_pairs = ["26", "37", "16", "29", "48", "05"]

        def expired() -> bool:
            return deadline is not None and time.monotonic() > deadline

        def climb(blanks: str) -> tuple[float, list[str]]:
            seq = list(FULL)
            rng.shuffle(seq)
            key = "".join(seq) + "/" + blanks
            best_score = scorer.score(self.decode(digits, key))
            stale = 0
            while stale < 600 and not expired():
                i, j = rng.randrange(28), rng.randrange(28)
                if i == j:
                    continue
                seq[i], seq[j] = seq[j], seq[i]
                sc = scorer.score(self.decode(digits, "".join(seq) + "/" + blanks))
                if sc > best_score:
                    best_score, stale = sc, 0
                else:
                    seq[i], seq[j] = seq[j], seq[i]  # revert
                    stale += 1
            return best_score, seq

        best_per_blank: dict[str, tuple[float, list[str]]] = {}
        while not expired():
            for blanks in blank_pairs:
                if expired():
                    break
                try:
                    score, seq = climb(blanks)
                except Exception:
                    continue
                if blanks not in best_per_blank or score > best_per_blank[blanks][0]:
                    best_per_blank[blanks] = (score, seq)

        results: list[Candidate] = []
        for blanks, (_, seq) in best_per_blank.items():
            key = "".join(seq) + "/" + blanks
            plain = self.decode(digits, key)
            results.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=key,
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"blanks": blanks},
                )
            )
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top]
