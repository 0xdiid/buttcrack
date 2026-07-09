"""Monome-Dinome cipher: a straddling substitution onto a digit stream.

The Monome-Dinome ("one-number / two-number") is a classic straddling
checkerboard with exactly TWO designated prefix digits. A 10-column board is
labelled ``0..9`` across the top. Eight high-frequency plaintext letters sit on
the top (un-prefixed) row in the eight columns whose digit is NOT one of the two
key digits; each is enciphered as that SINGLE column digit -- a *monome*. The two
key digits label two further rows that hold the remaining eighteen letters; each
of those is enciphered as a TWO-digit code -- a *dinome* -- formed by its row
prefix digit followed by its column digit. Because the eight monome digits never
begin a dinome (the prefix digits are reserved), the digit stream decodes
unambiguously left to right: a digit equal to a prefix consumes the next digit
too, otherwise it stands alone. ``8 + 2*9 = 26`` cells hold the whole alphabet,
so (unlike the VIC straddling board) there are no figure/stop cells and the
plaintext is letters only (J kept distinct, all 26 letters present).

KEY FORMAT
----------
``"PREFIXES/TOPWORD/FILLWORD"`` or any prefix thereof:

  * ``PREFIXES`` -- the two distinct prefix digits, e.g. ``"37"``. These are the
    columns left blank on the top row and used as the dinome row labels (in
    ascending digit order).
  * ``TOPWORD`` (optional) -- determines the eight monome letters placed on the
    top row. Its deduplicated letters (J kept) supply the high-frequency set; if
    fewer than eight distinct letters are given, the standard high-frequency
    letters ``ETAOINSR`` fill the rest. Default when omitted: ``"ETAOINSR"``.
  * ``FILLWORD`` (optional) -- a keyword used to build the mixed alphabet that
    fills the dinome rows: its deduplicated letters first, then the remaining
    A-Z, with every letter already used on the top row removed. Default when
    omitted: straight ``A-Z`` order (minus the top-row letters).

Examples::

    "37"                                  # prefixes 3,7; top ETAOINSR; A-Z fill
    "37/ETAOINSR"                         # explicit standard top row
    "26/SENORITA/CRYPTOGRAM"              # keyed top row and keyed fill

A bare two-digit key (``"37"``) uses the standard top row and A-Z fill.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: ACA-standard eight high-frequency English letters for the monome row.
DEFAULT_TOP = "ETAOINSR"


def _dedup(word: str) -> str:
    """Uppercased letters of ``word`` with duplicates removed, order kept."""
    seen: list[str] = []
    for ch in word.upper():
        if ch.isalpha() and ch not in seen:
            seen.append(ch)
    return "".join(seen)


def _parse_key(key: str) -> tuple[str, str, str]:
    """Return ``(prefixes, top_word, fill_word)`` from the key string."""
    parts = str(key).split("/")
    prefixes = parts[0].strip()
    if len(prefixes) != 2 or not prefixes.isdigit() or prefixes[0] == prefixes[1]:
        raise ValueError("monome-dinome prefixes must be two distinct digits, e.g. '37'")
    top_word = parts[1].strip() if len(parts) > 1 and parts[1].strip() else DEFAULT_TOP
    fill_word = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""
    return prefixes, top_word, fill_word


class _Board:
    """A built Monome-Dinome board: letter<->code lookups."""

    def __init__(self, key: str):
        prefixes, top_word, fill_word = _parse_key(key)
        # Prefix digits ascend so the two dinome rows have a deterministic order.
        self.prefixes = "".join(sorted(prefixes))

        # Top (monome) row: eight high-frequency letters, padded from the default
        # set, placed in the eight non-prefix columns in ascending column order.
        top_letters = _dedup(top_word)
        for ch in DEFAULT_TOP:
            if len(top_letters) >= 8:
                break
            if ch not in top_letters:
                top_letters += ch
        top_letters = top_letters[:8]

        top_cols = [str(d) for d in range(10) if str(d) not in self.prefixes]
        self.encode_map: dict[str, str] = {}
        self.decode_map: dict[str, str] = {}
        for col, ch in zip(top_cols, top_letters, strict=True):
            self.encode_map[ch] = col
            self.decode_map[col] = ch

        # Dinome rows: the remaining eighteen letters in a mixed alphabet built
        # from the (optional) fill keyword, then the rest of A-Z. The two rows
        # are filled row-major across ALL ten columns 0-9 (a dinome's second
        # digit may equal a prefix digit -- that is harmless, since decoding
        # keys off the FIRST digit only). Eighteen letters fill the first row
        # entirely (10) and eight cells of the second; the trailing two cells of
        # the second row are unused (the classic board parks a space/figure
        # there, but this letters-only cipher simply leaves them empty).
        remaining: list[str] = []
        for ch in _dedup(fill_word) + LETTERS:
            if ch in LETTERS and ch not in top_letters and ch not in remaining:
                remaining.append(ch)
        if len(remaining) != 18:
            raise ValueError(f"dinome rows need 18 letters, built {len(remaining)}")

        rows = self.prefixes  # already sorted ascending
        for ri, prefix in enumerate(rows):
            block = remaining[ri * 10 : ri * 10 + 10]
            for ci, ch in zip(range(10), block, strict=False):
                code = prefix + str(ci)
                self.encode_map[ch] = code
                self.decode_map[code] = ch


def _prepare(text: str) -> str:
    return only_letters(text)


class MonomeDinome(Cipher):
    """Monome-Dinome straddling substitution (letters -> digit stream).

    KEY FORMAT: ``"PREFIXES/TOPWORD/FILLWORD"`` (last two optional), e.g. ``"37"``,
    ``"37/ETAOINSR"`` or ``"26/SENORITA/CRYPTOGRAM"``. ``PREFIXES`` are the two
    distinct prefix (dinome row) digits; ``TOPWORD`` chooses the eight monome
    letters (padded from ``ETAOINSR``); ``FILLWORD`` keys the mixed alphabet that
    fills the two dinome rows (default straight A-Z). Operates on a clean
    uppercase letter stream; all 26 letters are kept distinct.
    """

    name = "monome-dinome"
    aliases = ("monome", "monomedinome")
    description = "Straddling monome/dinome substitution onto a digit stream (two prefix digits)."
    key_format = (
        "prefixes/topword/fillword (prefixes=2 distinct digits; topword and fillword optional)"
    )
    key_example = "26/SENORITA/CRYPTOGRAM"
    complexity = 4

    def encode(self, text: str, key: str) -> str:
        board = _Board(key)
        out: list[str] = []
        for ch in _prepare(text):
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
            if d in board.prefixes:
                ch = board.decode_map.get(digits[i : i + 2], "")
                i += 2
            else:
                ch = board.decode_map.get(d, "")
                i += 1
            out.append(ch)
        return "".join(out)

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
        """Best-effort keyless crack via simulated annealing over the board.

        The keyspace is C(10,2)=45 prefix-digit pairs times a 26-letter board
        permutation, so it is searched, not enumerated. For each candidate prefix
        pair we anneal: a random board is mutated by swapping two letters (and
        occasionally swapping which letters are monomes vs dinomes), keeping moves
        by the quadgram fitness of the decode. The best decode per prefix pair is
        returned, ranked.

        Recovery is NOT guaranteed: like every straddling/fractionating board, a
        single mis-placed cell mis-aligns the downstream parse, giving a deceptive
        fitness landscape. Returns ``[]`` for inputs too short to fingerprint.
        """
        digits = "".join(c for c in str(text) if c.isdigit())
        if len(digits) < 60:
            return []
        rng = rng or random.Random()
        deadline = None if timeout is None else time.monotonic() + timeout
        # Candidate prefix pairs: the two most-frequent leading digits tend to be
        # the prefixes (they head every dinome), so seed from frequency, then add
        # a few common conventional pairs.
        freq = sorted("0123456789", key=lambda d: -digits.count(d))
        seeds = {freq[0] + freq[1], "37", "26", "16", "29", "48"}
        prefix_pairs = ["".join(sorted(p)) for p in seeds if len(set(p)) == 2]

        restarts = int(opts.get("restarts", 2))
        temp0 = float(opts.get("temp", 8.0))
        step = float(opts.get("temp_step", 0.4))
        iters = int(opts.get("iters", 1200))

        def expired() -> bool:
            return deadline is not None and time.monotonic() > deadline

        def key_for(prefixes: str, board_letters: list[str]) -> str:
            # board_letters is the 26-letter ordering: first 8 = top row, then
            # the 18 dinome letters in row-major order. Encode it as TOP/FILL.
            top = "".join(board_letters[:8])
            fill = "".join(board_letters[8:])
            return f"{prefixes}/{top}/{fill}"

        def decode_with(prefixes: str, board_letters: list[str]) -> str:
            return self.decode(digits, key_for(prefixes, board_letters))

        candidates: list[Candidate] = []
        for prefixes in prefix_pairs:
            if expired():
                break
            best_letters = list(LETTERS)
            best_score = float("-inf")
            for _ in range(restarts):
                if expired():
                    break
                parent = list(LETTERS)
                rng.shuffle(parent)
                cur = scorer.score(decode_with(prefixes, parent))
                temp = temp0
                while temp > 0 and not expired():
                    for _ in range(iters):
                        i, j = rng.randrange(26), rng.randrange(26)
                        child = parent[:]
                        child[i], child[j] = child[j], child[i]
                        s = scorer.score(decode_with(prefixes, child))
                        delta = s - cur
                        if delta > 0 or rng.random() < math.exp(delta / temp):
                            parent, cur = child, s
                            if s > best_score:
                                best_letters, best_score = child[:], s
                    temp -= step
            plain = decode_with(prefixes, best_letters)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=key_for(prefixes, best_letters),
                    score=best_score,
                    confidence=scorer.confidence(plain),
                    meta={"prefixes": prefixes},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
