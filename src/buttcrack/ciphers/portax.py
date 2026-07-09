"""Portax cipher: ACA periodic DIGRAPHIC polyalphabetic derived from Porta.

Portax enciphers VERTICAL pairs of letters using a two-strip "slide". A
13-column window shows four rows; the per-column key letter sets the slide.

    row1 (fixed)   = A B C D E F G H I J K L M
    row2 (sliding) = N O P Q R S T U V W X Z ... (N..Z), shifted by ``s``
    row3 (A2 odd)  = A C E G I K M O Q S U W Y, shifted by ``s``
    row4 (A2 even) = B D F H J L N P R T V X Z, shifted by ``s``

The slide offset ``s`` for a key letter is its index in the combined A2
sequence ``ACEGIKMOQSUWYBDFHJLNPRTVXZ``.

LAYOUT
------
The plaintext stream is written horizontally into a block of ``period`` columns
(``period == len(key)``). Rows are paired ``(0,1),(2,3),(4,5),...``; within a
pair the upper row supplies the TOP letter and the lower row the BOTTOM letter
of each vertical pair. Each text column ``c`` uses the slide for ``key[c]``.

ENCIPHER (per vertical pair, given the column's slide ``s``)
-----------------------------------------------------------
The TOP plaintext letter is located in row1 or row2; the BOTTOM letter in row3
or row4. They are diagonally opposite corners of a rectangle and the ciphertext
is the OTHER two corners, the top-row corner first. If both lie in the same
window column, the substitutes are the OTHER two letters of that column
(top-half letter first). Decryption applies the identical rectangle in reverse.

The ciphertext is read off the block row by row (top row of a pair, then its
bottom row), so the digraphic output is interleaved with the layout.

KEY FORMAT
----------
A single keyword of letters; the period equals the keyword length. The
plaintext is padded with ``X`` to a whole number of row-pairs (an even number of
full ``period``-wide rows) so every vertical pair is complete.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

# Combined A2 letter sequence; a key letter's index here is its slide offset.
_SEQ = "ACEGIKMOQSUWYBDFHJLNPRTVXZ"
_ROW1 = "ABCDEFGHIJKLM"
_ROW2_BASE = "NOPQRSTUVWXYZ"
_A2_ODD = "ACEGIKMOQSUWY"
_A2_EVEN = "BDFHJLNPRTVXZ"


def _slide(s: int) -> list[str]:
    """Build the four 13-column rows for slide offset ``s``."""
    r2 = "".join(_ROW2_BASE[(i + s) % 13] for i in range(13))
    r3 = "".join(_A2_ODD[(i + s) % 13] for i in range(13))
    r4 = "".join(_A2_EVEN[(i + s) % 13] for i in range(13))
    return [_ROW1, r2, r3, r4]


def _key_offset(letter: str) -> int:
    return _SEQ.index(letter)


def _find_top(rows: list[str], ch: str) -> tuple[int, int]:
    if ch in rows[0]:
        return 0, rows[0].index(ch)
    return 1, rows[1].index(ch)


def _find_bot(rows: list[str], ch: str) -> tuple[int, int]:
    if ch in rows[2]:
        return 2, rows[2].index(ch)
    return 3, rows[3].index(ch)


def _enc_pair(top: str, bot: str, s: int) -> str:
    rows = _slide(s)
    tr, tc = _find_top(rows, top)
    br, bc = _find_bot(rows, bot)
    if tc == bc:
        col = [rows[r][tc] for r in range(4)]
        others = [col[r] for r in range(4) if r not in (tr, br)]
        return others[0] + others[1]
    return rows[tr][bc] + rows[br][tc]


def _dec_pair(ctop: str, cbot: str, s: int) -> str:
    rows = _slide(s)
    ctr, ctc = _find_top(rows, ctop)
    cbr, cbc = _find_bot(rows, cbot)
    if ctc == cbc:
        col = [rows[r][ctc] for r in range(4)]
        others = [col[r] for r in range(4) if r not in (ctr, cbr)]
        return others[0] + others[1]
    return rows[ctr][cbc] + rows[cbr][ctc]


def _pad_to_block(letters: str, period: int) -> str:
    """Pad with X to a whole number of row-pairs (even count of full rows)."""
    block = 2 * period
    if len(letters) % block:
        letters = letters + "X" * (block - len(letters) % block)
    return letters


def _transform(letters: str, offsets: Sequence[int], encrypt: bool) -> str:
    """Run Portax over a clean uppercase ``letters`` block.

    The text is already padded to an even number of ``period``-wide rows.
    Pairs of rows ``(0,1),(2,3),...`` give vertical digraphs; output is read off
    row by row, so the result has the same length and row layout as the input.
    """
    period = len(offsets)
    rows = [letters[i : i + period] for i in range(0, len(letters), period)]
    out_rows = [["" for _ in range(period)] for _ in rows]
    op = _enc_pair if encrypt else _dec_pair
    for rt in range(0, len(rows), 2):
        rb = rt + 1
        for col in range(period):
            res = op(rows[rt][col], rows[rb][col], offsets[col])
            out_rows[rt][col] = res[0]
            out_rows[rb][col] = res[1]
    return "".join("".join(r) for r in out_rows)


class Portax(Cipher):
    name = "portax"
    description = "ACA periodic digraphic polyalphabetic (Porta-derived vertical pairs)."
    key_format = "keyword (letters); period = keyword length"
    key_example = "EASY"
    complexity = 6

    def _offsets(self, key: str) -> list[int]:
        offsets = [_key_offset(c) for c in only_letters(key)]
        if not offsets:
            raise ValueError("portax key must contain letters")
        return offsets

    def encode(self, text: str, key: str) -> str:
        offsets = self._offsets(key)
        letters = _pad_to_block(only_letters(text), len(offsets))
        return _transform(letters, offsets, encrypt=True)

    def decode(self, text: str, key: str) -> str:
        offsets = self._offsets(key)
        letters = _pad_to_block(only_letters(text), len(offsets))
        return _transform(letters, offsets, encrypt=False)

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
        """Best-effort keyless recovery: detect period, then per-column slides.

        For a fixed period the digraphic block has independent columns, each
        governed by one of 13 distinct slides (offsets ``s`` and ``s+13`` give
        the same window). We hill-climb the per-column offsets against the
        full-block decrypt score. The true period is unknown, so we try every
        candidate period and keep the best-scoring decrypts. This is heuristic:
        keyed-alphabet/digraphic ciphers are hard keyless and recovery is not
        guaranteed; callers should treat results as ranked hypotheses.
        """
        letters = only_letters(text)
        if len(letters) < 8:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None
        forced = opts.get("key_length")
        max_len = int(opts.get("max_key_length", min(12, len(letters) // 4)))
        periods = [int(forced)] if forced else range(1, max(2, max_len) + 1)

        # 13 distinct slides; map a recovered offset back to a canonical key letter.
        def offset_to_letter(s: int) -> str:
            return _SEQ[s % 26]

        by_plain: dict[str, tuple[float, list[int], int]] = {}
        for period in periods:
            if period < 1:
                continue
            if deadline and time.monotonic() > deadline:
                break
            padded = _pad_to_block(letters, period)
            offsets = [0] * period
            best = scorer.score(_transform(padded, offsets, encrypt=False))
            improved = True
            while improved:
                improved = False
                for col in range(period):
                    if deadline and time.monotonic() > deadline:
                        break
                    cur = offsets[col]
                    for s in range(13):
                        if s == cur:
                            continue
                        offsets[col] = s
                        sc = scorer.score(_transform(padded, offsets, encrypt=False))
                        if sc > best:
                            best, cur, improved = sc, s, True
                    offsets[col] = cur
            plain = _transform(padded, offsets, encrypt=False)
            prev = by_plain.get(plain)
            if prev is None or period < prev[2]:
                by_plain[plain] = (best, offsets[:], period)

        candidates = [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key="".join(offset_to_letter(s) for s in offsets),
                score=score,
                confidence=scorer.confidence(plain),
                meta={"key_length": period},
            )
            for plain, (score, offsets, period) in by_plain.items()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
