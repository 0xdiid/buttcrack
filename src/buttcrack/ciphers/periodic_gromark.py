"""Periodic Gromark cipher.

The Periodic Gromark is the periodic variant of the Gromark (GRonsfeld + Mixed
Alphabet + Running Key).  It differs from the plain Gromark in three ways, all
driven by a single keyword:

* the **period** ``P`` equals the number of distinct letters in the keyword;
* the **primer** is the alphabetical-rank numbering of those distinct letters
  (e.g. ``WRIGHT`` -> distinct ``WRIGHT``, ranks ``G=1 H=2 I=3 R=4 T=5 W=6`` ->
  primer ``6 4 3 1 2 5``; ``TEACHER`` -> distinct ``TEACHR`` -> ``6 3 1 2 4 5``);
* the ciphertext is processed in **blocks of size P**, and every block gets an
  extra fixed shift so the "same keyed-alphabet shift structure" repeats with
  period ``P`` blocks.

KEY format
----------
A single keyword, e.g. ``"WRIGHT"``.  Its length (after de-duplicating repeated
letters) is the period ``P``; its alphabetical letter-order gives the ``P``-digit
primer; and it also builds the mixed (cipher) alphabet.

Construction
------------
* **Mixed alphabet** -- a K2-type ("sequence") columnar transposition: form the
  keyed alphabet (de-duped keyword followed by the rest of A-Z), write it
  row-wise into a block whose width equals the de-duped keyword length, number
  the columns by the alphabetical order of those keyword letters, then read the
  block off column by column in that numeric order.
* **Running key** -- chain addition (lagged-Fibonacci) from the ``P``-digit
  primer: ``digit[n] = (digit[n-P] + digit[n-P+1]) mod 10``, appended until the
  key is as long as the text.
* **Block offset** -- for block index ``b`` the extra shift is the position in
  the mixed alphabet of the keyword's ``b``-th (de-duped) letter; the keyword
  cycles every ``P`` blocks: ``off[b] = MIXED.index(keyword[b % P])``.

ENCRYPT
-------
For plaintext letter ``p`` at position ``i`` (block ``b = (i // P) % P``)::

    c = MIXED[(STRAIGHT.index(p) + running_key[i] + off[b]) mod 26]

DECRYPT
-------
``p = STRAIGHT[(MIXED.index(c) - running_key[i] - off[b]) mod 26]``.

Encrypt and decrypt are NOT reciprocal.
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import ALPHABET, only_letters, reflow
from .base import Cipher


def _dedup_keyword(keyword: str) -> str:
    """De-duplicated, uppercased, letters-only keyword (sets the period)."""
    seq: list[str] = []
    for ch in keyword.upper():
        if "A" <= ch <= "Z" and ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _keyed_alphabet(keyword: str) -> str:
    """De-duped keyword letters followed by the remaining A-Z letters."""
    seq = list(_dedup_keyword(keyword))
    for ch in ALPHABET:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _column_order(keyword: str) -> list[int]:
    """Rank each column by the alphabetical order of the keyword letters."""
    kw = keyword
    ranked_indices = sorted(range(len(kw)), key=lambda i: (kw[i], i))
    order = [0] * len(kw)
    for rank, idx in enumerate(ranked_indices):
        order[idx] = rank
    return order


def _mixed_alphabet(keyword: str) -> str:
    """Build the 26-letter mixed alphabet via K2-type columnar transposition."""
    kw = _dedup_keyword(keyword)
    if not kw:
        raise ValueError("periodic-gromark keyword must contain letters")
    width = len(kw)
    keyed = _keyed_alphabet(kw)
    rows = [keyed[i : i + width] for i in range(0, len(keyed), width)]
    order = _column_order(kw)
    cols_by_rank = sorted(range(width), key=lambda c: order[c])
    out: list[str] = []
    for col in cols_by_rank:
        for row in rows:
            if col < len(row):
                out.append(row[col])
    return "".join(out)


def _primer_from_keyword(keyword: str) -> list[int]:
    """Alphabetical-rank numbering (1-based) of the de-duped keyword letters."""
    kw = _dedup_keyword(keyword)
    ranked = sorted(range(len(kw)), key=lambda i: kw[i])
    rank = [0] * len(kw)
    for r, idx in enumerate(ranked):
        rank[idx] = r + 1
    return rank


def _running_key(primer: list[int], length: int) -> list[int]:
    """Chain-addition keystream: digit n = (digit[n-lag] + digit[n-lag+1]) % 10."""
    lag = len(primer)
    digits = list(primer)
    while len(digits) < length:
        nxt = (digits[len(digits) - lag] + digits[len(digits) - lag + 1]) % 10
        digits.append(nxt)
    return digits[:length]


def _components(keyword: str) -> tuple[str, list[int], list[int]]:
    """Return ``(mixed, block_offsets, primer)`` for a keyword."""
    kw = _dedup_keyword(keyword)
    if not kw:
        raise ValueError("periodic-gromark key must include a keyword")
    mixed = _mixed_alphabet(kw)
    pos = {ch: i for i, ch in enumerate(mixed)}
    block_offsets = [pos[ch] for ch in kw]
    primer = _primer_from_keyword(kw)
    return mixed, block_offsets, primer


def _encode_letters(letters: str, keyword: str) -> str:
    mixed, block_offsets, primer = _components(keyword)
    period = len(block_offsets)
    rk = _running_key(primer, len(letters))
    out = []
    for i, ch in enumerate(letters):
        block = (i // period) % period
        idx = (ord(ch) - 65 + rk[i] + block_offsets[block]) % 26
        out.append(mixed[idx])
    return "".join(out)


def _decode_letters(letters: str, keyword: str) -> str:
    mixed, block_offsets, primer = _components(keyword)
    period = len(block_offsets)
    pos = {ch: i for i, ch in enumerate(mixed)}
    rk = _running_key(primer, len(letters))
    out = []
    for i, ch in enumerate(letters):
        block = (i // period) % period
        idx = (pos[ch] - rk[i] - block_offsets[block]) % 26
        out.append(ALPHABET[idx])
    return "".join(out)


class PeriodicGromark(Cipher):
    name = "periodic-gromark"
    aliases = ("pgromark", "periodicgromark")
    description = "Periodic Gromark: block-periodic running key over a K2-mixed alphabet."
    key_format = "single keyword (its distinct-letter count sets the period and primer)"
    key_example = "WRIGHT"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        return _encode_letters(only_letters(text), key)

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(only_letters(text), key)

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
        """Best-effort crack against a supplied keyword (or word list).

        The fully keyless Periodic Gromark couples a keyword-derived mixed
        alphabet, a keyword-derived primer, and the keyword-derived block
        offsets -- all three flow from one unknown keyword, so the only sound
        attack here is a dictionary/word-list search.  If the caller supplies a
        ``keyword`` (or ``keywords`` list) via ``opts`` we decrypt with each and
        rank by quadgram fitness; with no hint we return ``[]`` rather than
        pretend to solve an intractable instance.
        """
        letters = only_letters(text)
        if len(letters) < 8:
            return []

        keywords: list[str] = []
        if opts.get("keyword"):
            keywords.append(str(opts["keyword"]))
        keywords.extend(str(k) for k in opts.get("keywords", []))
        keywords = [k for k in keywords if only_letters(k)]
        if not keywords:
            return []

        deadline = (time.monotonic() + timeout) if timeout else None
        results: list[tuple[float, str, str]] = []  # (score, plaintext, keyword)
        for keyword in keywords:
            if deadline and time.monotonic() > deadline:
                break
            try:
                plain = _decode_letters(letters, keyword)
            except (ValueError, KeyError):
                continue
            results.append((scorer.score(plain), plain, keyword.upper()))

        results.sort(key=lambda r: r[0], reverse=True)
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for score, plain, keyword in results:
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=keyword,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"keyword": keyword},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
