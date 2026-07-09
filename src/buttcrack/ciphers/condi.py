"""Condi cipher ("CONsecutive DIgraphs") — a self-keying polyalphabetic.

A keyed 26-letter alphabet plus an initial numeric offset. Encryption shifts
each plaintext letter forward (within the keyed alphabet) by the *current*
offset, then replaces the offset with the position of the plaintext letter just
enciphered. The offset therefore chains off the plaintext, so the cipher is not
periodic and ``encode != decode``.

Positions are 1-based (1..26), matching the ACA / bionspot worked examples; the
running offset values are those same 1-based positions.

KEY format
----------
``"<keyword> <offset>"`` — a keyword (used to build the keyed alphabet) and an
initial offset in 1..25, separated by whitespace, ``/`` or ``:``. Examples::

    "CRYPTOGRAM 10"
    "CRYPTOGRAM/10"
    "CRYPTOGRAM:10"

The keyword may be empty (``" 10"`` / ``"/10"``) to use the straight A-Z
alphabet. The full 26-letter alphabet is used (J is kept separate).
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def keyed_alphabet(keyword: str) -> str:
    """Build a 26-letter keyed alphabet: keyword letters first, then the rest."""
    seq: list[str] = []
    for ch in only_letters(keyword) + _ALPHABET:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _parse_key(key: str) -> tuple[str, int]:
    """Return ``(keyed_alphabet, initial_offset)`` from a ``"keyword offset"`` key."""
    raw = str(key).strip()
    # Split on the last separator so the keyword may itself contain spaces.
    sep_idx = -1
    for i, ch in enumerate(raw):
        if ch in " \t/:":
            sep_idx = i
    if sep_idx == -1:
        raise ValueError("condi key must be '<keyword> <offset>' (e.g. 'CRYPTOGRAM 10')")
    keyword = raw[:sep_idx]
    offset_str = raw[sep_idx + 1 :].strip()
    try:
        offset = int(offset_str)
    except ValueError as exc:
        raise ValueError(f"condi offset must be an integer 1..25, got {offset_str!r}") from exc
    # Normalize into 1..26 (callers conventionally use 1..25).
    offset = ((offset - 1) % 26) + 1
    return keyed_alphabet(keyword), offset


def _encode_letters(letters: str, alphabet: str, offset: int) -> str:
    pos1 = {ch: i + 1 for i, ch in enumerate(alphabet)}
    out: list[str] = []
    for p in letters:
        p1 = pos1[p]
        c1 = ((p1 + offset - 1) % 26) + 1
        out.append(alphabet[c1 - 1])
        offset = p1
    return "".join(out)


def _decode_letters(letters: str, alphabet: str, offset: int) -> str:
    pos1 = {ch: i + 1 for i, ch in enumerate(alphabet)}
    out: list[str] = []
    for c in letters:
        c1 = pos1[c]
        p1 = ((c1 - offset - 1) % 26) + 1
        out.append(alphabet[p1 - 1])
        offset = p1
    return "".join(out)


class Condi(Cipher):
    name = "condi"
    aliases = ("consecutive-digraphs",)
    description = "Self-keying polyalphabetic; offset chains off the previous plaintext letter."
    key_format = "keyword and initial offset 1..25, separated by space, '/', or ':'"
    key_example = "CRYPTOGRAM 10"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        alphabet, offset = _parse_key(key)
        return reflow(text, _encode_letters(only_letters(text), alphabet, offset))

    def decode(self, text: str, key: str) -> str:
        alphabet, offset = _parse_key(key)
        return reflow(text, _decode_letters(only_letters(text), alphabet, offset))

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
        """Hill-climb the keyed alphabet (and initial offset) against quadgrams.

        The offset chains deterministically off the plaintext, so a correct
        alphabet propagates cleanly. We do swap-based hill-climbing with random
        restarts, trying every initial offset 1..25 for each alphabet. Best
        effort: short ciphertext or a tight timeout may yield no usable result.
        """
        letters = only_letters(text)
        if len(letters) < 20:
            return []
        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 12))
        deadline = (time.monotonic() + timeout) if timeout else None

        def best_offset(alphabet: str) -> tuple[float, int, str]:
            best_s = float("-inf")
            best_off = 1
            best_plain = ""
            for off in range(1, 26):
                plain = _decode_letters(letters, alphabet, off)
                s = scorer.score(plain)
                if s > best_s:
                    best_s, best_off, best_plain = s, off, plain
            return best_s, best_off, best_plain

        results: list[tuple[float, str, int, str]] = []
        for r in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = list(_ALPHABET)
            if r != 0:
                rng.shuffle(parent)
            parent_alpha = "".join(parent)
            parent_score, parent_off, parent_plain = best_offset(parent_alpha)
            improved = True
            timed_out = False
            while improved and not timed_out:
                improved = False
                for i in range(25):
                    if deadline and time.monotonic() > deadline:
                        timed_out = True
                        break
                    for j in range(i + 1, 26):
                        child = parent[:]
                        child[i], child[j] = child[j], child[i]
                        child_alpha = "".join(child)
                        s, off, plain = best_offset(child_alpha)
                        if s > parent_score:
                            parent_score = s
                            parent = child
                            parent_alpha = child_alpha
                            parent_off = off
                            parent_plain = plain
                            improved = True
            results.append((parent_score, parent_alpha, parent_off, parent_plain))

        results.sort(key=lambda rs: rs[0], reverse=True)
        seen: set[str] = set()
        candidates: list[Candidate] = []
        for score, alphabet, off, plain in results:
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=f"{alphabet} {off}",
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"offset": off, "restarts": restarts},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
