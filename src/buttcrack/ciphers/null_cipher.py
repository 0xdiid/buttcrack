"""Null cipher (ACA): a concealment cipher / simple steganography.

The hidden message is carried by one significant letter per word of a cover
text; every other letter is a *null* (filler). The key names which letter of
each word is significant:

* ``"first"`` / ``"1"``           -- the first letter of each word
* ``"last"`` / ``"0"`` / ``"-1"`` -- the last letter of each word
* ``"middle"``                    -- the middle letter, index ``(len-1)//2``
* a positive integer ``N``        -- the Nth letter (1-based) from the start
* a negative integer ``-N``       -- the Nth letter from the end

Decoding extracts the keyed letter from each whitespace-separated word of the
cover and concatenates them -- this is the canonical ACA direction (the cover
is given and the message is recovered). Encoding deterministically synthesises
a cover whose keyed slot in word *i* holds plaintext letter *i*, so that
``decode(encode(msg, key), key) == msg``.

Worked example (ACA cipher sheet "Null"):
    CT: THE GREAT OLD PUMPERS.   key: middle   ->   pt: HELP
because the middle letters of THE/GREAT/OLD/PUMPERS are H/E/L/P.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

# Vowels avoided as filler so a synthesised cover does not read as words and so
# the significant slot stands out; any consonant filler round-trips identically.
_FILLER = "XZQK"


def _resolve_position(key: str, length: int) -> int:
    """Return the 0-based index into a word of ``length`` letters for ``key``.

    Negative results index from the end (Python-style). Out-of-range indices are
    clamped into the word so every word yields exactly one significant letter.
    """
    k = key.strip().lower()
    if k in ("first", "start", "begin"):
        idx = 0
    elif k in ("last", "end"):
        idx = -1
    elif k in ("middle", "mid", "centre", "center"):
        idx = (length - 1) // 2
    else:
        try:
            n = int(k)
        except ValueError as exc:
            raise ValueError(f"unrecognised null-cipher key: {key!r}") from exc
        if n > 0:
            idx = n - 1  # 1-based from the start
        elif n == 0:
            idx = -1  # treat 0 as "last", matching CryptoCrack's last-letter null
        else:
            idx = n  # negative: count from the end
    # Clamp into [-length, length-1] so short words still contribute a letter.
    if idx >= length:
        idx = length - 1
    elif idx < -length:
        idx = 0
    return idx % length


def _words(text: str) -> list[str]:
    """Letter-only words, in order, from a cover text."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text.upper():
        if "A" <= ch <= "Z":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


class NullCipher(Cipher):
    """Concealment cipher: one significant letter per word, the rest are nulls."""

    name = "null"
    aliases = ("nullcipher", "concealment")
    description = "Null/concealment cipher: a keyed letter position per word hides the message."
    key_format = "position: first|last|middle, or +/-N (Nth letter per word)"
    key_example = "first"
    complexity = 2

    def encode(self, text: str, key: str) -> str:
        """Synthesise a cover whose keyed slot in each word carries one letter.

        Each plaintext letter becomes a word of length ``slot+1`` (or ``2`` for
        first/last/middle so the slot is unambiguous) padded with consonant
        filler, with the plaintext letter dropped into the keyed position.
        """
        letters = [c for c in text.upper() if "A" <= c <= "Z"]
        words: list[str] = []
        for i, ch in enumerate(letters):
            # A 3-letter word makes first/middle/last all distinct and lets any
            # small numeric key land inside the word.
            word_len = 3
            k = key.strip().lower()
            if k.lstrip("-").isdigit():
                n = abs(int(k))
                if n > word_len:
                    word_len = n
            idx = _resolve_position(key, word_len)
            slots = [_FILLER[(i + j) % len(_FILLER)] for j in range(word_len)]
            slots[idx] = ch
            words.append("".join(slots))
        return " ".join(words)

    def decode(self, text: str, key: str) -> str:
        """Extract the keyed significant letter from each word of the cover."""
        words = _words(text)
        return "".join(w[_resolve_position(key, len(w))] for w in words)

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
        """Try each plausible significant-letter position and rank by fitness.

        Best-effort: with no key the search space is just the position rule, so
        we extract under first/last/middle plus a few small numeric positions
        and score each recovered message.
        """
        words = _words(text)
        if len(words) < 4:
            return []
        deadline = (time.monotonic() + timeout) if timeout else None

        max_len = max(len(w) for w in words)
        keys: list[str] = ["first", "last", "middle"]
        keys += [str(n) for n in range(2, min(max_len, 6) + 1)]
        keys += [str(-n) for n in range(2, min(max_len, 6) + 1)]

        candidates: list[Candidate] = []
        seen: set[str] = set()
        for key in keys:
            if deadline and time.monotonic() > deadline:
                break
            plain = self.decode(text, key)
            if not plain or plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=key,
                    score=scorer.score(plain),
                    confidence=scorer.confidence(plain),
                    meta={"position": key},
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
