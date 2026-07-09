"""Numbered Key cipher (ACA, introduced by BION, The Cryptogram May-Jun 2010).

A homophonic substitution that maps letters to two-digit numbers.  An *extended
key* is built from a key phrase by keeping the phrase's letters **with repeats**
and then appending the missing letters of the alphabet in order.  The extended
key is rotated by an offset (the ACA sheet says "perhaps starting in the middle
of the key and wrapping around to the beginning"), then its positions are
numbered ``00, 01, 02, ...`` left to right.  Each plaintext letter is enciphered
as one of the numbers sitting under an occurrence of that letter, so a letter
that repeats in the key phrase has several possible numbers (its homophones),
while every number decodes to exactly one letter.

Worked example from the ACA cipher sheet (NumberedKey.pdf)::

    Key phrase : "I like ciphers."   offset 18
    Extended   : i l i k e c i p h e r s a b d f g j m n o q t u v w x y z
    Rotated    : m n o q t u v w x y z i l i k e c i p h e r s a b d f g j
    Numbered   : 00=m 01=n 02=o 03=q 04=t 05=u 06=v 07=w 08=x 09=y 10=z
                 11=i 12=l 13=i 14=k 15=e 16=c 17=i 18=p 19=h 20=e 21=r
                 22=s 23=a 24=b 25=d 26=f 27=g 28=j

    Pt: THE ROAD TO SUCCESS IS ALWAYS UNDER CONSTRUCTION
    Ct: 04 19 20 21 02 23 25 04 02 22 05 16 16 15 22 22 11 22 23 12 07 23
        09 22 05 01 25 20 21 16 02 01 22 04 21 05 16 04 17 02 01

KEY FORMAT
----------
``"PHRASE"`` or ``"PHRASE/OFFSET"``.  ``PHRASE`` may be any text (only its
letters matter, repeats are kept); ``OFFSET`` is an integer rotation of the
extended key (default ``0``).  Letters are case-insensitive; ``J`` is its own
letter (the 26-letter alphabet is used unchanged).

Encryption picks, by default, the first (lowest-numbered) homophone of each
letter so that ``encode`` is deterministic and round-trips; pass an ``rng`` to
spread letters over their homophones the way the ACA construction intends.
Decryption is unique.  Output is a space-separated string of two-digit numbers.
"""

from __future__ import annotations

import random
import string

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

_ALPHA = string.ascii_uppercase


def _parse_key(key: str) -> tuple[str, int]:
    """Split ``"PHRASE/OFFSET"`` into ``(phrase, offset)``; offset defaults to 0."""
    s = str(key)
    if "/" in s:
        phrase, off = s.rsplit("/", 1)
        off = off.strip()
        if off.lstrip("-").isdigit():
            return phrase, int(off)
        return phrase, 0
    return s, 0


def _extended_key(phrase: str) -> str:
    """Phrase letters (repeats kept) followed by the missing letters A->Z."""
    kept = [c for c in phrase.upper() if c.isalpha()]
    present = set(kept)
    missing = [c for c in _ALPHA if c not in present]
    return "".join(kept) + "".join(missing)


def _numbered(phrase: str, offset: int) -> str:
    """The rotated extended key whose index == the number under each letter."""
    ext = _extended_key(phrase)
    n = len(ext)
    off = offset % n if n else 0
    return ext[off:] + ext[:off]


def _homophones(numbered: str) -> dict[str, list[int]]:
    """Map each letter to the sorted list of numbers (positions) standing for it."""
    table: dict[str, list[int]] = {}
    for i, ch in enumerate(numbered):
        table.setdefault(ch, []).append(i)
    return table


class NumberedKey(Cipher):
    """ACA Numbered Key: homophonic letter->two-digit-number substitution."""

    name = "numbered-key"
    aliases = ("numberedkey",)
    description = "Homophonic letter->number cipher from a keyed, repeat-bearing alphabet (ACA)."
    key_format = "phrase or phrase/offset (offset = integer rotation, default 0)"
    key_example = "I LIKE CIPHERS/18"
    needs_key = True
    complexity = 5

    # -- encode ----------------------------------------------------------
    def encode(self, text: str, key: str, *, rng: random.Random | None = None) -> str:
        phrase, offset = _parse_key(key)
        numbered = _numbered(phrase, offset)
        homs = _homophones(numbered)
        out: list[str] = []
        for ch in only_letters(text):
            opts = homs.get(ch)
            if not opts:
                continue
            num = rng.choice(opts) if rng is not None else opts[0]
            out.append(f"{num:02d}")
        return " ".join(out)

    # -- decode ----------------------------------------------------------
    def decode(self, text: str, key: str) -> str:
        phrase, offset = _parse_key(key)
        numbered = _numbered(phrase, offset)
        nums = self._tokens(text)
        return "".join(numbered[n] for n in nums if 0 <= n < len(numbered))

    @staticmethod
    def _tokens(text: str) -> list[int]:
        """Read the two-digit number tokens out of a ciphertext string."""
        return [int(tok) for tok in str(text).split() if tok.isdigit()]

    # -- crack -----------------------------------------------------------
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
        """Keyless cracking is not attempted; returns ``[]``.

        Numbered Key is homophonic: several ciphertext numbers can stand for the
        same plaintext letter (the repeated key-phrase letters), so recovering
        the plaintext keyless means fitting a *many-to-one* number->letter map.
        Empirically the n-gram (quadgram) fitness landscape for this map is
        deceptive: simulated-annealing / hill-climbing searches converge to maps
        that score *higher* than the true map yet read as garble, even on very
        long messages (~1000 letters).  Because no purely keyless search we tried
        reliably recovers the plaintext, we return no candidates rather than emit
        a confidently-wrong reading.  CryptoCrack itself relies on crib (known
        plaintext) support to solve this cipher, which is outside this keyless
        ``crack`` contract.
        """
        _ = (text, scorer, rng, timeout, opts)
        return []
