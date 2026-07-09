"""Fractionated Morse cipher (ACA), invented by FIDDLE in *The Cryptogram* 1960.

Plaintext is converted to an International Morse stream (``.``/``-`` with ``x`` for
the inter-letter gap and ``xx`` for the inter-word gap, no separator at the message
boundary), padded with trailing ``x`` to a multiple of three, then split into
trigrams over ``{".", "-", "x"}``. The 26 trigrams (all 27 minus the unused
``xxx``) are labelled by a keyed 26-letter alphabet; each trigram becomes its
label letter. Decryption reverses the labelling, reassembles the Morse stream,
and translates it back to text.

KEY FORMAT
    A single keyword/keyphrase. Repeated letters are dropped and the remaining
    A-Z letters are appended in order to form the 26-letter mixed alphabet that
    labels the trigram columns (e.g. ``CROWDED`` -> ``CROWDEABFGHIJKLMNPQSTUVXYZ``).
    An empty key (or one that reduces to nothing) gives the straight alphabet
    ``A-Z`` (the unkeyed/identity table).

The fixed column order enumerates trigrams of ``{".", "-", "x"}`` in base-3
counting with digit order ``. < - < x`` (so column 1 = ``...``, column 2 = ``..-``,
..., column 26 = ``xx-``; the 27th trigram ``xxx`` is left unlabelled).
"""

from __future__ import annotations

import string
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher
from .morse import morse_to_text, text_to_morse

# The 27 trigrams over {".", "-", "x"} in base-3 order (. < - < x). Only the first
# 26 are labelled; "xxx" (the 27th) is unused.
_SYMBOLS = (".", "-", "x")
_TRIGRAMS = [a + b + c for a in _SYMBOLS for b in _SYMBOLS for c in _SYMBOLS]
_LABELLED_TRIGRAMS = _TRIGRAMS[:26]  # excludes "xxx"


def _keyed_alphabet(keyword: str) -> str:
    """Mixed 26-letter alphabet: dedup keyword letters, then append unused A-Z."""
    seq: list[str] = []
    for ch in (keyword or "").upper():
        if ch.isalpha() and ch not in seq:
            seq.append(ch)
    for ch in string.ascii_uppercase:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _encode_table(alphabet: str) -> dict[str, str]:
    """trigram -> label letter."""
    return {_LABELLED_TRIGRAMS[i]: alphabet[i] for i in range(26)}


def _decode_table(alphabet: str) -> dict[str, str]:
    """label letter -> trigram."""
    return {alphabet[i]: _LABELLED_TRIGRAMS[i] for i in range(26)}


class FractionatedMorse(Cipher):
    name = "fractionated-morse"
    aliases = ("fracmorse", "fractionated_morse")
    description = "Morse stream fractionated into trigrams labelled by a keyed alphabet."
    key_format = "keyword (letters; builds the 26-letter mixed alphabet labelling trigrams)"
    key_example = "CROWDED"
    needs_key = False
    complexity = 5

    def encode(self, text: str, key: str = "") -> str:
        morse = text_to_morse(text)
        if not morse:
            return ""
        # Pad the END with 'x' until length is a multiple of three.
        while len(morse) % 3 != 0:
            morse += "x"
        table = _encode_table(_keyed_alphabet(key))
        return "".join(table[morse[i : i + 3]] for i in range(0, len(morse), 3))

    def decode(self, text: str, key: str = "") -> str:
        table = _decode_table(_keyed_alphabet(key))
        morse = "".join(table[ch] for ch in text.upper() if ch in table)
        # Trailing padding is 'x's; morse_to_text ignores empty groups so the pad
        # collapses harmlessly when splitting on x / xx.
        return morse_to_text(morse)

    def _decode_morse(self, text: str, alphabet: str) -> str:
        """Decode using a pre-built alphabet string (crack inner loop)."""
        table = {alphabet[i]: _LABELLED_TRIGRAMS[i] for i in range(26)}
        morse = "".join(table.get(ch, "") for ch in text)
        return morse_to_text(morse)

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top=5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ):
        import random as _random

        cipher_letters = [c for c in text.upper() if c.isalpha()]
        if len(cipher_letters) < 4:
            return []
        ct = "".join(cipher_letters)
        if rng is None:
            rng = _random.Random(0xF7AC)

        deadline = None if timeout is None else time.monotonic() + timeout
        alphabet = list(string.ascii_uppercase)

        def score_alpha(alpha: list[str]) -> float:
            plain = self._decode_morse(ct, "".join(alpha))
            # Illegal morse (e.g. "xxx" runs) yields '?'-free but lossy output;
            # an empty / very short recovery is penalised heavily.
            if not plain:
                return -1e9
            return scorer.score(plain)

        best_alpha = alphabet[:]
        best_score = score_alpha(best_alpha)

        restarts = int(opts.get("restarts", 6))
        for _ in range(restarts):
            if deadline is not None and time.monotonic() > deadline:
                break
            cur = alphabet[:]
            rng.shuffle(cur)
            cur_score = score_alpha(cur)
            improved = True
            while improved:
                if deadline is not None and time.monotonic() > deadline:
                    break
                improved = False
                for i in range(26):
                    if deadline is not None and time.monotonic() > deadline:
                        break
                    for j in range(i + 1, 26):
                        cur[i], cur[j] = cur[j], cur[i]
                        trial = score_alpha(cur)
                        if trial > cur_score:
                            cur_score = trial
                            improved = True
                        else:
                            cur[i], cur[j] = cur[j], cur[i]
            if cur_score > best_score:
                best_score = cur_score
                best_alpha = cur[:]

        # Recover the keyword form of the winning alphabet for display, if simple.
        alpha_str = "".join(best_alpha)
        plain = self._decode_morse(ct, alpha_str)
        if not plain:
            return []
        conf = scorer.confidence(plain)
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=alpha_str,
                score=scorer.score(plain),
                confidence=conf,
                meta={"alphabet": alpha_str, "log_score": round(best_score, 2)},
            )
        ][:top]
