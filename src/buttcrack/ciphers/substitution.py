"""Monoalphabetic substitution with quadgram hill-climbing (the keyless workhorse)."""

from __future__ import annotations

import random
import time
from collections import Counter

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

# English letters by descending frequency, for the frequency-seeded first restart.
_ENGLISH_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def _parse_key(key: str) -> str:
    k = only_letters(key)
    if len(k) != 26 or len(set(k)) != 26:
        raise ValueError("substitution key must be a 26-letter permutation of A-Z")
    return k


def _encode(text: str, cipher_alphabet: str) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(cipher_alphabet[ord(ch) - 65])
        elif "a" <= ch <= "z":
            out.append(cipher_alphabet[ord(ch) - 97].lower())
        else:
            out.append(ch)
    return "".join(out)


def _decode(text: str, cipher_alphabet: str) -> str:
    inverse = {c: chr(65 + i) for i, c in enumerate(cipher_alphabet)}
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(inverse[ch])
        elif "a" <= ch <= "z":
            out.append(inverse[ch.upper()].lower())
        else:
            out.append(ch)
    return "".join(out)


def _decrypt_with_dec(letters: str, dec: list[str]) -> str:
    """dec[cipher_index] = plaintext letter."""
    return "".join(dec[ord(c) - 65] for c in letters)


def _freq_seed(letters: str) -> list[str]:
    """Seed the decrypt map by aligning cipher-letter frequency to English."""
    counts = Counter(letters)
    cipher_by_freq = [chr(65 + i) for i in range(26)]
    cipher_by_freq.sort(key=lambda c: -counts.get(c, 0))
    dec = ["A"] * 26
    for cipher_letter, plain_letter in zip(cipher_by_freq, _ENGLISH_ORDER, strict=True):
        dec[ord(cipher_letter) - 65] = plain_letter
    return dec


def _dec_to_cipher_alphabet(dec: list[str]) -> str:
    """Convert a decrypt map (cipher->plain) to an encode key (plain->cipher)."""
    alphabet = [""] * 26
    for cipher_index, plain in enumerate(dec):
        alphabet[ord(plain) - 65] = chr(65 + cipher_index)
    return "".join(alphabet)


class Substitution(Cipher):
    name = "substitution"
    aliases = ("subst", "monoalpha", "aristocrat")
    description = "General monoalphabetic substitution; cracked by quadgram hill-climbing."
    key_format = "26-letter A-Z permutation"
    key_example = "QWERTYUIOPASDFGHJKLZXCVBNM"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        return _encode(text, _parse_key(key))

    def decode(self, text: str, key: str) -> str:
        return _decode(text, _parse_key(key))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        letters = only_letters(text)
        if len(letters) < 8:
            return []
        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 30))
        deadline = (time.monotonic() + timeout) if timeout else None

        results: list[tuple[float, list[str]]] = []
        for r in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = _freq_seed(letters) if r == 0 else _shuffled(rng)
            parent_score = scorer.score(_decrypt_with_dec(letters, parent))
            improved = True
            timed_out = False
            while improved and not timed_out:
                improved = False
                for i in range(25):
                    # Check inside the sweep so a single 325-swap pass can't
                    # overrun the budget on long text.
                    if deadline and time.monotonic() > deadline:
                        timed_out = True
                        break
                    for j in range(i + 1, 26):
                        child = parent[:]
                        child[i], child[j] = child[j], child[i]
                        s = scorer.score(_decrypt_with_dec(letters, child))
                        if s > parent_score:
                            parent_score, parent, improved = s, child, True
            results.append((parent_score, parent))

        # Deduplicate by the resulting plaintext, keep the best-scoring maps.
        results.sort(key=lambda rs: rs[0], reverse=True)
        seen: set[str] = set()
        candidates: list[Candidate] = []
        for score, dec in results:
            plain = _decrypt_with_dec(letters, dec)
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=_dec_to_cipher_alphabet(dec),
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"restarts": restarts},
                )
            )
            if len(candidates) >= top:
                break
        return candidates


def _shuffled(rng: random.Random) -> list[str]:
    letters = [chr(65 + i) for i in range(26)]
    rng.shuffle(letters)
    return letters
