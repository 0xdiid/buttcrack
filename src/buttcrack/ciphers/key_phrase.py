"""Key Phrase cipher: a 26-letter key phrase placed *under* a straight alphabet.

First described by Helen Fouche Gaines (The Cryptogram, Oct 1937; later in
*Elementary Cryptanalysis*). The key is a phrase of exactly 26 letters written
directly beneath the plaintext alphabet::

    plain:  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
    key:    W H A T S A N O T H E R W O R D F O R S Y N O N Y M

Each plaintext letter is replaced by the key-phrase letter sitting beneath it
(A->W, B->H, ...). Because a key phrase normally repeats letters, the cipher is
many-to-one: several plaintext letters share one ciphertext letter, so decoding
is ambiguous and word divisions are preserved to help the solver. When the key
phrase happens to be a 26-letter *permutation* of the alphabet the map is a
plain monoalphabetic substitution and decoding is unique.

KEY FORMAT
    A string of exactly 26 letters (the key phrase). Case and non-letters in the
    key argument are ignored; only the 26 A-Z letters are used, in order.

This module mirrors the CryptoCrack convention; its worked example
(key phrase ``WHATSANOTHERWORDFORSYNONYM``) is reproduced exactly by ``encode``.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher


def _parse_key(key: str) -> str:
    """Return the 26-letter key phrase (A-Z, uppercased)."""
    phrase = only_letters(key)
    if len(phrase) != 26:
        raise ValueError("key phrase must contain exactly 26 letters (A-Z)")
    return phrase


def _encode_letters(letters: str, phrase: str) -> str:
    """Substitute each A-Z letter by the key-phrase letter beneath it."""
    return "".join(phrase[ord(ch) - 65] for ch in letters)


def _inverse_map(phrase: str) -> dict[str, str]:
    """Map each ciphertext letter to a chosen plaintext letter.

    The key phrase is many-to-one, so the inverse is generally not a function.
    We pick, for each cipher letter, the *first* plaintext letter (lowest A-Z)
    that produces it. This makes decode deterministic and exact whenever the
    phrase is a permutation (the only case where decode can be lossless).
    """
    inverse: dict[str, str] = {}
    for plain_idx, cipher_ch in enumerate(phrase):
        inverse.setdefault(cipher_ch, chr(65 + plain_idx))
    return inverse


class KeyPhrase(Cipher):
    name = "key-phrase"
    aliases = ("keyphrase",)
    description = (
        "Key Phrase substitution: a 26-letter key phrase placed under the alphabet (many-to-one)."
    )
    key_format = "exactly 26 letters (the key phrase, placed under A-Z)"
    key_example = "WHATSANOTHERWORDFORSYNONYM"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        phrase = _parse_key(key)
        return _encode_letters(only_letters(text), phrase)

    def decode(self, text: str, key: str) -> str:
        phrase = _parse_key(key)
        inverse = _inverse_map(phrase)
        letters = only_letters(text)
        return "".join(inverse[ch] for ch in letters)

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
        """Best-effort keyless recovery by hill-climbing the inverse alphabet.

        The Key Phrase cipher is many-to-one and lossy: distinct plaintext
        letters collapse onto a shared ciphertext letter, so the original
        plaintext cannot in general be recovered from letters alone. Recovery is
        only well-posed when the underlying key phrase was a 26-letter
        permutation (the cipher then degenerates to a monoalphabetic
        substitution); for genuinely collapsed keys this returns low-confidence
        hypotheses that should be treated as guesses.
        """
        letters = only_letters(text)
        if len(letters) < 16:
            return []
        rng = rng or random.Random()
        restarts = int(opts.get("restarts", 30))
        deadline = (time.monotonic() + timeout) if timeout else None

        # Recoverability lives entirely in the permutation subcase, where the
        # cipher reduces to a monoalphabetic substitution and the cipher->plain
        # inverse is a bijection. We therefore hill-climb on that 26-letter
        # inverse with pairwise swaps (the proven monoalphabetic move), exactly
        # as the substitution solver does. Many-to-one keys collapse letters and
        # cannot be inverted from letters alone; those land as low-confidence
        # hypotheses.
        def decrypt(inverse: list[str]) -> str:
            return "".join(inverse[ord(c) - 65] for c in letters)

        def shuffled() -> list[str]:
            inv = [chr(65 + i) for i in range(26)]
            rng.shuffle(inv)
            return inv

        results: list[tuple[float, list[str]]] = []
        for _ in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = shuffled()
            parent_score = scorer.score(decrypt(parent))
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
                        s = scorer.score(decrypt(child))
                        if s > parent_score:
                            parent_score, parent, improved = s, child, True
            results.append((parent_score, parent))

        results.sort(key=lambda rs: rs[0], reverse=True)
        seen: set[str] = set()
        candidates: list[Candidate] = []
        for score, inverse in results:
            plain = decrypt(inverse)
            if plain in seen:
                continue
            seen.add(plain)
            # Reconstruct a key phrase consistent with this inverse: plaintext
            # letter P enciphers to the cipher letter C with inverse[C-65]==P.
            phrase = [""] * 26
            for cipher_idx, plain_ch in enumerate(inverse):
                phrase[ord(plain_ch) - 65] = chr(65 + cipher_idx)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key="".join(phrase),
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"restarts": restarts},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
