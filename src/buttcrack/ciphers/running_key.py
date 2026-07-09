"""Running Key cipher: Vigenere with a key as long as the message.

Instead of a short repeating keyword, the key is a long stretch of meaningful
text (e.g. a book passage) aligned one-to-one with the plaintext, so the key
never repeats and Kasiski/IoC period analysis gives no foothold. The standard
(ACA) combiner is Vigenere::

    encrypt:  C_i = (P_i + K_i) mod 26
    decrypt:  P_i = (C_i - K_i) mod 26

where ``K`` is the running-key text. The ACA worked example is a *self-key*:
the first half of the message enciphers the second half.

KEY FORMAT
----------
A long key text (any string; only its letters A-Z are used, case-insensitively).
It must contain at least as many letters as the message. Non-letters in both the
message and the key are ignored for the letter-by-letter combine, but the
original message layout (spaces/punctuation/case) is restored on output.

Not reciprocal: encode and decode are distinct (additive vs subtractive).
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher


def _key_indices(key: str, needed: int) -> list[int]:
    idx = [ord(c) - 65 for c in only_letters(key)]
    if not idx:
        raise ValueError("running-key key must contain letters")
    if len(idx) < needed:
        raise ValueError(
            f"running-key key has {len(idx)} letters but the message needs {needed}; "
            "the key text must be at least as long as the message"
        )
    return idx


def _to_letters(idx: list[int]) -> str:
    return "".join(chr(i + 65) for i in idx)


class RunningKey(Cipher):
    """Vigenere with a non-repeating, message-length natural-language key.

    The key is a long key text; only its letters are used and it must be at
    least as long as the message. ``C = (P + K) mod 26`` per the ACA convention.
    """

    name = "running-key"
    aliases = ("runningkey", "running_key")
    description = "Vigenere with a key as long as the message; C = (P + K) mod 26."
    key_format = "long key text, at least as long as the message (letters)"
    key_example = "the quick brown fox jumps over the lazy dog and then some more padding text here"
    complexity = 5
    # Keyless recovery is ill-posed (key length == text length): the beam search
    # produces confident English-looking garbage, so keep it out of `auto`.
    auto_crackable = False

    def encode(self, text: str, key: str) -> str:
        plain = [ord(c) - 65 for c in only_letters(text)]
        ks = _key_indices(key, len(plain))
        cipher = [(plain[i] + ks[i]) % 26 for i in range(len(plain))]
        return reflow(text, _to_letters(cipher))

    def decode(self, text: str, key: str) -> str:
        cipher = [ord(c) - 65 for c in only_letters(text)]
        ks = _key_indices(key, len(cipher))
        plain = [(cipher[i] - ks[i]) % 26 for i in range(len(cipher))]
        return reflow(text, _to_letters(plain))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Best-effort keyless attack via a joint plaintext+key beam search.

        With no repeating key there is no period to find. The only leverage is
        that BOTH the plaintext and the running key are natural English, so we
        beam-search the keystream letter by letter, scoring each partial pair by
        the quadgram fitness of the recovered plaintext AND of the implied key.

        This returns ranked English-on-both-streams candidates (useful as a
        human-assisted starting point), but it does NOT reliably recover the
        exact plaintext: with two English streams the combined score is
        symmetric under swapping plaintext<->key, and the streams readily
        interleave, so short messages are inherently ambiguous. Treat the output
        as hypotheses, not a guaranteed solution.
        """
        letters = only_letters(text)
        if len(letters) < scorer.n + 1:
            return []
        cvals = [ord(c) - 65 for c in letters]
        n = scorer.n
        log_probs = scorer.log_probs
        floor = scorer.floor
        beam = int(opts.get("beam", 400))
        deadline = (time.monotonic() + timeout) if timeout else None

        # Each state: (combined_score, key_letters, plain_letters).
        states: list[tuple[float, str, str]] = [(0.0, "", "")]
        for c in cvals:
            nxt: list[tuple[float, str, str]] = []
            for sc, key, plain in states:
                for k in range(26):
                    new_key = key + chr(k + 65)
                    new_plain = plain + chr((c - k) % 26 + 65)
                    add = 0.0
                    if len(new_key) >= n:
                        add += log_probs.get(new_key[-n:], floor)
                        add += log_probs.get(new_plain[-n:], floor)
                    nxt.append((sc + add, new_key, new_plain))
            nxt.sort(key=lambda s: s[0], reverse=True)
            states = nxt[:beam]
            if deadline and time.monotonic() > deadline:
                break

        states.sort(key=lambda s: s[0], reverse=True)
        candidates = [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key=key,
                score=scorer.score(plain),
                confidence=scorer.confidence(plain),
                meta={"key": key},
            )
            for _sc, key, plain in states[:top]
        ]
        return candidates
