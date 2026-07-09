"""Ragbaby cipher — a keyed 24-letter alphabet with a positional shift schedule.

The Ragbaby is a polyalphabetic substitution whose shift comes not from a
repeating keyword but from each letter's *position*: words of the plaintext are
numbered 1, 2, 3, ...; within each word the letters are numbered consecutively
starting from that word's number (word 1 -> letters 1,2,3,...; word 2 ->
2,3,4,...; etc.). Each plaintext letter is enciphered by counting forward in a
24-letter keyed alphabet by its position number; decryption counts backward.

The 24-letter alphabet merges ``J -> I`` and ``X -> W`` (the ACA standard), so a
plaintext ``J``/``X`` is treated as ``I``/``W`` and is not recoverable on decode.
Word divisions (spaces) are preserved and are part of the ciphertext; a
hyphenated or apostrophe-joined token counts as a single word.

KEY format
----------
A single keyword. The 24-letter keyed alphabet is built by writing the keyword
with repeats dropped, appending the unused letters A-Z in order, then merging
``J -> I`` and ``X -> W`` (e.g. ``FRANKLIN -> FRANKLIBCDEGHMOPQSTUVWYZ``).

As a convenience the key may instead *spell out* the finished alphabet: if the
last 24 letters of the key (after the J/X merge) already form a permutation of
the 24-letter set, those letters are used verbatim. This lets a key such as
``"Keyed alphabet ALPHBETCDFGIKMNOQRSUVWYZ"`` select that exact alphabet.
"""

from __future__ import annotations

import random
import re
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

# The 24-letter alphabet in natural order, with J and X removed (J->I, X->W).
ALPHABET_24 = "ABCDEFGHIKLMNOPQRSTUVWYZ"
_MERGE = {"J": "I", "X": "W"}
SIZE = 24

# A "word" for numbering purposes: a run of letters, optionally joined by single
# apostrophes or hyphens (so "don't" / "well-known" count as one word).
_WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")


def _merge_letters(text: str) -> str:
    """Uppercase A-Z only, applying the J->I and X->W merge."""
    out: list[str] = []
    for ch in text.upper():
        if "A" <= ch <= "Z":
            out.append(_MERGE.get(ch, ch))
    return "".join(out)


def build_alphabet(key: str) -> str:
    """Build the 24-letter keyed alphabet from ``key``.

    If the key's trailing 24 letters (post-merge) already spell a permutation of
    the 24-letter set, use them directly; otherwise treat the key as a keyword.
    """
    merged = _merge_letters(key)
    tail = merged[-SIZE:]
    if len(tail) == SIZE and set(tail) == set(ALPHABET_24):
        return tail

    seq: list[str] = []
    for ch in merged + ALPHABET_24:
        if ch not in seq:
            seq.append(ch)
    if len(seq) != SIZE:  # pragma: no cover - ALPHABET_24 guarantees completion
        raise ValueError(f"ragbaby alphabet needs {SIZE} letters, built {len(seq)}")
    return "".join(seq)


def _shift_letter(alphabet: str, pos: dict[str, int], ch: str, amount: int) -> str:
    """Shift a single (already-merged) letter by ``amount`` in the keyed alphabet."""
    return alphabet[(pos[ch] + amount) % SIZE]


def _transform(text: str, key: str, *, sign: int) -> str:
    """Encode (sign=+1) or decode (sign=-1) preserving layout and word numbering.

    Position number for the k-th (0-indexed) letter of word number ``w`` (1-based)
    is ``w + k``; numbers that exceed 24 wrap modulo 24 (handled by the shift mod).
    """
    alphabet = build_alphabet(key)
    pos = {ch: i for i, ch in enumerate(alphabet)}

    # Number the words by scanning for word tokens in the original text.
    word_no = 0
    char_word: list[int] = [0] * len(text)  # word number covering each char (0 = none)
    letter_index: list[int] = [0] * len(text)  # 0-based letter position within its word
    for m in _WORD_RE.finditer(text):
        word_no += 1
        li = 0
        for i in range(m.start(), m.end()):
            if text[i].isalpha():
                char_word[i] = word_no
                letter_index[i] = li
                li += 1

    out: list[str] = []
    for i, ch in enumerate(text):
        if not ch.isalpha():
            out.append(ch)
            continue
        amount = sign * (char_word[i] + letter_index[i])
        merged = _MERGE.get(ch.upper(), ch.upper())
        enc = _shift_letter(alphabet, pos, merged, amount)
        out.append(enc.lower() if ch.islower() else enc)
    return "".join(out)


class Ragbaby(Cipher):
    name = "ragbaby"
    aliases = ()
    description = "Keyed 24-letter alphabet shifted by each letter's word/position number."
    key_format = "single keyword (builds the 24-letter keyed alphabet, J->I/X->W)"
    key_example = "FRANKLIN"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        return _transform(text, key, sign=+1)

    def decode(self, text: str, key: str) -> str:
        return _transform(text, key, sign=-1)

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
        """Best-effort hill-climb over the 24-letter keyed alphabet.

        The shift schedule is fixed by word/letter position, so cracking reduces
        to recovering the keyed alphabet. We hill-climb a permutation of the
        24-letter alphabet by swapping pairs and scoring the full decryption with
        the quadgram scorer. Word/space layout in ``text`` carries the numbering,
        so it must be preserved on input.
        """
        # Need at least some letters and ideally several words for the numbering
        # to constrain the alphabet.
        if len(_merge_letters(text)) < 12:
            return []
        rng = rng or random.Random()
        # Each restart is cheap but the per-position shift schedule makes the
        # surface rugged, so many random restarts are needed to escape local
        # optima. Default high and rely on the deadline to stop early.
        restarts = int(opts.get("restarts", 500))
        deadline = (time.monotonic() + timeout) if timeout else None

        base = list(ALPHABET_24)

        def decrypt(alpha: str) -> str:
            return _transform(text, alpha, sign=-1)

        results: list[tuple[float, str]] = []
        for r in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = base[:]
            if r != 0:  # first restart starts from the natural-order alphabet
                rng.shuffle(parent)
            parent_key = "".join(parent)
            parent_score = scorer.score(_merge_letters(decrypt(parent_key)))
            improved = True
            timed_out = False
            while improved and not timed_out:
                improved = False
                for a in range(SIZE - 1):
                    if deadline and time.monotonic() > deadline:
                        timed_out = True
                        break
                    for b in range(a + 1, SIZE):
                        child = parent[:]
                        child[a], child[b] = child[b], child[a]
                        child_key = "".join(child)
                        s = scorer.score(_merge_letters(decrypt(child_key)))
                        if s > parent_score:
                            parent_score, parent, improved = s, child, True
            results.append((parent_score, "".join(parent)))

        results.sort(key=lambda rs: rs[0], reverse=True)
        seen: set[str] = set()
        candidates: list[Candidate] = []
        for score, alpha in results:
            plain = decrypt(alpha)
            letters = _merge_letters(plain)
            if letters in seen:
                continue
            seen.add(letters)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=alpha,
                    score=score,
                    confidence=scorer.confidence(letters),
                    meta={"restarts": restarts},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
