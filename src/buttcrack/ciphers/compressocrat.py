"""Compressocrat cipher (ACA), introduced by SHMOO in *The Cryptogram* 1983.

The Compressocrat uses a fixed Huffman-style "compression alphabet" to turn each
plaintext letter into a short string of the digits ``1``/``2``/``3`` (common
letters get 2-digit codes, rare letters up to 6 digits), so the concatenated
digit stream is usually *shorter* than the plaintext. The stream is padded with
one or two trailing ``1`` digits to a multiple of three, split into trigrams,
and each trigram (one of the 26 patterns ``111``..``332`` over ``{1,2,3}``,
enumerated in base-3 order; ``333`` is unused) is relabelled by a keyed 26-letter
alphabet. Decryption reverses the relabelling to recover the digit stream and
parses it with the (prefix-free) compression alphabet, discarding the trailing
pad. The result is shorter ciphertext, hence "Compressocrat".

The fixed compression alphabet (ACA cipher sheet)::

    A 13     B 32112  C 1112   D 213    E 31     F 3213   G 32113
    H 113    I 322    J 321112 K 11112  L 212    M 2111   N 23
    O 22     P 3212   Q 11113  R 323    S 112    T 12     U 1113
    V 11111  W 2112   X 321111 Y 2113   Z 321113

KEY FORMAT
    A single keyword/keyphrase. Repeated letters are dropped and the remaining
    A-Z letters are appended in order to form the 26-letter mixed alphabet that
    labels the trigram columns (e.g. ``YZFRACTION`` ->
    ``YZFRACTIONBDEGHJKLMPQSUVWX``). An empty key gives the straight alphabet
    ``A-Z``. The 26 columns are the trigrams ``111`` (column 1), ``112``,
    ``113``, ``121``, ... ``332`` (column 26) in base-3 order; ``333`` is never
    emitted, so a 26-letter alphabet covers every column.
"""

from __future__ import annotations

import math
import random
import string
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

# Fixed compression alphabet: plaintext letter -> digit code (digits 1,2,3).
# This is a prefix-free (Huffman) code, so the digit stream is uniquely parseable.
_PT_TO_CODE: dict[str, str] = {
    "A": "13",
    "B": "32112",
    "C": "1112",
    "D": "213",
    "E": "31",
    "F": "3213",
    "G": "32113",
    "H": "113",
    "I": "322",
    "J": "321112",
    "K": "11112",
    "L": "212",
    "M": "2111",
    "N": "23",
    "O": "22",
    "P": "3212",
    "Q": "11113",
    "R": "323",
    "S": "112",
    "T": "12",
    "U": "1113",
    "V": "11111",
    "W": "2112",
    "X": "321111",
    "Y": "2113",
    "Z": "321113",
}
_CODE_TO_PT: dict[str, str] = {code: letter for letter, code in _PT_TO_CODE.items()}
# Distinct code lengths, ascending, for greedy prefix matching during parse.
_CODE_LENGTHS: tuple[int, ...] = tuple(sorted({len(c) for c in _PT_TO_CODE.values()}))

# The 26 labelled trigrams over {1,2,3} in base-3 order (1<2<3). The 27th, "333",
# is left unlabelled (a 26-letter alphabet has no cell for it).
_TRIGRAMS: list[str] = [a + b + c for a in "123" for b in "123" for c in "123"]
_LABELLED_TRIGRAMS: list[str] = _TRIGRAMS[:26]  # excludes "333"


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


def _parse_digit_stream(digits: str) -> tuple[str, int]:
    """Greedily parse a digit stream into plaintext using the prefix-free code.

    Returns ``(plaintext, leftover)`` where ``leftover`` is the count of trailing
    digits that are *not* a legitimate pad. A legitimate pad is one or two
    trailing ``1`` digits (added during encoding to reach a multiple of three),
    which yields ``leftover == 0``.
    """
    out: list[str] = []
    i, n = 0, len(digits)
    while i < n:
        matched = False
        for length in _CODE_LENGTHS:
            seg = digits[i : i + length]
            letter = _CODE_TO_PT.get(seg)
            if letter is not None and len(seg) == length:
                out.append(letter)
                i += length
                matched = True
                break
        if not matched:
            break
    tail = digits[i:]
    leftover = 0 if tail in ("", "1", "11") else len(tail)
    return "".join(out), leftover


def _encode_table(alphabet: str) -> dict[str, str]:
    """trigram -> label letter."""
    return {_LABELLED_TRIGRAMS[i]: alphabet[i] for i in range(26)}


def _decode_table(alphabet: str) -> dict[str, str]:
    """label letter -> trigram."""
    return {alphabet[i]: _LABELLED_TRIGRAMS[i] for i in range(26)}


def _decode_with_alphabet(ciphertext: str, alphabet: str) -> tuple[str, int]:
    """Decode A-Z ciphertext through ``alphabet``; return (plaintext, leftover)."""
    table = _decode_table(alphabet)
    digits = "".join(table.get(ch, "") for ch in ciphertext)
    return _parse_digit_stream(digits)


class Compressocrat(Cipher):
    name = "compressocrat"
    aliases = ("compresso",)
    description = (
        "Huffman-style digit compression fractionated into trigrams labelled by a keyed alphabet."
    )
    key_format = "keyword (letters; deduped, completed to a 26-letter alphabet)"
    key_example = "CIPHER"
    needs_key = False
    complexity = 6

    def encode(self, text: str, key: str = "") -> str:
        letters = only_letters(text)
        if not letters:
            return ""
        digits = "".join(_PT_TO_CODE[ch] for ch in letters)
        # Pad the END with '1' until length is a multiple of three.
        while len(digits) % 3 != 0:
            digits += "1"
        table = _encode_table(_keyed_alphabet(key))
        return "".join(table[digits[i : i + 3]] for i in range(0, len(digits), 3))

    def decode(self, text: str, key: str = "") -> str:
        plaintext, _ = _decode_with_alphabet(only_letters(text), _keyed_alphabet(key))
        return plaintext

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
        """Keyless best-effort: simulated-annealing search over the keyed alphabet.

        Each ciphertext letter only encodes a 3-digit group, and the (fixed)
        compression code re-segments the *concatenated* digit stream, so a single
        alphabet swap changes the whole recovered text non-locally. That makes the
        objective landscape very rugged; plain hill-climbing stalls in local
        optima, so this uses simulated annealing (as for Trifid/Bifid). It needs a
        fair amount of text (ACA recommends 110-150 plaintext letters) and a
        generous ``timeout`` (~120-180s) to converge reliably.
        """
        ct = only_letters(text)
        if len(ct) < 30:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None
        iters = int(opts.get("iters", 1500))
        temp0 = float(opts.get("temp", 8.0))
        step = float(opts.get("temp_step", 0.4))

        def objective(alphabet: str) -> float:
            plain, leftover = _decode_with_alphabet(ct, alphabet)
            if len(plain) < 3:
                return float("-inf")
            # Penalise digits that fail to parse (a wrong alphabet leaves garbage).
            return scorer.score(plain) - leftover * 40.0

        best_alpha: str | None = None
        best_obj = float("-inf")
        base = list(string.ascii_uppercase)

        while deadline is None or time.monotonic() < deadline:
            parent = base[:]
            rng.shuffle(parent)
            cur = objective("".join(parent))
            temp = temp0
            while temp > 0:
                if deadline is not None and time.monotonic() > deadline:
                    break
                for _ in range(iters):
                    i, j = rng.randrange(26), rng.randrange(26)
                    if i == j:
                        continue
                    child = parent[:]
                    child[i], child[j] = child[j], child[i]
                    cand = objective("".join(child))
                    delta = cand - cur
                    if delta > 0 or rng.random() < math.exp(delta / temp):
                        parent, cur = child, cand
                        if cand > best_obj:
                            best_obj = cand
                            best_alpha = "".join(child)
                temp -= step
            if deadline is None:
                break  # no timeout => single annealing pass

        if best_alpha is None:
            return []
        plain, _ = _decode_with_alphabet(ct, best_alpha)
        if not plain:
            return []
        return [
            Candidate(
                plaintext=plain,
                cipher=self.name,
                key=best_alpha,
                score=scorer.score(plain),
                confidence=scorer.confidence(plain),
                meta={"alphabet": best_alpha, "log_score": round(best_obj, 2)},
            )
        ][:top]
