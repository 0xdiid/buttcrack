"""Gromark cipher (GRonsfeld + Mixed Alphabet + Running Key).

The Gromark is a periodic-running-key polyalphabetic cipher used by the ACA.

KEY format
----------
``"<5-digit primer>/<keyword>"`` (slash-separated), for example ``"32941/GRONSFELD"``.

* The **primer** is a 5-digit numeric group. It seeds a lagged-Fibonacci /
  chain-addition running key: the next digit is ``(d[i] + d[i+1]) mod 10``,
  where the lag equals the primer length (5). Digits are appended left to right
  until the key is as long as the plaintext. (Primer ``12345`` ->
  ``1+2=3, 2+3=5, 3+4=7, 4+5=9, 5+3=8, ...`` giving ``1 2 3 4 5 3 5 7 9 8 ...``.)
* The **keyword** builds a mixed (cipher) alphabet by a K2-type columnar
  ("sequence") transposition: form the keyed alphabet (deduped keyword followed
  by the rest of A-Z), write it row-wise into a block whose width equals the
  keyword length, number the columns by the alphabetical order of the keyword
  letters, then read the block off column by column in that numeric order.

ENCRYPT
-------
Align the straight alphabet ``A-Z`` (indices 0..25) above the mixed alphabet.
For each plaintext letter take its index ``i`` in the straight alphabet, add the
running-key digit ``d`` (0..9), and emit ``MIXED[(i + d) mod 26]``.

DECRYPT
-------
Find the ciphertext letter's index ``j`` in the mixed alphabet; the plaintext
letter is ``STRAIGHT[(j - d) mod 26]``.

Encrypt and decrypt are NOT reciprocal.
"""

from __future__ import annotations

import itertools
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import ALPHABET, only_letters, reflow
from .base import Cipher

PRIMER_LEN = 5


def _keyed_alphabet(keyword: str) -> str:
    """Deduped keyword letters followed by the remaining A-Z letters."""
    seq: list[str] = []
    for ch in keyword.upper():
        if "A" <= ch <= "Z" and ch not in seq:
            seq.append(ch)
    for ch in ALPHABET:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _column_order(keyword: str) -> list[int]:
    """Rank each column by the alphabetical order of the keyword letters.

    ``order[col]`` is the 0-based rank of that column; ties (repeated letters)
    are broken left to right.
    """
    kw = keyword.upper()
    ranked_indices = sorted(range(len(kw)), key=lambda i: (kw[i], i))
    order = [0] * len(kw)
    for rank, idx in enumerate(ranked_indices):
        order[idx] = rank
    return order


def _mixed_alphabet(keyword: str) -> str:
    """Build the 26-letter mixed alphabet via K2-type columnar transposition."""
    kw = "".join(ch for ch in keyword.upper() if "A" <= ch <= "Z")
    if not kw:
        raise ValueError("gromark keyword must contain letters")
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


def _running_key(primer: list[int], length: int) -> list[int]:
    """Chain-addition keystream: digit n = (digit[n-lag] + digit[n-lag+1]) % 10."""
    lag = len(primer)
    digits = list(primer)
    while len(digits) < length:
        nxt = (digits[len(digits) - lag] + digits[len(digits) - lag + 1]) % 10
        digits.append(nxt)
    return digits[:length]


def _parse_key(key: str) -> tuple[list[int], str]:
    """Parse ``"<primer>/<keyword>"`` into (primer digits, keyword)."""
    raw = str(key)
    if "/" in raw:
        primer_part, keyword = raw.split("/", 1)
    else:
        # Fall back to leading digits + trailing letters.
        primer_part = "".join(ch for ch in raw if ch.isdigit())
        keyword = "".join(ch for ch in raw if ch.isalpha())
    primer_digits = [int(ch) for ch in primer_part if ch.isdigit()]
    _fmt = f"key must be '<{PRIMER_LEN}-digit primer>/<keyword>' e.g. '32941/GRONSFELD'"
    if len(primer_digits) != PRIMER_LEN:
        raise ValueError(f"gromark {_fmt} — got {len(primer_digits)} primer digits")
    if not only_letters(keyword):
        raise ValueError(f"gromark {_fmt} — missing keyword")
    return primer_digits, keyword


def _encode_letters(letters: str, primer: list[int], mixed: str) -> str:
    rk = _running_key(primer, len(letters))
    out = []
    for ch, d in zip(letters, rk, strict=True):
        i = ord(ch) - 65
        out.append(mixed[(i + d) % 26])
    return "".join(out)


def _decode_letters(letters: str, primer: list[int], mixed: str) -> str:
    rk = _running_key(primer, len(letters))
    pos = {ch: i for i, ch in enumerate(mixed)}
    out = []
    for ch, d in zip(letters, rk, strict=True):
        j = pos[ch]
        out.append(ALPHABET[(j - d) % 26])
    return "".join(out)


class Gromark(Cipher):
    name = "gromark"
    aliases = ("gronsfeldmixed",)
    description = "Gromark: chain-addition running key over a K2-mixed alphabet."
    key_format = "primer (5 digits)/keyword (slash-separated)"
    key_example = "32941/GRONSFELD"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        primer, keyword = _parse_key(key)
        mixed = _mixed_alphabet(keyword)
        return _encode_letters(only_letters(text), primer, mixed)

    def decode(self, text: str, key: str) -> str:
        primer, keyword = _parse_key(key)
        mixed = _mixed_alphabet(keyword)
        return _decode_letters(only_letters(text), primer, mixed)

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
        """Best-effort crack when a candidate keyword (or word list) is supplied.

        The full keyless Gromark problem couples two large unknowns: a 5-digit
        primer (10^5 chains) and a mixed alphabet (effectively a 26-letter
        permutation). Searching both jointly is intractable within a quadgram
        budget, so this crack is deliberately scoped: if the caller supplies a
        keyword via ``opts["keyword"]`` (or a list via ``opts["keywords"]``), we
        brute-force all 100000 primers against that mixed alphabet and return the
        best-scoring decryptions. With no keyword hint we return ``[]`` rather
        than pretend to solve an intractable instance.
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
        results: list[tuple[float, str, str]] = []  # (score, plaintext, key_repr)
        for keyword in keywords:
            try:
                mixed = _mixed_alphabet(keyword)
            except ValueError:
                continue
            pos = {ch: i for i, ch in enumerate(mixed)}
            for combo in itertools.product(range(10), repeat=PRIMER_LEN):
                if deadline and time.monotonic() > deadline:
                    break
                primer = list(combo)
                rk = _running_key(primer, len(letters))
                plain = "".join(
                    ALPHABET[(pos[ch] - d) % 26] for ch, d in zip(letters, rk, strict=True)
                )
                key_repr = "".join(str(d) for d in primer) + "/" + keyword.upper()
                results.append((scorer.score(plain), plain, key_repr))
            if deadline and time.monotonic() > deadline:
                break

        results.sort(key=lambda r: r[0], reverse=True)
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for score, plain, key_repr in results:
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=key_repr,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"primer": key_repr.split("/", 1)[0]},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
