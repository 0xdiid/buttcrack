"""Sequence Transposition cipher (ACA / CryptoCrack).

The Sequence Transposition was introduced in the Nov-Dec 2015 issue of *The
Cryptogram* (ACA) by MSCREP. It is a columnar-style transposition driven by a
chain-addition digital sequence rather than by writing the plaintext into a
fixed-width block.

KEY format
----------
``"<5-digit primer>/<10-letter keyword>"`` (slash-separated), for example
``"69315/GUMMYBEARS"`` or ``"31752/CRYPTOGRAM"``.

* The **primer** is a 5-digit numeric group seeding a chain-addition (lag-2)
  *sequence*: starting from the primer, the next digit is the sum of the two
  preceding digits mod 10 (drop the carry). Digits are appended left to right
  until the sequence is as long as the plaintext. (Primer ``69315`` ->
  ``6+9=5, 9+3=2, 3+1=4, 1+5=6, 5+5=0, ...`` giving ``6 9 3 1 5 5 2 4 6 0 ...``.)
* The **keyword** must contain exactly 10 distinct-position letters. It is
  converted to a 10-digit *column header* by alphabetical rank, numbering the
  alphabetically-first letter ``1`` up through ``9`` and the tenth ``0`` (ties
  broken left to right). For example ``GUMMYBEARS -> 4956023178`` and
  ``CRYPTOGRAM -> 2706953814``.

ENCRYPT
-------
Lay the sequence digit under each plaintext letter. Drop each plaintext letter
into the column whose header digit equals that letter's sequence digit (so all
letters carrying sequence digit ``d`` collect, top to bottom, under the single
column headed ``d``). Finally read the columns off left to right in the original
keyword order, concatenating their contents, to form the ciphertext.

DECRYPT
-------
Regenerate the sequence for the ciphertext length, which fixes how many letters
fall in each column and the column each output position came from; slice the
ciphertext back into columns and redeal the letters in sequence order.

The key may also be given as ``"<primer>/<10 header digits>"`` (e.g.
``"69315/4,9,5,6,0,2,3,1,7,8"``) which names the column headers directly; this
is the form :meth:`SequenceTransposition.crack` reports so it round-trips
straight back into :meth:`decode`.

This implementation matches the worked examples on both the ACA cipher sheet
(``GUMMYBEARS`` / primer ``69315``) and the CryptoCrack user guide
(``CRYPTOGRAM`` / primer ``31752``).
"""

from __future__ import annotations

import time
from itertools import permutations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

PRIMER_LEN = 5
NUM_COLUMNS = 10


def _sequence(primer: list[int], length: int) -> list[int]:
    """Chain-addition (lag-2) digit sequence seeded by ``primer``.

    digit[n] = (digit[n-2] + digit[n-1]) mod 10, appended until ``length`` long.
    """
    digits = list(primer)
    i = 0
    while len(digits) < length:
        digits.append((digits[i] + digits[i + 1]) % 10)
        i += 1
    return digits[:length]


def _keyword_headers(keyword: str) -> list[int]:
    """Map a 10-letter keyword to its column-header digits (rank 1..9 then 0).

    The alphabetically-first letter is numbered ``1``, the next ``2``, ... the
    tenth ``0``; ties are broken left to right. (``GUMMYBEARS -> 4956023178``.)
    """
    kw = "".join(ch for ch in keyword.upper() if "A" <= ch <= "Z")
    if len(kw) != NUM_COLUMNS:
        raise ValueError(
            f"sequence-transposition keyword must be exactly {NUM_COLUMNS} letters, got {len(kw)}"
        )
    ranked = sorted(range(len(kw)), key=lambda i: (kw[i], i))
    headers = [0] * len(kw)
    for rank, idx in enumerate(ranked):
        headers[idx] = (rank + 1) % NUM_COLUMNS
    return headers


def _parse_headers(token: str) -> list[int]:
    """Parse an explicit ``"d,d,d,..."`` header list into 10 distinct digits."""
    digits = [int(x) for x in token.replace(",", " ").split() if x.strip()]
    if len(digits) != NUM_COLUMNS or sorted(digits) != list(range(NUM_COLUMNS)):
        raise ValueError("sequence-transposition header list must be a permutation of digits 0..9")
    return digits


def _parse_key(key: str) -> tuple[list[int], list[int]]:
    """Parse ``"<primer>/<keyword-or-headers>"`` into (primer, header digits)."""
    raw = str(key)
    if "/" in raw:
        primer_part, rest = raw.split("/", 1)
    else:
        primer_part = "".join(ch for ch in raw if ch.isdigit())[:PRIMER_LEN]
        rest = raw[len(primer_part) :]
    primer_digits = [int(ch) for ch in primer_part if ch.isdigit()]
    if len(primer_digits) != PRIMER_LEN:
        raise ValueError(
            f"sequence-transposition primer must be {PRIMER_LEN} digits, got {len(primer_digits)}"
        )
    rest = rest.strip()
    if any(ch.isalpha() for ch in rest):
        headers = _keyword_headers(rest)
    else:
        headers = _parse_headers(rest)
    return primer_digits, headers


def _encode_letters(letters: str, primer: list[int], headers: list[int]) -> str:
    seq = _sequence(primer, len(letters))
    digit_to_col = {d: c for c, d in enumerate(headers)}
    columns: list[list[str]] = [[] for _ in range(NUM_COLUMNS)]
    for ch, d in zip(letters, seq, strict=True):
        columns[digit_to_col[d]].append(ch)
    return "".join("".join(col) for col in columns)


def _decode_letters(cipher: str, primer: list[int], headers: list[int]) -> str:
    seq = _sequence(primer, len(cipher))
    digit_to_col = {d: c for c, d in enumerate(headers)}
    assign = [digit_to_col[d] for d in seq]
    lengths = [0] * NUM_COLUMNS
    for col in assign:
        lengths[col] += 1
    columns: list[list[str]] = []
    idx = 0
    for length in lengths:
        columns.append(list(cipher[idx : idx + length]))
        idx += length
    cursor = [0] * NUM_COLUMNS
    out: list[str] = []
    for col in assign:
        out.append(columns[col][cursor[col]])
        cursor[col] += 1
    return "".join(out)


class SequenceTransposition(Cipher):
    name = "sequence-transposition"
    aliases = ("seqtrans", "seqtransposition")
    description = "Sequence Transposition: columns chosen by a chain-addition digit sequence."
    key_format = "primer (5 digits)/10-letter keyword (slash-separated)"
    key_example = "69315/GUMMYBEARS"
    complexity = 6

    # Transposition cannot preserve spacing; operate on a clean letter stream.
    def encode(self, text: str, key: str) -> str:
        primer, headers = _parse_key(key)
        return _encode_letters(only_letters(text), primer, headers)

    def decode(self, text: str, key: str) -> str:
        primer, headers = _parse_key(key)
        return _decode_letters(only_letters(text), primer, headers)

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
        """Best-effort crack when the 5-digit primer is supplied.

        With the primer known (CryptoCrack's "Key Primer" setting), the cipher
        reduces to recovering which of the ten sequence-digit buckets occupies
        each of the ten output columns -- a permutation of ``0..9``. We hill-climb
        that permutation against ``scorer`` (random restarts, transposition
        moves), honoring ``timeout``. With no primer hint the keyless problem also
        ranges over 10^5 chains, which is not searched here, so we return ``[]``.

        Pass the primer via ``opts["primer"]`` (5 digits, e.g. ``"69315"``).
        Reported keys use the ``"<primer>/<10 header digits>"`` form, which feeds
        straight back into :meth:`decode`.
        """
        letters = only_letters(text)
        if len(letters) < 12:
            return []

        primer_opt = opts.get("primer")
        if primer_opt is None:
            return []
        primer = [int(ch) for ch in str(primer_opt) if ch.isdigit()]
        if len(primer) != PRIMER_LEN:
            return []

        import random

        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        # Pre-bucket the ciphertext is not possible (we have ciphertext, not
        # plaintext); instead decode is cheap, so score candidate permutations
        # directly. ``headers[col] = d`` means column ``col`` is headed by digit
        # ``d``; a permutation of 0..9 is exactly one such header assignment.
        def decode_with(headers: list[int]) -> str:
            return _decode_letters(letters, primer, headers)

        def score_of(headers: list[int]) -> float:
            return scorer.score(decode_with(headers))

        best_headers = list(range(NUM_COLUMNS))
        rng.shuffle(best_headers)
        best_score = score_of(best_headers)
        best_plain = decode_with(best_headers)

        restarts = 0
        while True:
            if deadline and time.monotonic() > deadline:
                break
            cur = list(range(NUM_COLUMNS))
            rng.shuffle(cur)
            cur_score = score_of(cur)
            improved = True
            while improved:
                improved = False
                if deadline and time.monotonic() > deadline:
                    break
                for a, b in permutations(range(NUM_COLUMNS), 2):
                    if a >= b:
                        continue
                    cur[a], cur[b] = cur[b], cur[a]
                    trial = score_of(cur)
                    if trial > cur_score:
                        cur_score = trial
                        improved = True
                    else:
                        cur[a], cur[b] = cur[b], cur[a]
                if deadline and time.monotonic() > deadline:
                    break
            if cur_score > best_score:
                best_score = cur_score
                best_headers = cur[:]
                best_plain = decode_with(cur)
            restarts += 1
            if restarts >= 60 and not deadline:
                break

        primer_str = "".join(str(d) for d in primer)
        key_repr = primer_str + "/" + ",".join(str(d) for d in best_headers)
        return [
            Candidate(
                plaintext=reflow(text, best_plain),
                cipher=self.name,
                key=key_repr,
                score=best_score,
                confidence=scorer.confidence(best_plain),
                meta={"primer": primer_str},
            )
        ][:top]
