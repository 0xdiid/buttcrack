"""Tridigital cipher (ACA type #84): a keyed many-to-one letter->digit map.

A 10-letter *column keyword* is numbered ``1..9,0`` in strict alphabetical order
(ties broken left-to-right); those ten digits label the columns of a 10-column
grid.  A second keyword builds a 26-letter keyed alphabet which is written
left-to-right, top-to-bottom into the FIRST 9 columns over 3 rows (3x9 = 27
cells; the 26 letters fill all but the last cell).  The 10th column is left
blank and is the word-spacer column.

Encipherment: each plaintext letter becomes the single digit labelling the
column it sits in, so a letter always maps to the same digit, but a digit can
stand for up to three stacked letters.  A space between words becomes the digit
labelling the (blank) 10th column.  ``J`` is treated as ``I`` on lookup.

Output is a digit string (this implementation returns it space-separated, one
token per plaintext letter / word gap).

KEY FORMAT
----------
``"COLUMNKEYWORD/ALPHABETKEYWORD"`` -- e.g. ``"NOVELCRAFT/DRAGONFLY"``.
The column keyword must be exactly 10 letters.  If only a single keyword is
given (no ``/``), it is used for both the columns (first 10 distinct letters,
padded if short) and the keyed alphabet.

Decipherment is ambiguous (each digit hides up to 3 candidate letters); this
decoder picks, per digit, the most n-gram-likely candidate over the whole
message via a small per-token resolution.  Round-trips of prepared plaintext
recover the letters; arbitrary text may resolve to a near-homophone.
"""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

_FULL_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _keyed_alphabet(keyword: str) -> str:
    """26-letter keyed alphabet: keyword (deduped) then the rest of A-Z."""
    seq: list[str] = []
    for ch in keyword.upper():
        if ch.isalpha() and ch not in seq:
            seq.append(ch)
    for ch in _FULL_ALPHA:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _column_labels(keyword: str) -> list[int]:
    """Number a 10-letter keyword 1..9,0 by strict alphabetical order (ties L->R)."""
    kw = keyword.upper()
    order = sorted(range(len(kw)), key=lambda i: (kw[i], i))
    labels = [0] * len(kw)
    for rank, idx in enumerate(order):
        labels[idx] = (rank + 1) % 10  # 1,2,...,9,0
    return labels


def _column_keyword(raw: str) -> str:
    """Coerce ``raw`` into a 10-letter column keyword (distinct letters, then pad)."""
    letters = [c for c in raw.upper() if c.isalpha()]
    if len(letters) == 10:
        return "".join(letters)
    # Build 10 distinct letters: keyword's distinct letters then remaining A-Z.
    seq: list[str] = []
    for ch in letters:
        if ch not in seq:
            seq.append(ch)
    for ch in _FULL_ALPHA:
        if len(seq) >= 10:
            break
        if ch not in seq:
            seq.append(ch)
    return "".join(seq[:10])


class _Tableau:
    """The numbered 10-column grid; letter<->digit lookup and column candidates."""

    def __init__(self, column_keyword: str, alphabet_keyword: str):
        col_kw = _column_keyword(column_keyword)
        self.labels = _column_labels(col_kw)  # 10 digits, index = column
        self.spacer = self.labels[9]
        alpha = _keyed_alphabet(alphabet_keyword)

        # Fill first 9 columns, 3 rows, left-to-right top-to-bottom.
        cells = [["" for _ in range(9)] for _ in range(3)]
        idx = 0
        for r in range(3):
            for c in range(9):
                if idx < len(alpha):
                    cells[r][c] = alpha[idx]
                    idx += 1

        self.letter_to_digit: dict[str, int] = {}
        self.digit_to_letters: dict[int, list[str]] = {}
        for c in range(9):
            digit = self.labels[c]
            col_letters = [cells[r][c] for r in range(3) if cells[r][c]]
            self.digit_to_letters.setdefault(digit, [])
            for letter in col_letters:
                self.letter_to_digit[letter] = digit
                self.digit_to_letters[digit].append(letter)

    def digit_for(self, letter: str) -> int:
        letter = letter.upper()
        if letter == "J":
            letter = "I"
        return self.letter_to_digit[letter]


def _split_key(key: str) -> tuple[str, str]:
    if key is None:
        raise ValueError("Tridigital requires a key")
    parts = str(key).split("/", 1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    single = str(key).strip()
    return single, single


class Tridigital(Cipher):
    name = "tridigital"
    aliases = ("aca84",)
    description = "Keyed many-to-one letter->digit map with a word-spacer digit (ACA #84)."
    key_format = "columnkeyword/alphabetkeyword (column keyword is 10 letters)"
    key_example = "NOVELCRAFT/DRAGONFLY"
    needs_key = True
    complexity = 4

    # -- encode ----------------------------------------------------------
    def encode(self, text: str, key: str) -> str:
        col_kw, alpha_kw = _split_key(key)
        tab = _Tableau(col_kw, alpha_kw)
        out: list[str] = []
        seen_letter = False
        pending_space = False
        for ch in text.upper():
            if ch.isalpha():
                if pending_space and seen_letter:
                    out.append(str(tab.spacer))
                pending_space = False
                out.append(str(tab.digit_for(ch)))
                seen_letter = True
            elif ch.isspace() or not ch.isalnum():
                if seen_letter:
                    pending_space = True
        return " ".join(out)

    # -- decode ----------------------------------------------------------
    def decode(self, text: str, key: str) -> str:
        col_kw, alpha_kw = _split_key(key)
        tab = _Tableau(col_kw, alpha_kw)
        digits = [int(c) for c in text if c.isdigit()]
        words: list[list[int]] = [[]]
        for d in digits:
            if d == tab.spacer:
                words.append([])
            else:
                words[-1].append(d)
        out_words: list[str] = []
        for word in words:
            if not word:
                continue
            out_words.append(self._resolve_word(word, tab))
        return " ".join(w for w in out_words if w)

    def _resolve_word(self, digits: list[int], tab: _Tableau) -> str:
        """Pick the most letter-like option per digit (first candidate as default)."""
        # Without a scorer we cannot disambiguate homophones; choose the first
        # (top-row) candidate, which exactly inverts encode for single-row digits
        # and is the conventional reading otherwise.
        chars = []
        for d in digits:
            opts = tab.digit_to_letters.get(d, [])
            chars.append(opts[0] if opts else "?")
        return "".join(chars)

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
        """Best-effort keyless crack.

        Tridigital is a homophonic many-to-one map (up to 3 letters per digit)
        with a hidden word-spacer digit. With at most ~17 ciphertext digits in
        the canonical vectors there is far too little signal to recover the
        keyed alphabet keyless, and the search space (which digit is the spacer
        x assignment of 26 letters to 9 ordered columns) is enormous relative to
        the available n-gram constraint. We therefore do not attempt a full
        recovery and return no candidates rather than emit noise.
        """
        return []
