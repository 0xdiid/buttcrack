"""Quagmire III: periodic polyalphabetic over a single KEYED alphabet (K3 keying).

The Quagmire family is a periodic polyalphabetic cipher (a Vigenere-style array)
in which one or both alphabets are mixed (keyed). Quagmire III uses the SAME keyed
alphabet for BOTH the plaintext header row and the basis of every cipher row, with
an indicator keyword fixing the period and the per-column rotations.

Construction (following the ACA description)::

    1. Build one keyed alphabet from a keyword: the keyword with repeats removed,
       followed by the unused letters of A-Z in order.
       e.g. AUTOMOBILE -> AUTOMBILECDFGHJKNPQRSVWXYZ.
       This keyed alphabet is the plaintext header row AND the basis of every
       cipher row.
    2. An indicator keyword fixes the period (= its length). Each indicator letter
       generates one column's cipher row: rotate the keyed alphabet so the
       indicator letter lands in the column of an "alignment" letter of the header.
       When no alignment letter is given it defaults to the keyed alphabet's FIRST
       letter (``header[0]``) — i.e. a plain Vigenere in the keyed alphabet
       (indicator letter == first keyed letter is the identity column). This matches
       the solver, which emits Q3 keys as ``KEYWORD/INDICATOR/<first-keyed-letter>``.
       Pass an explicit third field to force a different alignment (e.g. the ACA
       ``A`` setting). For the classic ``AUTOMOBILE/HIGHWAY`` vector ``header[0]``
       is already ``A``, so the default is unchanged there.

Encrypt: for plaintext letter ``P`` in column ``j`` (j cycles 0..period-1), find
the POSITION of ``P`` in the keyed plaintext header, then emit the letter at that
position in column ``j``'s rotated keyed cipher row.

Decrypt: find ``C``'s position in column ``j``'s cipher row, emit the keyed header
letter at that position.

Full 26-letter alphabet, no I/J merge, no padding, period = indicator length, not
reciprocal.

KEY FORMAT (one ``--key`` string, slash-separated)::

    "ALPHABETKEYWORD/INDICATORKEYWORD"            -> alignment = keyed alphabet's first letter
    "ALPHABETKEYWORD/INDICATORKEYWORD/ALIGN"      -> explicit alignment letter ALIGN

The published example (see tests) is keyed alphabet AUTOMOBILE, indicator HIGHWAY
-> key string "AUTOMOBILE/HIGHWAY" (header[0] == A). For a keyword that does not
start with A, e.g. "MONARCHY/SENTINEL", the alignment defaults to M (header[0]).
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def keyed_alphabet(keyword: str) -> str:
    """Keyword with repeats removed, then the unused A-Z letters in order."""
    seen: list[str] = []
    for ch in keyword.upper():
        if "A" <= ch <= "Z" and ch not in seen:
            seen.append(ch)
    for ch in _ALPHABET:
        if ch not in seen:
            seen.append(ch)
    return "".join(seen)


def _parse_key(key: str) -> tuple[str, str, str | None]:
    """Return (alphabet_keyword_letters, indicator_letters, alignment_letter_or_None).

    ``None`` alignment means "default to the keyed alphabet's first letter", which the
    caller resolves once it has built the header (see :func:`_resolve_align`).
    """
    parts = key.split("/")
    if len(parts) < 2:
        raise ValueError("quagmire3 key must be 'ALPHABETKEYWORD/INDICATORKEYWORD[/ALIGN]'")
    alpha_kw = only_letters(parts[0])
    indicator = only_letters(parts[1])
    align_field = only_letters(parts[2]) if len(parts) >= 3 else ""
    align = align_field[0] if align_field else None
    if not alpha_kw:
        raise ValueError("quagmire3 alphabet keyword must contain letters")
    if not indicator:
        raise ValueError("quagmire3 indicator keyword must contain letters")
    return alpha_kw, indicator, align


def _resolve_align(align: str | None, header: str) -> str:
    """Alignment letter, defaulting to the keyed alphabet's first letter (PK convention)."""
    return align if align else header[0]


def _cipher_rows(header: str, indicator: str, align: str) -> list[str]:
    """One rotated keyed cipher row per indicator letter.

    For indicator letter ``L`` the keyed alphabet is rotated so that ``L`` sits in
    the column of the alignment letter, i.e. ``row[hpos[align]] == L``.
    """
    hpos = {c: i for i, c in enumerate(header)}
    align_idx = hpos[align]
    rows: list[str] = []
    for ch in indicator:
        rot = (hpos[ch] - align_idx) % 26
        rows.append(header[rot:] + header[:rot])
    return rows


def _encode_letters(letters: str, header: str, rows: list[str]) -> str:
    hpos = {c: i for i, c in enumerate(header)}
    period = len(rows)
    out = []
    for j, p in enumerate(letters):
        out.append(rows[j % period][hpos[p]])
    return "".join(out)


def _decode_letters(letters: str, header: str, rows: list[str]) -> str:
    period = len(rows)
    row_pos = [{c: i for i, c in enumerate(row)} for row in rows]
    out = []
    for j, c in enumerate(letters):
        out.append(header[row_pos[j % period][c]])
    return "".join(out)


class QuagmireIII(Cipher):
    name = "quagmire3"
    aliases = ("quag3", "quagmire-iii")
    description = (
        "Periodic polyalphabetic over a single keyed alphabet (K3) with an indicator keyword."
    )
    key_format = "alphabet-keyword/indicator-keyword (optional /align letter)"
    key_example = "AUTOMOBILE/HIGHWAY"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        alpha_kw, indicator, align = _parse_key(key)
        header = keyed_alphabet(alpha_kw)
        rows = _cipher_rows(header, indicator, _resolve_align(align, header))
        return _encode_letters(only_letters(text), header, rows)

    def decode(self, text: str, key: str) -> str:
        alpha_kw, indicator, align = _parse_key(key)
        header = keyed_alphabet(alpha_kw)
        rows = _cipher_rows(header, indicator, _resolve_align(align, header))
        return reflow(text, _decode_letters(only_letters(text), header, rows))

    def crack(
        self,
        text,
        scorer: NgramScorer,
        *,
        top=5,
        rng=None,
        timeout: float | None = None,
        **opts,
    ) -> list[Candidate]:
        """Keyless crack: keyword dictionary attack, plus an opt-in blind anneal.

        A Quagmire III is a periodic substitution where every column is a rotation
        of one shared keyed alphabet. The reliable route at ACA lengths is the
        keyword **dictionary attack** (KRYPTOS, ...): a real keyed-alphabet keyword
        is recovered cheaply, then the per-column shifts by quadgram.

        Blind recovery of an *arbitrary* keyed alphabet is opt-in via ``--blind``.
        It is genuinely hard: the correct alphabet is an isolated optimum with
        essentially no gradient — a single swap away from it already scores like a
        random alphabet — so simulated annealing rarely converges at puzzle lengths
        and would otherwise just burn the whole timeout. The dictionary attack or a
        crib (see ``butt crib`` / ``--crib``) are the dependable levers.
        """
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None
        letters = only_letters(text)
        if len(letters) < 40:
            return []

        from . import _quagmire_solver as qs

        candidates: list[Candidate] = qs.dictionary_candidates(
            self, "Q3", text, scorer, deadline=deadline, rng=rng, **opts
        )
        if opts.get("blind"):
            candidates += qs.blind_candidates(
                self, "Q3", text, scorer, deadline=deadline, rng=rng, **opts
            )
        return qs._dedup_by_plaintext(candidates)[:top]
