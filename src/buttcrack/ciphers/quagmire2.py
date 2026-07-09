"""Quagmire II cipher: ACA periodic polyalphabetic, K2 keying.

A *straight* (A-Z) plaintext alphabet is run against a *keyed* ciphertext
alphabet. The keyed ciphertext alphabet is built from a keyword (the keyword's
letters with repeats removed, then the remaining A-Z letters in order), e.g.

    SPRINGFEVER -> SPRINGFEVABCDHJKLMOQTUWXYZ

An indicator keyword sets the period and the per-column rotation. For each
indicator letter the keyed ciphertext alphabet is written cyclically so that the
indicator letter sits under the *alignment* plaintext letter (usually A). With
indicator ``FLOWER`` aligned under A the six ciphertext rows are::

    F: FEVABCDHJKLMOQTUWXYZSPRING
    L: LMOQTUWXYZSPRINGFEVABCDHJK
    O: OQTUWXYZSPRINGFEVABCDHJKLM
    W: WXYZSPRINGFEVABCDHJKLMOQTU
    E: EVABCDHJKLMOQTUWXYZSPRINGF
    R: RINGFEVABCDHJKLMOQTUWXYZSP

ENCRYPT: for plaintext letter ``P`` in column ``j`` (j = position mod period),
take ``P``'s ordinal in the straight alphabet and read the letter at that
position in column ``j``'s rotated keyed alphabet::

    C = row_j[ord(P) - 65]

DECRYPT inverts this: find ``C``'s index in column ``j``'s keyed alphabet and
emit the straight-alphabet letter at that index.

Full 26-letter alphabet (no I/J merge), period = indicator length, no padding.
The cipher is not reciprocal. It is the mirror image of Quagmire I (which keys
the plaintext alphabet against a straight ciphertext alphabet).

KEY FORMAT
----------
Pack both keywords into one ``--key`` separated by ``/``::

    "ALPHABETKEY/INDICATORKEY"          # alignment letter defaults to A
    "ALPHABETKEY/INDICATORKEY/ALIGN"    # explicit alignment letter

``ALPHABETKEY`` builds the keyed ciphertext alphabet; ``INDICATORKEY`` sets the
period and the rotations; ``ALIGN`` is the plaintext letter the indicator
letters are written under (the ACA default is ``A``).

Example: ``"SPRINGFEVER/FLOWER"`` reproduces the published ACA vector.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def keyed_alphabet(keyword: str) -> str:
    """Keyword letters (repeats removed) followed by the unused A-Z letters."""
    seen: list[str] = []
    for c in keyword.upper():
        if "A" <= c <= "Z" and c not in seen:
            seen.append(c)
    for c in _ALPHABET:
        if c not in seen:
            seen.append(c)
    return "".join(seen)


def _parse_key(key: str) -> tuple[str, str, str]:
    """Split ``ALPHABET/INDICATOR[/ALIGN]`` into (keyed_alphabet, indicator, align)."""
    parts = key.split("/")
    if len(parts) < 2:
        raise ValueError("quagmire2 key must be 'ALPHABETKEY/INDICATORKEY' (optionally '/ALIGN')")
    alpha_kw = only_letters(parts[0])
    indicator = only_letters(parts[1])
    align = only_letters(parts[2]) if len(parts) >= 3 and only_letters(parts[2]) else "A"
    if not indicator:
        raise ValueError("quagmire2 indicator keyword must contain letters")
    return keyed_alphabet(alpha_kw), indicator, align[0]


def _rows(keyed: str, indicator: str, align: str) -> list[str]:
    """One rotated keyed ciphertext alphabet per indicator letter.

    Each row is the keyed ciphertext alphabet written cyclically so the indicator
    letter sits under the ``align`` plaintext letter. ``align`` indexes the
    STRAIGHT plaintext alphabet (A=0, B=1, ...), so the indicator letter must
    land at row position ``ord(align) - 65``; the row therefore starts
    ``(index(indicator) - (ord(align) - 65)) % 26`` letters into the keyed
    alphabet. With the usual ``align == 'A'`` (position 0) that is simply
    "rotate the keyed alphabet to start at the indicator letter".
    """
    base = ord(align) - 65
    rows = []
    for ind in indicator:
        start = (keyed.index(ind) - base) % 26
        rows.append(keyed[start:] + keyed[:start])
    return rows


class QuagmireII(Cipher):
    name = "quagmire2"
    aliases = ("quag2", "quagmireii")
    description = (
        "Periodic polyalphabetic (ACA K2): straight plaintext vs keyed ciphertext alphabet."
    )
    key_format = "alphabet-keyword/indicator-keyword (optional /align letter)"
    key_example = "SPRINGFEVER/FLOWER"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        keyed, indicator, align = _parse_key(key)
        rows = _rows(keyed, indicator, align)
        period = len(rows)
        out = []
        for i, p in enumerate(only_letters(text)):
            out.append(rows[i % period][ord(p) - 65])
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        keyed, indicator, align = _parse_key(key)
        rows = _rows(keyed, indicator, align)
        period = len(rows)
        # Per-column lookup tables: ciphertext letter -> straight-alphabet index.
        inverse = [{c: idx for idx, c in enumerate(row)} for row in rows]
        out = []
        for i, c in enumerate(only_letters(text)):
            out.append(chr(65 + inverse[i % period][c]))
        return "".join(out)

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
        """Keyless recovery: keyword dictionary attack, plus an opt-in blind anneal.

        Every column shares the SAME keyed ciphertext alphabet, differing only by
        rotation, so the cracker searches a single 26-letter keyed alphabet. The
        reliable route at ACA lengths is the keyword **dictionary attack** (a real
        keyed-alphabet keyword is recovered cheaply, then the per-column shifts by
        quadgram). Blind recovery of an *arbitrary* keyed alphabet is opt-in via
        ``--blind``: the alphabet is an isolated optimum with essentially no gradient
        (one swap from the true alphabet scores like a random one), so simulated
        annealing rarely converges at puzzle lengths and would otherwise just burn
        the whole timeout. The dictionary attack or a crib are the dependable levers.
        """
        letters = only_letters(text)
        if len(letters) < 40:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        from . import _quagmire_solver as qs

        candidates: list[Candidate] = qs.dictionary_candidates(
            self, "Q2", text, scorer, deadline=deadline, rng=rng, **opts
        )
        if opts.get("blind"):
            candidates += qs.blind_candidates(
                self, "Q2", text, scorer, deadline=deadline, rng=rng, **opts
            )
        return qs._dedup_by_plaintext(candidates)[:top]
