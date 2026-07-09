"""Quagmire IV cipher: periodic polyalphabetic with two keyed alphabets.

Quagmire IV (ACA "K4" keying) is the most general of the four Quagmire
ciphers. It uses THREE keywords:

  * a keyword giving a keyed PLAINTEXT alphabet,
  * a *different* keyword giving a keyed CIPHERTEXT alphabet, and
  * an indicator keyword that sets the period and each column's rotation.

(Quagmire I uses a keyed plaintext alphabet vs a straight ciphertext alphabet;
Quagmire II the reverse; Quagmire III the same keyed alphabet on both sides;
Quagmire IV two different keyed alphabets.)

Tableau construction
--------------------
Build each keyed alphabet by writing the (de-duplicated) keyword first, then the
remaining letters of A-Z in order. No I/J merge; the full 26 letters are used.

The indicator keyword is written under a chosen "alignment" letter of the keyed
plaintext alphabet (ACA default: the FIRST letter of the keyed plaintext
alphabet, i.e. position 0). For each indicator letter, the keyed ciphertext
alphabet is rotated so that the indicator letter falls at the alignment letter's
position in the keyed plaintext alphabet; that rotated sequence is the column's
cipher row. With the default alignment (position 0) this is simply: the column's
cipher row is the keyed ciphertext alphabet rotated to *start* at the indicator
letter.

Encrypt
-------
For plaintext letter ``P`` in column ``j``: take ``P``'s index in the keyed
plaintext alphabet, and emit the letter at that same index in column ``j``'s
cipher row.

Decrypt
-------
For ciphertext letter ``C`` in column ``j``: find ``C``'s index in column
``j``'s cipher row, and emit the keyed-plaintext-alphabet letter at that index.

The transform is periodic with period = indicator length, operates on a clean
uppercase A-Z stream (non-letters dropped on encode), uses no padding, and is
not reciprocal.

KEY FORMAT
----------
``"PTKEY/CTKEY/INDICATOR"`` -- three keywords separated by ``/``:

  * ``PTKEY``     keyword for the keyed plaintext alphabet,
  * ``CTKEY``     keyword for the keyed ciphertext alphabet,
  * ``INDICATOR`` indicator keyword (its length is the period).

An optional fourth field ``"PTKEY/CTKEY/INDICATOR/X"`` overrides the alignment
letter ``X`` (which must occur in the keyed plaintext alphabet); when omitted the
ACA default is used, i.e. the first letter of the keyed plaintext alphabet.

Example (ACA worked example):
    encode("THISONEEMPLOYSTHREEKEYWORDS", "SENSORY/PERCEPTION/EXTRA")
    -> "VBMRFCYISPMPBRRHEICXRREIGDX"
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from ._periodic import columns
from .base import Cipher

_AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def keyed_alphabet(keyword: str) -> str:
    """Keyed 26-letter alphabet: de-duped keyword then the remaining letters."""
    seen: list[str] = []
    for ch in keyword.upper():
        if ch.isalpha() and ch not in seen:
            seen.append(ch)
    for ch in _AZ:
        if ch not in seen:
            seen.append(ch)
    return "".join(seen)


def _parse_key(key: str) -> tuple[str, str, str, str]:
    """Return (pt_alphabet, ct_alphabet, indicator, align_letter)."""
    parts = key.split("/")
    if len(parts) < 3:
        raise ValueError(
            "quagmire4 key must be 'PTKEY/CTKEY/INDICATOR' "
            "(optional 4th field overrides the alignment letter)"
        )
    pt = keyed_alphabet(only_letters(parts[0]))
    ct = keyed_alphabet(only_letters(parts[1]))
    indicator = only_letters(parts[2]).upper()
    if not indicator:
        raise ValueError("quagmire4 indicator keyword must contain letters")
    align = only_letters(parts[3]).upper()[:1] if len(parts) >= 4 and parts[3].strip() else pt[0]
    if align not in pt:
        raise ValueError("quagmire4 alignment letter must occur in the plaintext alphabet")
    return pt, ct, indicator, align


def _cipher_rows(ct: str, indicator: str, align_pos: int) -> list[str]:
    """One rotated keyed-cipher row per indicator letter.

    The indicator letter is placed at ``align_pos`` within the row, so the row is
    ``ct`` rotated left by ``(ct.index(ind) - align_pos) % 26``.
    """
    rows: list[str] = []
    for ind in indicator:
        r = (ct.index(ind) - align_pos) % 26
        rows.append(ct[r:] + ct[:r])
    return rows


def _encode_letters(letters: str, pt: str, rows: list[str]) -> str:
    period = len(rows)
    out = []
    for j, p in enumerate(letters):
        out.append(rows[j % period][pt.index(p)])
    return "".join(out)


def _decode_letters(letters: str, pt: str, rows: list[str]) -> str:
    period = len(rows)
    out = []
    for j, c in enumerate(letters):
        out.append(pt[rows[j % period].index(c)])
    return "".join(out)


class QuagmireIV(Cipher):
    name = "quagmire4"
    aliases = ("quag4", "quagmireiv")
    description = (
        "Periodic polyalphabetic with a keyed plaintext alphabet, a different "
        "keyed ciphertext alphabet, and an indicator keyword (ACA K4)."
    )
    key_format = "pt-keyword/ct-keyword/indicator-keyword (optional /align letter)"
    key_example = "SENSORY/PERCEPTION/EXTRA"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        pt, ct, indicator, align = _parse_key(key)
        rows = _cipher_rows(ct, indicator, pt.index(align))
        return _encode_letters(only_letters(text).upper(), pt, rows)

    def decode(self, text: str, key: str) -> str:
        pt, ct, indicator, align = _parse_key(key)
        rows = _cipher_rows(ct, indicator, pt.index(align))
        return _decode_letters(only_letters(text).upper(), pt, rows)

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
        """Best-effort keyless recovery (period detection + alphabet hill-climb).

        Strategy: for each candidate period, collapse each period column onto a
        single composite substitution alphabet. With the default alignment, a
        Quagmire IV column maps plaintext index ``k`` to ``ct_row[k]`` where the
        plaintext alphabet supplies the index -- so every column is a
        monoalphabetic substitution that is the same composite permutation
        rotated per column. We hill-climb a single 26-letter composite map
        (plaintext-index -> output letter for the reference column) against the
        quadgram score of the whole text, deriving the per-column rotations from
        the relative column offsets. This recovers the *plaintext* even though it
        cannot disentangle the two keyword alphabets, so the reported key is the
        recovered composite tableau rather than the original keywords.

        These keyed-alphabet ciphers are genuinely hard keyless and need a lot of
        ciphertext; on short inputs this often fails. Returns ``[]`` when no
        period is found or text is too short.
        """
        letters = only_letters(text).upper()
        if len(letters) < 40:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        max_len = int(opts.get("max_key_length", min(15, len(letters) // 8)))
        forced = opts.get("key_length")
        periods: Sequence[int] = [int(forced)] if forced else range(2, max_len + 1)

        best: tuple[float, str, int] | None = None  # (score, plaintext, period)
        for period in periods:
            if deadline and time.monotonic() > deadline:
                break
            res = self._crack_period(letters, period, scorer, rng, deadline)
            if res is None:
                continue
            score, plain = res
            if best is None or score > best[0]:
                best = (score, plain, period)

        if best is None:
            return []
        score, plain, period = best
        return [
            Candidate(
                plaintext=reflow(text, plain),
                cipher=self.name,
                key=None,
                score=score,
                confidence=scorer.confidence(plain),
                meta={"key_length": period, "note": "composite tableau (keywords not recovered)"},
            )
        ]

    # -- crack internals -------------------------------------------------
    def _crack_period(
        self,
        letters: str,
        period: int,
        scorer: NgramScorer,
        rng: random.Random,
        deadline: float | None,
    ) -> tuple[float, str] | None:
        """Hill-climb a composite tableau for one period; return (score, plain).

        Each column is decrypted with a 26-letter ``dec`` map (ciphertext letter
        -> plaintext letter) that shares one composite permutation across columns
        differing only by a per-column rotation of the ciphertext input. We solve
        the composite permutation and the per-column rotations jointly by a swap
        hill-climb over the composite map plus a rotation sweep.
        """
        cols = columns(letters, period)

        # Per-column rotation offsets (relative); column 0 fixed at 0. We let the
        # hill-climb choose all column rotations and a shared 26-letter map.
        rotations = [0] * period
        # Composite decrypt map: index by (ciphertext_index + rotation) % 26 ->
        # plaintext letter. Seed identity.
        comp = list(_AZ)

        def decrypt() -> str:
            out_cols: list[list[str]] = [[] for _ in range(period)]
            for j, col in enumerate(cols):
                rot = rotations[j]
                for ch in col:
                    out_cols[j].append(comp[(ord(ch) - 65 + rot) % 26])
            # reassemble in original order
            out = []
            idxs = [0] * period
            for i in range(len(letters)):
                j = i % period
                out.append(out_cols[j][idxs[j]])
                idxs[j] += 1
            return "".join(out)

        best_score = scorer.score(decrypt())
        improved = True
        sweeps = 0
        while improved:
            improved = False
            sweeps += 1
            if sweeps > 40 or (deadline and time.monotonic() > deadline):
                break
            # 1) optimise per-column rotations
            for j in range(period):
                if deadline and time.monotonic() > deadline:
                    return (best_score, decrypt())
                cur = rotations[j]
                local_best = cur
                for r in range(26):
                    if r == cur:
                        continue
                    rotations[j] = r
                    s = scorer.score(decrypt())
                    if s > best_score:
                        best_score, local_best, improved = s, r, True
                rotations[j] = local_best
            # 2) optimise composite map by swaps
            for a in range(25):
                if deadline and time.monotonic() > deadline:
                    return (best_score, decrypt())
                for b in range(a + 1, 26):
                    comp[a], comp[b] = comp[b], comp[a]
                    s = scorer.score(decrypt())
                    if s > best_score:
                        best_score, improved = s, True
                    else:
                        comp[a], comp[b] = comp[b], comp[a]
        return (best_score, decrypt())
