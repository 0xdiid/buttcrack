"""Quagmire I cipher: periodic polyalphabetic over a KEYED plaintext alphabet.

Quagmire I is the ACA "K1" periodic cipher: a Vigenere-style periodic cipher in
which the PLAINTEXT alphabet is a keyword-mixed (keyed) alphabet while every
ciphertext alphabet is a STRAIGHT (A-Z) alphabet shifted to a column-specific
position. (Contrast: Quag II = straight plaintext vs keyed ciphertext;
Quag III = the same keyed alphabet on both sides; Quag IV = two distinct keyed
alphabets.)

Construction
------------
1. Keyed plaintext alphabet ``KP`` from a keyword: write the keyword dropping
   repeated letters, then the remaining A-Z letters in order. E.g.
   ``SPRINGFEVER`` -> ``SPRINGFEVABCDHJKLMOQTUWXYZ``.
2. An INDICATOR keyword sets the period (= its length) and the per-column
   shifts. The indicator key is placed vertically under an ALIGNMENT letter of
   the keyed plaintext alphabet (conventionally ``A``). For column ``j`` the
   straight ciphertext alphabet is rotated so the indicator letter sits under
   that alignment letter, i.e. ``SA_j[index_in_KP(align)] == indicator_j``.

Encryption (per the ACA description)::

    j  = i mod period                      # column for the i-th letter
    sh = (ord(indicator_j) - ord(align)_in_KP) mod 26
    C  = standardAZ[(index_in_KP(P) + sh) mod 26]

Decryption reverses: find the position of ``C`` in the straight A-Z alphabet,
subtract the shift, index back into ``KP``.

Full 26-letter alphabet (no I/J merge); period = indicator length; no padding.

KEY FORMAT
----------
``"ALPHABETKEYWORD/INDICATOR"`` or, to override the alignment letter,
``"ALPHABETKEYWORD/INDICATOR/ALIGN"`` (ALIGN is a single letter, default ``A``).
The first field builds the keyed plaintext alphabet; the second is the indicator
keyword that sets the period and the per-column shifts. Example:
``"SPRINGFEVER/FLOWER"`` (period 6, aligned under A).
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

from ..result import Candidate
from ..scoring import NgramScorer, index_of_coincidence
from ..text import only_letters, reflow
from .base import Cipher

_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def keyed_alphabet(keyword: str) -> str:
    """Mixed alphabet: keyword (deduped) followed by the unused A-Z letters."""
    out: list[str] = []
    for ch in keyword.upper():
        if "A" <= ch <= "Z" and ch not in out:
            out.append(ch)
    for ch in _STD:
        if ch not in out:
            out.append(ch)
    return "".join(out)


def _parse_key(key: str) -> tuple[str, str, str]:
    """Return ``(keyed_plain_alphabet, indicator, align_letter)``."""
    parts = key.split("/")
    if len(parts) < 2:
        raise ValueError(
            "quagmire1 key must be 'ALPHABETKEYWORD/INDICATOR' "
            "(optionally '/ALIGN'); got " + repr(key)
        )
    alpha_kw = only_letters(parts[0])
    indicator = only_letters(parts[1])
    align = only_letters(parts[2]) if len(parts) >= 3 and only_letters(parts[2]) else "A"
    if not indicator:
        raise ValueError("quagmire1 indicator keyword must contain letters")
    return keyed_alphabet(alpha_kw), indicator, align[0]


def _column_shifts(kp: str, indicator: str, align: str) -> list[int]:
    """Per-column additive shifts derived from the indicator and alignment letter."""
    base = kp.index(align)
    return [(ord(ind) - 65 - base) % 26 for ind in indicator]


class QuagmireI(Cipher):
    name = "quagmire1"
    aliases = ("quag1", "quagmirei")
    description = (
        "Periodic polyalphabetic with a keyed plaintext alphabet against "
        "straight ciphertext alphabets (ACA K1)."
    )
    key_format = "alphabet-keyword/indicator-keyword (optional /align letter)"
    key_example = "SPRINGFEVER/FLOWER"
    complexity = 6

    # -- core transforms -------------------------------------------------
    def encode(self, text: str, key: str) -> str:
        kp, indicator, align = _parse_key(key)
        shifts = _column_shifts(kp, indicator, align)
        kp_index = {c: i for i, c in enumerate(kp)}
        period = len(shifts)
        out = []
        for i, p in enumerate(only_letters(text)):
            sh = shifts[i % period]
            out.append(_STD[(kp_index[p] + sh) % 26])
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        kp, indicator, align = _parse_key(key)
        shifts = _column_shifts(kp, indicator, align)
        period = len(shifts)
        out = []
        for i, c in enumerate(only_letters(text)):
            sh = shifts[i % period]
            out.append(kp[(ord(c) - 65 - sh) % 26])
        return "".join(out)

    # -- best-effort keyless crack --------------------------------------
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
        """Best-effort keyless recovery.

        Because every column shares the SAME keyed plaintext alphabet and differs
        only by an additive shift, the cipher's strength collapses: align the
        columns by their relative shifts (chi-squared per column) so all columns
        fold onto ONE monoalphabetic stream, then solve that monoalphabet with the
        standard quadgram swap-climb over the full text length. The period is
        chosen by the per-column IoC (each column is a monoalphabet, so columns
        are English-like only at the true period).
        """
        letters = only_letters(text)
        if len(letters) < 60:
            return []
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        from . import _quagmire_solver as qs

        candidates: list[Candidate] = []

        # Column-folding solve FIRST: it is the precise Q1 method (it recovers the
        # keyed alphabet exactly, for word AND non-word keys) and finishes in
        # seconds, so it must not be starved by the dictionary attack's budget.
        forced = opts.get("key_length") or opts.get("period")
        max_len = int(opts.get("max_period", min(12, len(letters) // 12)))
        periods = [int(forced)] if forced else list(range(1, max(2, max_len) + 1))

        best: tuple[float, str, list[int], list[str], int] | None = None
        for period in periods:
            if period < 1:
                continue
            if deadline and time.monotonic() > deadline:
                break
            cand = self._crack_period(letters, scorer, period, rng, deadline, opts)
            if cand is None:
                continue
            if best is None or cand[0] > best[0]:
                best = cand

        if best is not None:
            score, plain, shifts, dec, period = best
            # dec[k] = plaintext letter for collapsed-index k. Rebuild the keyed
            # plaintext alphabet KP: KP[(ord(c)-65 - shift) % 26] == plaintext, i.e.
            # KP[k] = dec[k]. Then derive the indicator (align = A) from the shifts.
            kp = "".join(dec)
            base = kp.index("A") if "A" in kp else 0
            indicator = "".join(chr((base + s) % 26 + 65) for s in shifts)
            keyword = self._kp_to_keyword(kp)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=f"{keyword}/{indicator}",
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"period": period, "keyed_alphabet": kp, "indicator": indicator},
                )
            )

        # Keyword dictionary attack with the remaining budget — adds the long-key
        # case (famous keyed alphabet, key too long for column-folding to align).
        candidates += qs.dictionary_candidates(
            self, "Q1", text, scorer, deadline=deadline, rng=rng, **opts
        )

        return qs._dedup_by_plaintext(candidates)[:top]

    def _crack_period(
        self,
        letters: str,
        scorer: NgramScorer,
        period: int,
        rng: random.Random,
        deadline: float | None,
        opts,
    ) -> tuple[float, str, list[int], list[str], int] | None:
        cols = [letters[j::period] for j in range(period)]
        if any(len(c) < 3 for c in cols):
            return None
        # Reject periods whose per-column IoC is clearly non-English (random ~ .038,
        # English ~ .066). For period 1 this is just the whole-text IoC.
        avg_ioc = sum(index_of_coincidence(c) for c in cols) / period
        if period > 1 and avg_ioc < 0.052:
            return None

        # Fold the columns onto a single monoalphabetic stream by aligning each
        # column to column 0 by monogram cross-correlation. Because the plaintext
        # alphabet is keyed (not the identity), per-column chi-squared against
        # English gives wrong shifts; aligning columns to EACH OTHER recovers the
        # correct RELATIVE shifts. The common offset is absorbed by the solved
        # monoalphabet, so the resulting decrypt map IS the keyed alphabet.
        shifts = self._seed_shifts(cols)
        collapsed = self._collapse(letters, shifts, period)

        restarts = int(opts.get("restarts", 6))
        dec, score = self._solve_mono(collapsed, scorer, rng, restarts, deadline)
        plain = "".join(dec[ord(c) - 65] for c in collapsed)
        return score, plain, shifts, dec, period

    @staticmethod
    def _seed_shifts(cols: Sequence[str]) -> list[int]:
        """Relative additive shifts that align every column to column 0.

        For each column, pick the shift that maximizes the cross-correlation of
        its monogram counts with column 0's. This recovers the column-to-column
        shift differences (the indicator differences) without needing to know the
        keyed alphabet; the absolute offset is left for the monoalphabet solve.
        """
        from collections import Counter

        def counts(col: str) -> list[int]:
            c = Counter(col)
            return [c.get(chr(65 + i), 0) for i in range(26)]

        ref = counts(cols[0])
        shifts: list[int] = []
        for col in cols:
            cc = counts(col)
            best_s, best_dot = 0, -1.0
            for s in range(26):
                dot = sum(ref[i] * cc[(i + s) % 26] for i in range(26))
                if dot > best_dot:
                    best_dot, best_s = float(dot), s
            shifts.append(best_s)
        return shifts

    @staticmethod
    def _collapse(letters: str, shifts: Sequence[int], period: int) -> str:
        """Subtract per-column shifts so all columns share one monoalphabet."""
        return "".join(
            chr((ord(c) - 65 - shifts[i % period]) % 26 + 65) for i, c in enumerate(letters)
        )

    def _solve_mono(
        self,
        collapsed: str,
        scorer: NgramScorer,
        rng: random.Random,
        restarts: int,
        deadline: float | None,
    ) -> tuple[list[str], float]:
        """Quadgram swap-climb a monoalphabetic substitution (dec[cipher]=plain)."""
        best_dec = self._freq_seed(collapsed)
        best_score = scorer.score("".join(best_dec[ord(c) - 65] for c in collapsed))
        for r in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            if r == 0:
                parent = self._freq_seed(collapsed)
            else:
                parent = [chr(65 + i) for i in range(26)]
                rng.shuffle(parent)
            pscore = scorer.score("".join(parent[ord(c) - 65] for c in collapsed))
            improved = True
            while improved:
                improved = False
                for i in range(25):
                    if deadline and time.monotonic() > deadline:
                        improved = False
                        break
                    for j in range(i + 1, 26):
                        child = parent[:]
                        child[i], child[j] = child[j], child[i]
                        s = scorer.score("".join(child[ord(c) - 65] for c in collapsed))
                        if s > pscore:
                            pscore, parent, improved = s, child, True
            if pscore > best_score:
                best_score, best_dec = pscore, parent
        return best_dec, best_score

    @staticmethod
    def _freq_seed(letters: str) -> list[str]:
        from collections import Counter

        order = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
        counts = Counter(letters)
        cipher_by_freq = sorted((chr(65 + i) for i in range(26)), key=lambda c: -counts.get(c, 0))
        dec = ["A"] * 26
        for cipher_letter, plain_letter in zip(cipher_by_freq, order, strict=True):
            dec[ord(cipher_letter) - 65] = plain_letter
        return dec

    @staticmethod
    def _kp_to_keyword(kp: str) -> str:
        """Best-effort keyword prefix of a recovered keyed alphabet.

        Strips the trailing run already in ascending A-Z order, leaving the
        keyword prefix (a heuristic; the full keyed alphabet is in meta).
        """
        n = len(kp)
        cut = n
        for i in range(n - 1, 0, -1):
            if kp[i] > kp[i - 1]:
                cut = i
            else:
                break
        return kp[:cut] if cut < n else kp
