"""Interrupted Key cipher — a periodic substitution whose keyword is *reset*.

First described in the Oct 1935 issue of the ACA's *The Cryptogram*. It is an
ordinary periodic cipher (Vigenere, Beaufort, Variant Beaufort or Porta) except
that the pointer into the keyword is restarted from the beginning at certain
points in the message instead of cycling continuously. The classic triggers are
word divisions, "after a chosen letter", or random; the ACA only requires that
the full keyword appear at least once.

Because ``encode``/``decode`` operate on a *clean* letter stream (no spaces),
the interruption rule must be carried by the key. Two interruption modes are
supported:

KEY format
----------
``"<keyword>"``
    No interruption — a plain periodic cipher (the keyword just repeats).

``"<keyword>/I=<letter>"``
    *Interruptor* mode (the most general ACA form): the keyword pointer is
    restarted to position 0 immediately **after** every plaintext letter equal
    to ``<letter>`` is enciphered. (Decryption restarts after the recovered
    plaintext letter, so the chain stays self-consistent.) Example::

        "TWAIN/I=E"

``"<keyword>/G=<n>,<n>,...""``
    *Group* mode: the message is split into consecutive groups of the given
    lengths and the keyword is restarted at the head of each group. This
    reproduces the word-division convention on a clean (space-free) stream —
    the group lengths are exactly the word lengths. A trailing group is implied
    if the lengths sum to less than the text (the final group runs to the end).
    Example (the CryptoCrack worked example, word lengths of "If you tell the
    truth you dont have to remember anything")::

        "TWAIN/G=2,3,4,3,5,3,4,4,2,8,8"

An optional family prefix selects the underlying table (default Vigenere)::

    "VIG:TWAIN/G=2,3,4"     "BEAU:KEY/I=E"     "PORTA:KEY"

Accepted family tokens: ``VIG``/``VIGENERE``, ``BEAU``/``BEAUFORT``,
``VAR``/``VARBEAUFORT``/``VARIANT``, ``PORTA``.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

# Per-family letter equations on 0-25 ints. ``enc(shift, p) -> c``,
# ``dec(shift, c) -> p`` (mod 26 applied by the caller). Porta shifts are the
# table index 0-12 selected by the key letter's pair.
LetterFn = Callable[[int, int], int]


def _porta(t: int, x: int) -> int:
    if x < 13:
        return 13 + (x + t) % 13
    return (x - 13 - t) % 13


# family -> (enc, dec, shift_of_key_letter, period_of_allowed_shifts)
_FAMILIES: dict[str, tuple[LetterFn, LetterFn, Callable[[int], int], int]] = {
    "VIG": (lambda s, p: p + s, lambda s, c: c - s, lambda k: k, 26),
    "BEAU": (lambda s, p: s - p, lambda s, c: s - c, lambda k: k, 26),
    "VAR": (lambda s, p: p - s, lambda s, c: c + s, lambda k: k, 26),
    "PORTA": (_porta, _porta, lambda k: k // 2, 13),
}

_FAMILY_ALIASES = {
    "VIG": "VIG",
    "VIGENERE": "VIG",
    "BEAU": "BEAU",
    "BEAUFORT": "BEAU",
    "VAR": "VAR",
    "VARIANT": "VAR",
    "VARBEAUFORT": "VAR",
    "VARIANTBEAUFORT": "VAR",
    "PORTA": "PORTA",
}


class _ParsedKey:
    """Resolved key: a family, a keyword's per-position shifts, and a reset rule."""

    __slots__ = ("family", "shifts", "interruptor", "groups")

    def __init__(
        self,
        family: str,
        shifts: list[int],
        interruptor: int | None,
        groups: list[int] | None,
    ) -> None:
        self.family = family
        self.shifts = shifts
        self.interruptor = interruptor  # 0-25 plaintext letter, or None
        self.groups = groups  # explicit group lengths, or None


def _parse_key(key: str) -> _ParsedKey:
    raw = str(key).strip()
    family = "VIG"
    if ":" in raw:
        fam_tok, raw = raw.split(":", 1)
        fam = fam_tok.strip().upper()
        if fam not in _FAMILY_ALIASES:
            raise ValueError(f"interrupted-key: unknown family {fam_tok!r}")
        family = _FAMILY_ALIASES[fam]
        raw = raw.strip()

    interruptor: int | None = None
    groups: list[int] | None = None
    if "/" in raw:
        keyword_part, rule = raw.split("/", 1)
        rule = rule.strip()
        upper = rule.upper()
        if upper.startswith("I="):
            letters = only_letters(rule[2:])
            if len(letters) != 1:
                raise ValueError("interrupted-key: I= needs exactly one letter (e.g. /I=E)")
            interruptor = ord(letters) - 65
        elif upper.startswith("G="):
            spec = rule[2:].strip()
            try:
                groups = [int(x) for x in spec.replace(" ", ",").split(",") if x != ""]
            except ValueError as exc:
                raise ValueError(
                    "interrupted-key: G= needs integer lengths (e.g. /G=2,3,4)"
                ) from exc
            if not groups or any(g <= 0 for g in groups):
                raise ValueError("interrupted-key: G= group lengths must be positive integers")
        else:
            raise ValueError("interrupted-key: rule after '/' must be 'I=<letter>' or 'G=...'")
    else:
        keyword_part = raw

    _enc, _dec, shift_of, _period = _FAMILIES[family]
    kw = only_letters(keyword_part)
    if not kw:
        raise ValueError("interrupted-key: keyword must contain letters")
    shifts = [shift_of(ord(ch) - 65) for ch in kw]
    return _ParsedKey(family, shifts, interruptor, groups)


def _reset_positions(letters: str, parsed: _ParsedKey) -> set[int]:
    """Indices in ``letters`` at which the keyword pointer is reset to 0.

    For group mode these are the group heads; for plain mode only index 0.
    (Interruptor mode is handled inline because it depends on the plaintext
    letter at each step, which differs between encode and decode.)
    """
    heads = {0}
    if parsed.groups is not None:
        pos = 0
        for g in parsed.groups:
            pos += g
            if pos < len(letters):
                heads.add(pos)
    return heads


def _encode_letters(letters: str, parsed: _ParsedKey) -> str:
    enc, _dec, _so, _p = _FAMILIES[parsed.family]
    shifts = parsed.shifts
    n = len(shifts)
    heads = _reset_positions(letters, parsed)
    out: list[str] = []
    j = 0
    for i, ch in enumerate(letters):
        if i in heads:
            j = 0
        p = ord(ch) - 65
        out.append(chr(enc(shifts[j % n], p) % 26 + 65))
        # Interruptor restarts the pointer *after* enciphering the trigger letter.
        if parsed.interruptor is not None and p == parsed.interruptor:
            j = 0
        else:
            j += 1
    return "".join(out)


def _decode_letters(letters: str, parsed: _ParsedKey) -> str:
    _enc, dec, _so, _p = _FAMILIES[parsed.family]
    shifts = parsed.shifts
    n = len(shifts)
    heads = _reset_positions(letters, parsed)
    out: list[str] = []
    j = 0
    for i, ch in enumerate(letters):
        if i in heads:
            j = 0
        c = ord(ch) - 65
        p = dec(shifts[j % n], c) % 26
        out.append(chr(p + 65))
        # Mirror the encoder: restart after recovering the trigger plaintext letter.
        if parsed.interruptor is not None and p == parsed.interruptor:
            j = 0
        else:
            j += 1
    return "".join(out)


class InterruptedKey(Cipher):
    name = "interrupted-key"
    aliases = ("intkey", "interrupted")
    description = (
        "Periodic cipher (Vigenere/Beaufort/Porta) whose keyword pointer is reset "
        "at word breaks, after a chosen letter, or at given group boundaries."
    )
    key_format = "keyword, optionally /I=<letter> or /G=<lens>, optional FAMILY: prefix"
    key_example = "TWAIN/I=E"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        parsed = _parse_key(key)
        return reflow(text, _encode_letters(only_letters(text), parsed))

    def decode(self, text: str, key: str) -> str:
        parsed = _parse_key(key)
        return reflow(text, _decode_letters(only_letters(text), parsed))

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
        """Best-effort: dictionary-style keyword search in *interruptor* mode.

        The interruptor variant has no period to leverage, so we sweep a wordlist
        of candidate keywords across all four families and every plausible
        interruptor letter, scoring the full decrypt with the n-gram model. The
        caller may pass ``wordlist=[...]`` (otherwise we have no dictionary and
        cannot crack, returning ``[]``). Group mode is not cracked here because
        its reset pattern is unknown without the original word divisions.
        """
        letters = only_letters(text)
        # `--wordlist FILE` is loaded by the engine into the `keywords` opt; tests
        # may also pass `keywords=[...]` (or the legacy `wordlist=[...]`) directly.
        wordlist = opts.get("keywords") or opts.get("wordlist")
        if len(letters) < 12 or not wordlist:
            return []
        deadline = (time.monotonic() + timeout) if timeout else None

        results: list[tuple[float, str, str]] = []
        # Interruptor letters worth trying: the common English high-frequency
        # restart triggers plus an explicit option to test "no interruption".
        triggers = [None, *(ord(c) - 65 for c in "ETAOINSHR")]
        for fam in ("VIG", "BEAU", "VAR", "PORTA"):
            _enc, _dec, shift_of, _p = _FAMILIES[fam]
            for word in wordlist:
                if deadline and time.monotonic() > deadline:
                    break
                kw = only_letters(str(word))
                if not kw:
                    continue
                shifts = [shift_of(ord(ch) - 65) for ch in kw]
                for trig in triggers:
                    parsed = _ParsedKey(fam, shifts, trig, None)
                    plain = _decode_letters(letters, parsed)
                    s = scorer.score(plain)
                    rule = "" if trig is None else f"/I={chr(trig + 65)}"
                    results.append((s, f"{fam}:{kw}{rule}", plain))
            if deadline and time.monotonic() > deadline:
                break

        results.sort(key=lambda r: r[0], reverse=True)
        seen: set[str] = set()
        out: list[Candidate] = []
        for score, keystr, plain in results:
            if plain in seen:
                continue
            seen.add(plain)
            out.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=keystr,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={},
                )
            )
            if len(out) >= top:
                break
        return out
