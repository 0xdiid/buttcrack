"""Progressive Key cipher (ACA / CryptoCrack).

A double-enciphered periodic polyalphabetic cipher. The message is first
enciphered with one of the periodic ciphers (Vigenere, Beaufort, Variant
Beaufort or Porta) using a *keyword*, then each successive *group* of letters
(group size = keyword length) is re-enciphered with the same base cipher using a
key letter that "progresses" by a fixed amount per group.

KEY format
----------
``"<keyword>/<progression>"`` or ``"<keyword>/<progression>/<base>"``.

* ``keyword`` -- the periodic keyword for the first encipherment. Its length
  defines the group size for the progression.
* ``progression`` -- a non-negative integer (the ACA "progression index"). Group
  ``g`` (0-based) is re-enciphered with the additional shift ``(progression * g)
  mod 26``. So with progression 3 the groups use key letters A, D, G, J, ...
* ``base`` (optional) -- one of ``vigenere`` (default), ``beaufort``,
  ``variant`` (variant Beaufort) or ``porta``. Selects the letter equation used
  for *both* the keyword layer and the progression layer.

For example, with keyword ``POLITICS``, progression ``3`` and a Vigenere base,
the second 8-letter group is Vigenere-enciphered with the keyword and then
shifted by ``D`` (=3), the third group by ``G`` (=6), and so on.

ENCRYPT
-------
For plaintext letter ``p`` at index ``i`` (group ``g = i // L``)::

    c = base_enc((progression * g) mod 26, base_enc(key[i mod L], p))

DECRYPT reverses both layers in the opposite order. Encode and decode are not
reciprocal in general (they are for the Beaufort / Porta bases, as for the
underlying periodic ciphers).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters, reflow
from .base import Cipher

# (key_value, letter_index) -> result_index; callers apply % 26.
LetterFn = Callable[[int, int], int]


def _vig_enc(k: int, p: int) -> int:
    return p + k


def _vig_dec(k: int, c: int) -> int:
    return c - k


def _beau(k: int, x: int) -> int:
    # Reciprocal Beaufort: enc == dec.
    return k - x


def _var_enc(k: int, p: int) -> int:
    return p - k


def _var_dec(k: int, c: int) -> int:
    return c + k


def _porta(t: int, x: int) -> int:
    """Reciprocal Della Porta map for table ``t`` (0-12), letter ``x`` (0-25)."""
    if x < 13:
        return 13 + (x + t) % 13
    return (x - 13 - t) % 13


# base name -> (enc, dec, key-letter -> key-value).
# The key-value maps an A-Z letter to the parameter the enc/dec functions expect
# (a 0-25 shift for the Vigenere family, a 0-12 table index for Porta).
_BASES: dict[str, tuple[LetterFn, LetterFn, Callable[[int], int]]] = {
    "vigenere": (_vig_enc, _vig_dec, lambda x: x),
    "beaufort": (_beau, _beau, lambda x: x),
    "variant": (_var_enc, _var_dec, lambda x: x),
    "porta": (_porta, _porta, lambda x: x // 2),
}

_BASE_ALIASES = {
    "vig": "vigenere",
    "vigenere": "vigenere",
    "beaufort": "beaufort",
    "beau": "beaufort",
    "variant": "variant",
    "variant-beaufort": "variant",
    "varbeaufort": "variant",
    "porta": "porta",
}


def _parse_key(key: str) -> tuple[str, int, str]:
    """Parse ``"keyword/progression[/base]"`` -> (keyword, progression, base)."""
    parts = [p.strip() for p in str(key).split("/")]
    if len(parts) < 2:
        raise ValueError("progressive-key key must be 'keyword/progression[/base]'")
    keyword = only_letters(parts[0])
    if not keyword:
        raise ValueError("progressive-key keyword must contain letters")
    try:
        progression = int(parts[1])
    except ValueError as exc:
        raise ValueError("progressive-key progression must be an integer") from exc
    if progression < 0:
        raise ValueError("progressive-key progression must be non-negative")
    base = "vigenere"
    if len(parts) >= 3 and parts[2]:
        name = _BASE_ALIASES.get(parts[2].lower())
        if name is None:
            raise ValueError(f"progressive-key unknown base cipher: {parts[2]!r}")
        base = name
    return keyword, progression, base


def _transform(
    letters: str,
    keyword: str,
    progression: int,
    enc: LetterFn,
    prog_enc: LetterFn,
    key_val: Callable[[int], int],
) -> str:
    """Apply the keyword layer (``enc``) then the progression layer (``prog_enc``).

    For decode the caller swaps in the dec functions and reverses the layer
    order by passing the decrypt of the progression layer as ``enc`` of the
    keyword layer's inverse problem (see :meth:`ProgressiveKey.decode`).
    """
    shifts = [key_val(ord(ch) - 65) for ch in keyword]
    length = len(shifts)
    out: list[str] = []
    for i, ch in enumerate(letters):
        x = ord(ch) - 65
        group = i // length
        prog_shift = (progression * group) % 26
        y = enc(shifts[i % length], x) % 26
        z = prog_enc(prog_shift, y) % 26
        out.append(chr(z + 65))
    return "".join(out)


# English letter log-frequencies (A-Z), for the per-column keyword solve.
_ENG_LOGFREQ = [
    -1.074,
    -2.264,
    -1.703,
    -1.583,
    -0.895,
    -1.891,
    -1.912,
    -1.393,
    -1.135,
    -3.204,
    -2.617,
    -1.481,
    -1.660,
    -1.267,
    -1.126,
    -1.702,
    -3.591,
    -1.335,
    -1.213,
    -1.055,
    -1.556,
    -2.394,
    -2.109,
    -3.155,
    -1.798,
    -3.769,
]


def _period_cic(vals: list[int], period: int) -> float:
    """Mean per-coset IoC (x26) at ``period`` -- high when the columns are peaked."""
    if period < 1 or len(vals) < 2 * period:
        return 0.0
    total = 0.0
    for c in range(period):
        col = vals[c::period]
        m = len(col)
        if m < 2:
            continue
        counts = [0] * 26
        for x in col:
            counts[x] += 1
        total += 26.0 * sum(k * (k - 1) for k in counts) / (m * (m - 1))
    return total / period


def _keyless_recover(
    letters: str,
    scorer: NgramScorer,
    *,
    bases: list[str],
    max_period: int,
    prefilter: int,
    top: int,
) -> list[tuple[float, str, str]]:
    """Recover a Progressive Key with NO keyword hint (score, plaintext, key_repr).

    Key idea: the *progression* is recoverable before the keyword. Undoing the
    progression layer (a per-group shift) re-aligns every group to ONE keyword cipher, so
    the correct (period, progression, base) SNAPS the de-drifted columns from flat back to
    peaked -- detectable by per-coset IoC alone, cheaply, without touching the keyword. That
    escapes the coupled keyword+progression search (the keyword need not be a dictionary
    word). We rank all (period, base, progression) by de-drifted column IoC, then for the
    best few recover each keyword letter by per-column English fit and score the read.
    """
    ct = [ord(c) - 65 for c in letters]
    n = len(ct)
    ranked: list[tuple[float, int, str, int]] = []
    for base in bases:
        _enc, dec, _kv = _BASES[base]
        for period in range(2, max_period + 1):
            for g in range(26):
                resid = [dec((g * (i // period)) % 26, ct[i]) % 26 for i in range(n)]
                ranked.append((_period_cic(resid, period), period, base, g))
    ranked.sort(key=lambda r: r[0], reverse=True)

    out: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for _cic, period, base, g in ranked[:prefilter]:
        enc, dec, key_val = _BASES[base]
        resid = [dec((g * (i // period)) % 26, ct[i]) % 26 for i in range(n)]
        # recover each keyword letter by best English per-column fit
        keyword_chars: list[str] = []
        for col in range(period):
            column = resid[col::period]
            best_letter, best_fit = 0, -1e18
            for letter in range(26):
                shift = key_val(letter)
                fit = sum(_ENG_LOGFREQ[dec(shift, y) % 26] for y in column)
                if fit > best_fit:
                    best_fit, best_letter = fit, letter
            keyword_chars.append(chr(best_letter + 65))
        keyword = "".join(keyword_chars)
        key_repr = f"{keyword}/{g}/{base}"
        if key_repr in seen:
            continue
        seen.add(key_repr)
        # decode via the keyword-layer decrypt (progression already known)
        shifts = [key_val(ord(ch) - 65) for ch in keyword]
        plain = "".join(chr(dec(shifts[i % period], resid[i]) % 26 + 65) for i in range(n))
        out.append((scorer.score(plain), plain, key_repr))
    out.sort(key=lambda r: r[0], reverse=True)
    return out[:top]


class ProgressiveKey(Cipher):
    name = "progressive-key"
    aliases = ("progkey", "progressivekey")
    description = "Double-enciphered periodic cipher with a per-group progressing key."
    key_format = "keyword/progression (optionally /base: vigenere|beaufort|variant|porta)"
    key_example = "POLITICS/3"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        keyword, progression, base = _parse_key(key)
        enc, _dec, key_val = _BASES[base]
        letters = only_letters(text)
        return _transform(letters, keyword, progression, enc, enc, key_val)

    def decode(self, text: str, key: str) -> str:
        keyword, progression, base = _parse_key(key)
        _enc, dec, key_val = _BASES[base]
        letters = only_letters(text)
        # Undo the progression layer first, then the keyword layer, both with the
        # base cipher's decrypt equation.
        shifts = [key_val(ord(ch) - 65) for ch in keyword]
        length = len(shifts)
        out: list[str] = []
        for i, ch in enumerate(letters):
            z = ord(ch) - 65
            group = i // length
            prog_shift = (progression * group) % 26
            y = dec(prog_shift, z) % 26
            x = dec(shifts[i % length], y) % 26
            out.append(chr(x + 65))
        return "".join(out)

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
        """Recover a Progressive Key, with or without a keyword hint.

        The instance couples three unknowns: a keyword (a free-form word), a
        progression index (0-25) and the base cipher (4 choices).

        **With a keyword hint** (``opts["keyword"]`` or ``opts["keywords"]``) we
        brute-force all 26 progressions and (unless ``opts["base"]`` is given) all
        four bases for each supplied keyword, score every decryption and return
        the best.

        **Without a keyword hint** we recover it keyless via :func:`_keyless_recover`:
        the progression is found *first* (undoing the per-group progression re-aligns
        every group to one keyword cipher, snapping the columns from flat back to
        peaked -- rankable by column IoC alone, no keyword needed), then each keyword
        letter is recovered by a per-column English fit. Tuning: ``opts["max_period"]``
        (group-size ceiling, default ``n//8`` capped at 15) and ``opts["prefilter"]``
        (how many top (period, base, progression) triples to fully solve, default 24).
        """
        letters = only_letters(text)
        if len(letters) < 8:
            return []

        keywords: list[str] = []
        if opts.get("keyword"):
            keywords.append(str(opts["keyword"]))
        keywords.extend(str(k) for k in opts.get("keywords", []))
        keywords = [k for k in keywords if only_letters(k)]

        forced_base = opts.get("base")
        if forced_base is not None:
            name = _BASE_ALIASES.get(str(forced_base).lower())
            if name is None:
                return []
            bases = [name]
        else:
            bases = ["vigenere", "beaufort", "variant", "porta"]

        if not keywords:
            # Keyless recovery: recover the progression first (de-drift un-flattens the
            # columns -> detectable by column IoC), then each keyword letter by per-column
            # English fit. No dictionary keyword hint required.
            max_period = int(opts.get("max_period", min(len(letters) // 8, 15)))
            prefilter = int(opts.get("prefilter", 24))
            recovered = _keyless_recover(
                letters,
                scorer,
                bases=bases,
                max_period=max(2, max_period),
                prefilter=prefilter,
                top=top,
            )
            return [
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=key_repr,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"base": key_repr.rsplit("/", 1)[-1], "keyless": True},
                )
                for score, plain, key_repr in recovered
            ]

        deadline = (time.monotonic() + timeout) if timeout else None
        results: list[tuple[float, str, str]] = []  # (score, plaintext, key_repr)
        for keyword in keywords:
            kw = only_letters(keyword)
            for base in bases:
                if deadline and time.monotonic() > deadline:
                    break
                for progression in range(26):
                    plain = self.decode(letters, f"{kw}/{progression}/{base}")
                    key_repr = f"{kw}/{progression}/{base}"
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
                    meta={"base": key_repr.rsplit("/", 1)[-1]},
                )
            )
            if len(candidates) >= top:
                break
        return candidates
