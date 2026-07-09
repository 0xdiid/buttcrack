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
        """Best-effort brute force over keyword, progression and base cipher.

        The full keyless problem couples three unknowns: a keyword (effectively a
        free-form word), a progression index (0-25 distinct values) and the base
        cipher (4 choices). With no constraints the keyword space is unbounded, so
        this crack is scoped to supplied candidate keywords:

        * ``opts["keyword"]`` -- a single candidate keyword, or
        * ``opts["keywords"]`` -- an iterable of candidate keywords.

        For each keyword we brute-force all 26 progressions and (unless
        ``opts["base"]`` is given) all four base ciphers, score every decryption
        and return the best. With no keyword hint we return ``[]`` rather than
        pretend to solve an unbounded instance.
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

        forced_base = opts.get("base")
        if forced_base is not None:
            name = _BASE_ALIASES.get(str(forced_base).lower())
            if name is None:
                return []
            bases = [name]
        else:
            bases = ["vigenere", "beaufort", "variant", "porta"]

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
