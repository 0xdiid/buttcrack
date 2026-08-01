"""The ``hill-additive`` cipher: a Hill matrix over a periodic additive.

The solver lives in :mod:`buttcrack.hill_affine`; it is imported lazily inside the methods
so that registering this cipher does not drag the layered/scoring stack into the cipher
package's import cycle.
"""

from __future__ import annotations

from ..text import only_letters
from .base import Cipher


class HillAdditive(Cipher):
    """`CT = M · (P + K)` — a Hill matrix over a periodic additive.

    Key: ``MATRIXWORD/ADDITIVEWORD[/ALPHABET]``, e.g. ``HAMMERING/TEMPER/KRYPTOS``. The
    matrix word is read row-major and must be a perfect-square length; the additive word
    sets the letter-level period. Both index into ``ALPHABET`` (default the plain A-Z; pass
    a keyword to index in a keyed alphabet instead).
    """

    name = "hill-additive"
    aliases = ("hilladd", "hill-offset", "affine-hill")
    description = "Hill matrix over a periodic additive: CT = M*(P + K) mod 26."
    key_format = "MATRIXWORD/ADDITIVEWORD[/ALPHABET] (matrix word length must be a square)"
    key_example = "HAMMERING/TEMPER/KRYPTOS"
    complexity = 7

    @staticmethod
    def _helpers():
        from ..hill_affine import additive_word, apply_inverse, crack_hill_additive
        from ..layered import alphabet_header
        from .hill import inverse_mod26, is_invertible_mod26, matrix_from_word

        return dict(
            additive_word=additive_word,
            apply_inverse=apply_inverse,
            crack_hill_additive=crack_hill_additive,
            alphabet_header=alphabet_header,
            inverse_mod26=inverse_mod26,
            is_invertible_mod26=is_invertible_mod26,
            matrix_from_word=matrix_from_word,
        )

    @staticmethod
    def _parse(key: str) -> tuple[list[list[int]], list[int], str, int]:
        H = HillAdditive._helpers()
        parts = [p for p in str(key).split("/") if p != ""]
        if len(parts) < 2:
            raise ValueError("hill-additive key is MATRIXWORD/ADDITIVEWORD[/ALPHABET]")
        mword, aword = parts[0].upper(), parts[1].upper()
        alphabet = parts[2].upper() if len(parts) > 2 else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        header = H["alphabet_header"](alphabet)
        pos = {c: i for i, c in enumerate(header)}
        m = H["matrix_from_word"](mword, header)
        n = len(m)
        if not H["is_invertible_mod26"](m):
            raise ValueError(f"matrix from {mword!r} is not invertible mod 26")
        add = [pos[c] for c in only_letters(aword) if c in pos]
        if not add:
            raise ValueError("additive word has no letters in the alphabet")
        return m, add, header, n

    def encode(self, text: str, key: str) -> str:
        m, add, header, n = self._parse(key)
        pos = {c: i for i, c in enumerate(header)}
        idx = [pos[c] for c in only_letters(text).upper() if c in pos]
        shifted = [(v + add[i % len(add)]) % 26 for i, v in enumerate(idx)]
        out: list[str] = []
        for blk in [shifted[i : i + n] for i in range(0, (len(shifted) // n) * n, n)]:
            for i in range(n):
                out.append(header[sum(m[i][k] * blk[k] for k in range(n)) % 26])
        return "".join(out)

    def decode(self, text: str, key: str) -> str:
        H = HillAdditive._helpers()
        m, add, header, n = self._parse(key)
        pos = {c: i for i, c in enumerate(header)}
        idx = [pos[c] for c in only_letters(text).upper() if c in pos]
        stream = H["apply_inverse"](idx, H["inverse_mod26"](m), n)
        return "".join(header[(v - add[i % len(add)]) % 26] for i, v in enumerate(stream))

    def crack(self, text, scorer, *, top=5, rng=None, timeout=None, **opts):
        H = HillAdditive._helpers()
        """Scan matrix keywords; the additive falls out analytically for each one."""
        from ..result import Candidate
        from ..words import _words

        n = int(opts.get("n", 3))
        alphabet = str(opts.get("alphabet", "KRYPTOS"))
        words = opts.get("words")
        if words is None:
            words = [w.upper() for w in _words() if len(w) in (n, n * n) and w.isalpha()]
        sols = H["crack_hill_additive"](
            text,
            scorer,
            words,
            n=n,
            alphabet=alphabet,
            top=top,
            periods=tuple(opts.get("periods", (1, 2, 3, 4, 6, 8, 9, 12))),
        )
        out = []
        for s in sols:
            add_word = H["additive_word"](s.additive, s.alphabet)
            key = f"{s.matrix_word or 'matrix'}/{add_word}/{alphabet}"
            out.append(Candidate(cipher=self.name, key=key, plaintext=s.plaintext, score=s.score))
        return out
