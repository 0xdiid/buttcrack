"""Josse cipher (Major H. D. Josse, French Army, ~1889).

A **ciphertext-autokey over a keyed mixed alphabet** — obscure enough that it is
absent from the ACA registry, but fully specified and, importantly, *breakable
ciphertext-only at short lengths*.

The system was reconstructed by Géraud-Stewart & Naccache (*Cryptologia* 45(4),
2021) from Josse's sealed 1889 deposit, and broken by Lasry (*Cryptologia* 47(1),
2023) with a ciphertext-only attack that succeeds from ~75 letters.

KEY format
----------
``"<keyword>"`` — the keyword builds a mixed alphabet, which supplies the numbering
used by the arithmetic. Historically the alphabet is written into a table and read
off **column-wise**; the numbering is then position-in-that-alphabet.

Josse's original drops ``W`` (25 letters). This implementation is the natural
26-letter generalisation by default (``drop=None``); pass ``drop="W"`` for the
historical form.

ENCRYPT / DECRYPT
-----------------
With ``num`` the position of a letter in the mixed alphabet and ``let`` its inverse::

    C_1 = let( -num(P_1)                 mod n )
    C_i = let(  num(P_i) + num(C_{i-1})  mod n )

so decryption is a plain first difference::

    P_1 = let( -num(C_1)                 mod n )
    P_i = let(  num(C_i) - num(C_{i-1})  mod n )

Encrypt and decrypt are NOT reciprocal.

CRYPTANALYSIS (why the keyless attack works)
--------------------------------------------
The attack decomposes, which is what makes it tractable. Write
``D_i = num(C_i) - num(C_{i-1})``. If the numbering is **correct**, then
``D_i = num(P_i)`` exactly — the difference sequence is a *monoalphabetic image of
the plaintext*. Therefore:

* **Stage 1** — search the numbering by maximising the **DIGRAPH index of
  coincidence of D**. D is a monoalphabetic image of the plaintext exactly when the
  numbering is right, and digraph IoC is *mono-invariant*, so it reads English's
  value at the true numbering and the flat value everywhere else. Measured at
  n=153: true 4.12 vs a random band of 1.08 +- 0.23 (z = +13.3; best of 300 random
  numberings only 1.88). At n=75 it is still z = +5.7.
  ⚠ Two weaker objectives were tried first and BOTH fail — do not reuse them:
  unigram IoC(D) is overfittable (a wrong numbering reaches 2.32 vs the true 1.89,
  because maximising concentration finds spikes, not language), and sorted-profile
  matching is too coarse (26 numbers; many numberings match). Only a mono-INVARIANT
  n-gram statistic identifies the numbering.
  Stage 1 keeps the top-K numberings, not just the argmax.
* **Stage 2** — D is then an ordinary simple substitution cryptogram, solved by
  n-gram hill-climbing.

A naive single-stage anneal over the alphabet does *not* converge, because the
alphabet is used twice (to number the ciphertext, and to read the difference back
out), so one transposition perturbs the decode twice.
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import ALPHABET, only_letters, reflow
from .base import Cipher


def _keyed_alphabet(keyword: str, drop: str | None = None) -> str:
    """Deduped keyword letters, then the remaining alphabet (minus ``drop``)."""
    alpha = ALPHABET if not drop else ALPHABET.replace(drop.upper(), "")
    seq: list[str] = []
    for ch in keyword.upper():
        if ch in alpha and ch not in seq:
            seq.append(ch)
    for ch in alpha:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _columnar_read(alphabet: str, width: int) -> str:
    """Write ``alphabet`` row-wise into ``width`` columns, read it back column-wise.

    This is Josse's own table step; it is what turns a keyword into a *mixed*
    numbering rather than a merely keyed one.
    """
    if width <= 1:
        return alphabet
    cols: list[list[str]] = [[] for _ in range(width)]
    for i, ch in enumerate(alphabet):
        cols[i % width].append(ch)
    return "".join("".join(c) for c in cols)


def mixed_alphabet(key: str, drop: str | None = None) -> str:
    """Full Josse numbering alphabet from a key.

    ``key`` may be ``"KEYWORD"`` or ``"KEYWORD/WIDTH"``; the width defaults to the
    number of distinct keyword letters (Josse's table is keyword-wide).
    """
    kw, _, w = key.partition("/")
    base = _keyed_alphabet(kw, drop)
    if w.strip():
        width = int(w)
    else:
        width = len({c for c in kw.upper() if c.isalpha()}) or 1
    return _columnar_read(base, width)


class Josse(Cipher):
    name = "josse"
    aliases = ("josse-cipher",)
    description = "Ciphertext-autokey over a keyed mixed alphabet (Josse, 1889)"
    key_format = "keyword[/table-width]"
    key_example = "CHIEN"
    complexity = 4

    # ---------------------------------------------------------------- codec
    def _codec(self, key: str, drop: str | None):
        alpha = mixed_alphabet(key, drop)
        num = {ch: i for i, ch in enumerate(alpha)}
        return alpha, num, len(alpha)

    def encode(self, text: str, key: str, *, drop: str | None = None) -> str:
        alpha, num, n = self._codec(key, drop)
        out: list[str] = []
        prev = 0
        for i, ch in enumerate(only_letters(text)):
            if ch not in num:  # letter outside a reduced alphabet (e.g. historical W)
                continue
            v = (-num[ch]) % n if i == 0 else (num[ch] + prev) % n
            out.append(alpha[v])
            prev = v
        return reflow(text, "".join(out))

    def decode(self, text: str, key: str, *, drop: str | None = None) -> str:
        alpha, num, n = self._codec(key, drop)
        letters = [c for c in only_letters(text) if c in num]
        out: list[str] = []
        prev = 0
        for i, ch in enumerate(letters):
            v = num[ch]
            d = (-v) % n if i == 0 else (v - prev) % n
            out.append(alpha[d])
            prev = v
        return reflow(text, "".join(out))

    # ------------------------------------------------------------ analysis
    @staticmethod
    def difference_sequence(text: str, numbering: list[int], n: int = 26) -> list[int]:
        """``D_i = num(C_i) - num(C_{i-1})`` for a candidate numbering.

        ``numbering[k]`` is the number assigned to ``ALPHABET[k]``.
        """
        letters = only_letters(text)
        v = [numbering[ord(c) - 65] for c in letters]
        out = [(-v[0]) % n]
        out.extend((v[i] - v[i - 1]) % n for i in range(1, len(v)))
        return out

    #: sorted English unigram profile, high to low (set lazily)
    _ENG_SORTED: list[float] | None = None

    @classmethod
    def _eng_sorted(cls) -> list[float]:
        if cls._ENG_SORTED is None:
            from importlib import resources

            counts = [0.0] * 26
            raw = resources.files("buttcrack.data").joinpath("english_monograms.txt").read_text()
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] in ALPHABET:
                    counts[ALPHABET.index(parts[0])] = float(parts[1])
            tot = sum(counts) or 1.0
            cls._ENG_SORTED = sorted((x / tot for x in counts), reverse=True)
        return cls._ENG_SORTED

    @staticmethod
    def digraph_ioc(seq: list[int], n: int = 26) -> float:
        """Mono-invariant digraph IoC of ``seq`` — the stage-1 objective."""
        counts: dict[int, int] = {}
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            k = a * n + b
            counts[k] = counts.get(k, 0) + 1
        m = len(seq) - 1
        if m < 2:
            return 0.0
        return (n * n) * sum(v * (v - 1) for v in counts.values()) / (m * (m - 1))

    @classmethod
    def profile_fit(cls, seq: list[int], n: int = 26) -> float:
        """Negative SSE between D's sorted frequency profile and English's.

        The stage-1 objective. Zero is a perfect match; more negative is worse.
        """
        counts = [0] * n
        for x in seq:
            counts[x] += 1
        total = len(seq) or 1
        obs = sorted((c / total for c in counts), reverse=True)
        eng = cls._eng_sorted()
        return -sum((a - b) ** 2 for a, b in zip(obs, eng, strict=True))

    @staticmethod
    def _ioc(seq: list[int], n: int = 26) -> float:
        counts = [0] * n
        for x in seq:
            counts[x] += 1
        total = len(seq)
        if total < 2:
            return 0.0
        s = sum(c * (c - 1) for c in counts)
        return n * s / (total * (total - 1))

    def crack(
        self,
        text: str,
        scorer: NgramScorer,
        *,
        top: int = 5,
        rng=None,
        timeout: float | None = None,
        restarts: int = 40,
        steps: int = 24000,
        **opts,
    ) -> list[Candidate]:
        """Two-stage keyless attack (see module docstring)."""
        import random

        from .substitution import Substitution

        rnd = rng or random.Random(0xB07)
        letters = only_letters(text)
        if len(letters) < 40:
            return []
        started = time.time()

        # -- stage 1: recover the numbering by PROFILE FIT (not IoC — see docstring)
        keep: list[tuple[float, list[int]]] = []
        best_num, best_ioc = None, -1e18
        for _ in range(restarts):
            if timeout and time.time() - started > timeout * 0.6:
                break
            num = list(range(26))
            rnd.shuffle(num)
            cur = self.digraph_ioc(self.difference_sequence(text, num))
            levels = 200
            per = max(1, steps // levels)
            for lev in range(levels):
                t = 0.60 * (1 - lev / levels) + 1e-9
                for _ in range(per):
                    a, b = rnd.randrange(26), rnd.randrange(26)
                    if a == b:
                        continue
                    num[a], num[b] = num[b], num[a]
                    cand = self.digraph_ioc(self.difference_sequence(text, num))
                    import math

                    if cand >= cur or rnd.random() < math.exp((cand - cur) / max(t, 1e-9)):
                        cur = cand
                        if cand > best_ioc:
                            best_ioc, best_num = cand, list(num)
                    else:
                        num[a], num[b] = num[b], num[a]
            keep.append((cur, list(num)))
        if best_num is None:
            return []

        # -- stage 2: each surviving numbering turns D into a simple substitution
        # cryptogram. The stage-1 objective is noisy at n~150, so try the top-K
        # numberings rather than trusting the argmax.
        keep.sort(key=lambda kv: -kv[0])
        seen: set[tuple[int, ...]] = set()
        shortlist: list[list[int]] = [best_num]
        for _, num in keep:
            key_t = tuple(num)
            if key_t in seen:
                continue
            seen.add(key_t)
            shortlist.append(num)
            if len(shortlist) >= int(opts.get("shortlist", 12)):
                break

        sub = Substitution()
        out: list[Candidate] = []
        for num in shortlist:
            if timeout and time.time() - started > timeout:
                break
            diff = self.difference_sequence(text, num)
            crypto = "".join(ALPHABET[d] for d in diff)
            inner = sub.crack(crypto, scorer, top=1, rng=rnd, timeout=None)
            for cand in inner[:1]:
                out.append(
                    self._candidate(
                        scorer,
                        cand.plaintext,
                        None,
                        numbering="".join(ALPHABET[i] for i in num),
                        digraph_ioc=round(self.digraph_ioc(diff), 4),
                        diff_ioc=round(self._ioc(diff), 4),
                        inner_key=cand.key,
                    )
                )
        out.sort(key=lambda c: -c.score)
        return out[:top]
