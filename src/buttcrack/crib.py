"""Crib-drag locator for the additive Vigenere family.

Given a *crib* — a guessed plaintext fragment (a common or thematic word, **not**
the known answer) — slide it across the ciphertext and, at each position, compute
the key letters that would produce it. For a running key the implied key fragment
is itself English, so scoring it by n-gram fitness pinpoints where the crib really
sits and hands you a chunk of the key to extend; for a word-keyed Vigenere it
surfaces the repeating keyword. This needs no knowledge of the solution — only a
guess worth trying — which is exactly the lever that breaks a long/running key
when pure statistics stall.
"""

from __future__ import annotations

from .ciphers.quagmire3 import keyed_alphabet
from .scoring import NgramScorer, get_scorer
from .text import only_letters
from .words import long_word_coverage

# Implied key letter at a position, given ciphertext index ``c`` and crib index
# ``p`` (both 0-25), for each additive-family cipher.
_KEY_OF = {
    "vigenere": lambda c, p: (c - p) % 26,
    "beaufort": lambda c, p: (c + p) % 26,
    "variant-beaufort": lambda c, p: (p - c) % 26,
}

#: Standard A-Z alphabet, used as the ``alphabet='STD'`` reference and the n-gram
#: scoring space (the scorer is trained on A-Z text).
_STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _resolve_alphabet(alphabet: str) -> str:
    """Map an alphabet name (or literal 26-letter string) to its A-Z permutation.

    ``'STD'`` is the plain alphabet; any other name is treated as a KEYWORD and run
    through the KRYPTOS-style keyed-alphabet construction (keyword, dedup, then the
    rest of A-Z). A bare 26-letter permutation is accepted verbatim.
    """
    a = alphabet.upper()
    if a == "STD" or a == "STANDARD":
        return _STD
    if len(a) == 26 and set(a) == set(_STD):
        return a
    return keyed_alphabet(a)


def _to_alpha_index(letters: str, index: dict[str, int]) -> list[int]:
    return [index[c] for c in letters]


def crib_drag(text: str, crib: str, *, scorer: NgramScorer | None = None, top: int = 6) -> dict:
    """Best crib placements per Vigenere-family cipher, ranked by implied-key fitness.

    Returns ``{cipher: [{position, key_fragment, plaintext_window, score}, ...]}``.
    A high score means the key letters the crib implies there read like language
    (the running-key signal). Cribs shorter than the scorer's n-gram size can't be
    scored meaningfully — use a crib of at least 4 letters.
    """
    scorer = scorer or get_scorer()
    ct = [ord(x) - 65 for x in only_letters(text)]
    cb = [ord(x) - 65 for x in only_letters(crib)]
    out: dict[str, list[dict]] = {}
    if len(cb) < 1 or len(cb) > len(ct):
        return out
    crib_str = "".join(chr(65 + x) for x in cb)
    for name, key_of in _KEY_OF.items():
        placements: list[dict] = []
        for pos in range(len(ct) - len(cb) + 1):
            frag = "".join(chr(65 + key_of(ct[pos + i], cb[i])) for i in range(len(cb)))
            placements.append(
                {
                    "position": pos,
                    "key_fragment": frag,
                    "plaintext_window": crib_str,
                    "score": round(scorer.score(frag), 2),
                }
            )
        placements.sort(key=lambda d: d["score"], reverse=True)
        out[name] = placements[:top]
    return out


# ===========================================================================
# Product-cipher crib solver (two superimposed periodic shifts) via union-find
# ===========================================================================


class _DiffUF:
    """Weighted union-find over Z/26 for DIFFERENCE constraints.

    ``pot[x]`` holds ``value(x) - value(parent(x)) mod 26``. ``find`` returns the
    component root and ``value(x) - value(root)``. ``union(x, y, w)`` asserts
    ``value(x) - value(y) == w``; it returns ``False`` only on a contradiction (the
    two nodes already share a component with an incompatible offset).
    """

    def __init__(self, n: int) -> None:
        self.par = list(range(n))
        self.pot = [0] * n  # potential to PARENT

    def find(self, x: int) -> tuple[int, int]:
        path: list[tuple[int, int]] = []
        cur, acc = x, 0
        while self.par[cur] != cur:
            path.append((cur, acc))
            acc = (acc + self.pot[cur]) % 26
            cur = self.par[cur]
        root = cur  # acc == value(x) - value(root)
        for node, before in path:  # value(node)-value(root) = acc - (value(x)-value(node))
            self.pot[node] = (acc - before) % 26
            self.par[node] = root
        return root, acc

    def union(self, x: int, y: int, w: int) -> bool:
        rx, px = self.find(x)  # value(x) - value(rx) == px
        ry, py = self.find(y)  # value(y) - value(ry) == py
        if rx == ry:
            return ((px - py) % 26) == (w % 26)
        # value(x)-value(y)=w  =>  attach ry under rx with pot = px - py - w
        self.par[ry] = rx
        self.pot[ry] = (px - py - w) % 26
        return True


def product_crib_solve(
    text: str,
    crib: str,
    pos: int,
    p: int,
    q: int,
    *,
    alphabet: str = "KRYPTOS",
) -> dict:
    """Anchor a crib in a two-shift PRODUCT cipher and propagate every forced letter.

    Model (additive in the ``alphabet``-index space)::

        c_idx[i] = pt_idx[i] + a[i % p] + b[i % q]   (mod 26)

    A crib placed at ``pos`` asserts ``pt_idx[pos+j]`` for each crib letter, which
    fixes ``a[(pos+j) % p] + b[(pos+j) % q]``. Reparametrising ``b'_c := -b_c`` turns
    each into the difference constraint ``a_r - b'_c = w`` over a weighted union-find.
    Any text position ``i`` whose row node ``a[i % p]`` and column node ``b[i % q]``
    share a component is then DETERMINED with no free constant and is decoded.

    A correct crib of length ``>= p + q - 1`` connects the whole graph and so decodes
    the entire message. Returns ``{plaintext, determined, total, fraction,
    contradiction, word_coverage, score, alphabet, ...}`` where ``plaintext`` uses
    ``'.'`` for still-undetermined positions.
    """
    alpha = _resolve_alphabet(alphabet)
    index = {c: i for i, c in enumerate(alpha)}
    ct = only_letters(text)
    cb = only_letters(crib)
    n = len(ct)
    if not cb or pos < 0 or pos + len(cb) > n or p < 1 or q < 1:
        return {
            "plaintext": "." * n,
            "determined": 0,
            "total": n,
            "fraction": 0.0,
            "contradiction": False,
            "word_coverage": 0.0,
            "score": 0.0,
            "alphabet": alpha,
            "position": pos,
            "p": p,
            "q": q,
        }
    c_idx = _to_alpha_index(ct, index)
    crib_idx = _to_alpha_index(cb, index)

    uf = _DiffUF(p + q)
    ok = True
    for j, ci in enumerate(crib_idx):
        i = pos + j
        r, c = i % p, i % q
        w = (c_idx[i] - ci) % 26  # a_r + b_c = w  ->  a_r - b'_c = w with b'_c node = p+c
        if not uf.union(r, p + c, w):
            ok = False

    out = ["."] * n
    determined = 0
    for i in range(n):
        rr, pr = uf.find(i % p)
        rc, pc = uf.find(p + (i % q))
        if rr == rc:
            s = (pr - pc) % 26  # a_r + b_c (free constant cancels)
            out[i] = alpha[(c_idx[i] - s) % 26]
            determined += 1
    plaintext = "".join(out)
    letters = plaintext.replace(".", "")
    coverage = long_word_coverage(letters) if len(letters) >= 4 else 0.0
    scorer = get_scorer()
    score = scorer.score(letters) if len(letters) >= 4 else 0.0
    return {
        "plaintext": plaintext,
        "determined": determined,
        "total": n,
        "fraction": round(determined / n, 4) if n else 0.0,
        "contradiction": not ok,
        "word_coverage": round(coverage, 4),
        "score": round(score, 2),
        "alphabet": alpha,
        "position": pos,
        "p": p,
        "q": q,
    }


def product_crib_sweep(
    text: str,
    cribs,
    *,
    p_range,
    q_range,
    alphabet: str = "KRYPTOS",
    positions=None,
    min_coverage: float = 0.4,
    top: int = 10,
) -> list[dict]:
    """Sweep ``product_crib_solve`` over cribs / periods / positions, keep readable solves.

    For every crib, every ``(p, q)`` with ``p < q`` and ``gcd(p, q) == 1`` (the
    product cipher is degenerate otherwise), and every candidate ``positions`` (default:
    all offsets where the crib fits), the union-find solve is run and the propagated
    plaintext is gated on word coverage. Returns the best ``top`` candidates (no
    contradiction, ``word_coverage >= min_coverage``) sorted by coverage then score.
    """
    from math import gcd

    if isinstance(cribs, str):
        cribs = [cribs]
    ct = only_letters(text)
    n = len(ct)
    results: list[dict] = []
    seen: set[str] = set()
    for crib in cribs:
        cb = only_letters(crib)
        if not cb or len(cb) > n:
            continue
        place = positions if positions is not None else range(n - len(cb) + 1)
        for p in p_range:
            for q in q_range:
                if p < 1 or q < 1 or p >= q or gcd(p, q) != 1:
                    continue
                for pos in place:
                    res = product_crib_solve(text, cb, pos, p, q, alphabet=alphabet)
                    if res["contradiction"] or res["word_coverage"] < min_coverage:
                        continue
                    key = res["plaintext"]
                    if key in seen:
                        continue
                    seen.add(key)
                    res = dict(res)
                    res["crib"] = cb
                    results.append(res)
    results.sort(key=lambda r: (r["word_coverage"], r["score"]), reverse=True)
    return results[:top]


# ===========================================================================
# Keyed-alphabet crib-drag (Quagmire family)
# ===========================================================================


def keyed_alphabet_crib_drag(
    text: str,
    crib: str,
    *,
    alphabet: str = "KRYPTOS",
    conventions=("vigenere", "beaufort", "variant"),
    scorer: NgramScorer | None = None,
    top: int = 6,
) -> dict:
    """Crib-drag a guessed word inside a KEYED alphabet (Quagmire family).

    Identical in spirit to :func:`crib_drag` but the additive arithmetic happens in
    the ``alphabet``-index space (KRYPTOS by default), so the implied key fragment is
    reported as keyed-alphabet letters. For a word-keyed Quagmire the correct offset
    surfaces the repeating indicator keyword; for a running keyed key it surfaces an
    English-looking fragment. Conventions follow the family grammar (in index space):

      * ``vigenere`` decrypt ``p = c - k``  -> implied ``k = c - p``
      * ``beaufort`` decrypt ``p = k - c``  -> implied ``k = c + p``
      * ``variant``  decrypt ``p = c + k``  -> implied ``k = p - c``

    Returns ``{convention: [{position, key_fragment, plaintext_window, score}, ...]}``.
    """
    scorer = scorer or get_scorer()
    alpha = _resolve_alphabet(alphabet)
    index = {c: i for i, c in enumerate(alpha)}
    ct = only_letters(text)
    cb = only_letters(crib)
    out: dict[str, list[dict]] = {}
    if not cb or len(cb) > len(ct):
        return out
    c_idx = _to_alpha_index(ct, index)
    crib_idx = _to_alpha_index(cb, index)
    key_of = {
        "vigenere": lambda c, p: (c - p) % 26,
        "beaufort": lambda c, p: (c + p) % 26,
        "variant": lambda c, p: (p - c) % 26,
    }
    for name in conventions:
        fn = key_of[name]
        placements: list[dict] = []
        for pos in range(len(ct) - len(cb) + 1):
            frag = "".join(alpha[fn(c_idx[pos + i], crib_idx[i])] for i in range(len(cb)))
            placements.append(
                {
                    "position": pos,
                    "key_fragment": frag,
                    "plaintext_window": cb,
                    "score": round(scorer.score(frag), 2),
                }
            )
        placements.sort(key=lambda d: d["score"], reverse=True)
        out[name] = placements[:top]
    return out


# ===========================================================================
# Plaintext-autokey crib unzip
# ===========================================================================


def _autokey_unzip(
    c_idx: list[int],
    alpha: str,
    seeds: dict[int, int],
    primer_len: int,
    beaufort: bool,
) -> str | None:
    """Propagate a plaintext-autokey from seeded positions; ``None`` if it cannot fill.

    Recurrence (in index space), key letter for position ``i`` is ``PT[i - L]``:

      * vigenere: ``C = PT + key`` -> ``PT[i] = C[i] - PT[i-L]``,  ``PT[i-L] = C[i] - PT[i]``
      * beaufort: ``C = key - PT`` -> ``PT[i] = PT[i-L] - C[i]``,  ``PT[i-L] = C[i] + PT[i]``
    """
    n = len(c_idx)
    pt = dict(seeds)
    changed = True
    while changed:
        changed = False
        for i in range(primer_len, n):
            ci = c_idx[i]
            if (i - primer_len) in pt and i not in pt:  # forward
                k = pt[i - primer_len]
                pt[i] = (ci - k) % 26 if not beaufort else (k - ci) % 26
                changed = True
            if i in pt and (i - primer_len) not in pt:  # backward
                pi = pt[i]
                pt[i - primer_len] = (ci - pi) % 26 if not beaufort else (ci + pi) % 26
                changed = True
    if len(pt) < n - primer_len:  # the first L (primer) positions may stay unknown
        return None
    return "".join(alpha[pt[i]] if i in pt else "?" for i in range(n))


def autokey_crib_unzip(
    text: str,
    cribs,
    *,
    alphabets=("KRYPTOS", "STD"),
    conventions=("vigenere", "beaufort"),
    max_primer: int = 14,
    positions=None,
    scorer: NgramScorer | None = None,
    min_coverage: float = 0.4,
    top: int = 8,
) -> list[dict]:
    """Crib-drag UNZIP of a PLAINTEXT-autokey cipher.

    In a plaintext-autokey the running key is the plaintext itself shifted by the
    primer length ``L``: ``key[i] = PT[i - L]`` (the first ``L`` letters come from an
    unknown primer). A crib spanning ``>= L`` consecutive positions therefore seeds
    all ``L`` interleaved recurrence chains, and the whole plaintext unzips forward
    (and backward into the primer where reachable). Every ``(crib, position, primer
    length L, convention, alphabet)`` is tried; fully-unzipped candidates that read as
    English (word-coverage gate) are returned, best-first.

    ``conventions``: ``vigenere`` (``C = PT + key``) or ``beaufort`` (``C = key - PT``),
    in the ``alphabet``-index space. Returns a list of
    ``{plaintext, crib, position, primer_len, convention, alphabet, word_coverage,
    score}`` sorted by coverage then score.
    """
    scorer = scorer or get_scorer()
    if isinstance(cribs, str):
        cribs = [cribs]
    ct = only_letters(text)
    n = len(ct)
    results: list[dict] = []
    seen: set[str] = set()
    bf_of = {"vigenere": False, "beaufort": True}
    for alpha_name in alphabets:
        alpha = _resolve_alphabet(alpha_name)
        index = {c: i for i, c in enumerate(alpha)}
        c_idx = _to_alpha_index(ct, index)
        for crib in cribs:
            cb = only_letters(crib)
            m = len(cb)
            if not cb or m > n:
                continue
            crib_idx = _to_alpha_index(cb, index)
            place = positions if positions is not None else range(n - m + 1)
            for conv in conventions:
                beaufort = bf_of[conv]
                for primer_len in range(1, min(max_primer, m) + 1):
                    for pos in place:
                        seeds = {pos + k: crib_idx[k] for k in range(m)}
                        full = _autokey_unzip(c_idx, alpha, seeds, primer_len, beaufort)
                        if full is None or "?" in full[primer_len:]:
                            continue
                        letters = full.replace("?", "")
                        if len(letters) < 4:
                            continue
                        coverage = long_word_coverage(letters)
                        if coverage < min_coverage:
                            continue
                        if full in seen:
                            continue
                        seen.add(full)
                        results.append(
                            {
                                "plaintext": full,
                                "crib": cb,
                                "position": pos,
                                "primer_len": primer_len,
                                "convention": conv,
                                "alphabet": alpha_name.upper(),
                                "word_coverage": round(coverage, 4),
                                "score": round(scorer.score(letters), 2),
                            }
                        )
    results.sort(key=lambda r: (r["word_coverage"], r["score"]), reverse=True)
    return results[:top]
