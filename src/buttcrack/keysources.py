"""Derive candidate keys from a corpus of *previously solved* messages, and build /
decompose word-pair composed keys (a construction some puzzle series use).

Two recurring needs in an "each puzzle builds on the last" series that no other module
covered, so they were hand-rolled repeatedly:

1. :func:`keys_from_corpus` — the key is *hidden in an earlier puzzle*: a whole prior
   plaintext used as a running key, an acrostic, a window, a single word, or a light
   transform of prior-solution text. This enumerates those candidates (with provenance)
   so they can be fed to the running-key screen, a Vigenere/Quagmire key, or — via
   :func:`buttcrack.ciphers.columnar._read_order` — a transposition order.

2. :func:`compose_key` / :func:`decompose_key` — a long "random-looking" periodic key
   can itself be two short words combined through a Quagmire, e.g.
   ``QuagmireKEYED(WATERMELON x4, key=LAVENDER)`` (lcm(10,8)=40) or ``(MAPLE x9,
   RIVERBANK)`` (lcm(5,9)=45). Build such a key from a word pair, or recover the word
   pair from a periodic key (a self-validating check when it hits).
"""

from __future__ import annotations

from math import gcd

from .text import only_letters

KRYPTOS = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
STANDARD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: encode conventions in keyed-alphabet index space (match validate/runkey):
#:   vigenere c = p + k    beaufort c = k - p    variant c = p - k
CONVENTIONS = ("vigenere", "beaufort", "variant")


def _alphabet(name_or_alphabet: str) -> str:
    s = name_or_alphabet.upper()
    if s in ("KRYPTOS", "KRY"):
        return KRYPTOS
    if s in ("STD", "STANDARD"):
        return STANDARD
    if len(s) == 26 and set(s) == set(STANDARD):
        return s
    raise ValueError("alphabet must be 'KRYPTOS', 'STD', or a 26-letter permutation")


def _atbash(s: str) -> str:
    return "".join(chr(155 - ord(c)) if "A" <= c <= "Z" else c for c in s.upper())


def _reverse(s: str) -> str:
    return only_letters(s)[::-1]


def _acrostic(text: str, sep: str) -> str:
    """First letter of each unit (``sep`` = 'word'|'sentence'|'line')."""
    import re

    if sep == "word":
        units = text.split()
    elif sep == "line":
        units = text.splitlines()
    elif sep == "sentence":
        units = re.split(r"[.!?]+", text)
    else:
        raise ValueError("sep must be word/sentence/line")
    out = []
    for u in units:
        letters = only_letters(u)
        if letters:
            out.append(letters[0])
    return "".join(out)


def keys_from_corpus(
    corpus,
    *,
    labels=None,
    full: bool = True,
    acrostics: bool = True,
    word_keys: bool = True,
    min_word: int = 3,
    window_lengths=(),
    transforms: bool = True,
) -> list[dict]:
    """Enumerate candidate key strings derived from prior-solution texts.

    ``corpus`` is a mapping ``label -> text`` or an iterable of texts (pass ``labels`` for
    names). Returns a deduplicated list of ``{"value", "kind", "source"}`` dicts:

    - ``full`` — the whole cleaned text (use as a *running key*).
    - ``acrostic-word`` / ``acrostic-sentence`` / ``acrostic-line`` — initial-letter strings.
    - ``word`` — each distinct word of length >= ``min_word`` (use as a keyword / square key).
    - ``window:L`` — every length-``L`` substring (only when ``window_lengths`` given; this is
      crib-drag territory — for an exhaustive slide also see :func:`buttcrack.crib.crib_drag`).
    - ``reverse:`` / ``atbash:`` — light transforms of the above (when ``transforms``).

    Every value is uppercase letters only. Any value also works as a transposition read
    order via ``columnar._read_order`` (rank of its letters).
    """
    if isinstance(corpus, dict):
        items = list(corpus.items())
    else:
        corpus = list(corpus)
        labs = labels if labels is not None else [f"#{i}" for i in range(len(corpus))]
        items = list(zip(labs, corpus))

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, kind: str, source: str) -> None:
        value = only_letters(value)
        if len(value) < 2:
            return
        key = (value, kind)
        if key in seen:
            return
        seen.add(key)
        out.append({"value": value, "kind": kind, "source": source})

    base: list[dict] = []  # candidates that transforms should also be applied to
    for label, text in items:
        cleaned = only_letters(text)
        if full and cleaned:
            base.append({"value": cleaned, "kind": "full", "source": label})
        if acrostics:
            for sep in ("word", "sentence", "line"):
                a = _acrostic(text, sep)
                if a:
                    base.append({"value": a, "kind": f"acrostic-{sep}", "source": label})
        if word_keys:
            for w in {only_letters(tok) for tok in text.split()}:
                if len(w) >= min_word:
                    add(w, "word", label)
        for L in window_lengths:
            for i in range(0, len(cleaned) - L + 1):
                add(cleaned[i : i + L], f"window:{L}", label)

    for c in base:
        add(c["value"], c["kind"], c["source"])
    if transforms:
        for c in base:
            add(_reverse(c["value"]), f"reverse:{c['kind']}", c["source"])
            add(_atbash(c["value"]), f"atbash:{c['kind']}", c["source"])
    return out


# --- composed keys (word-pair canon) -----------------------------------------------

def _combine(a: int, b: int, convention: str) -> int:
    if convention == "vigenere":
        return (a + b) % 26
    if convention == "beaufort":
        return (b - a) % 26
    if convention == "variant":
        return (a - b) % 26
    raise ValueError(f"unknown convention {convention!r}")


def _uncombine(c: int, b: int, convention: str) -> int:
    """Recover ``a`` from a composed letter ``c`` and the second word's letter ``b``."""
    if convention == "vigenere":
        return (c - b) % 26
    if convention == "beaufort":
        return (b - c) % 26
    if convention == "variant":
        return (c + b) % 26
    raise ValueError(f"unknown convention {convention!r}")


def compose_key(
    word_a: str, word_b: str, *, alphabet: str = "KRYPTOS",
    convention: str = "vigenere", period: int | None = None,
) -> str:
    """Build a composed key: ``word_a`` enciphered under ``word_b``.

    Both words are cycled to ``period`` (default ``lcm(len a, len b)``) and combined letter
    by letter in ``alphabet`` index space. Returns the period-length key string.
    """
    alpha = _alphabet(alphabet)
    pos = {ch: i for i, ch in enumerate(alpha)}
    a = only_letters(word_a)
    b = only_letters(word_b)
    if not a or not b:
        raise ValueError("both words must contain letters")
    if period is None:
        period = a.__len__() * b.__len__() // gcd(len(a), len(b))
    return "".join(
        alpha[_combine(pos[a[i % len(a)]], pos[b[i % len(b)]], convention)]
        for i in range(period)
    )


def _minimal_period(s: str) -> int:
    """Smallest p such that ``s`` is ``s[:p]`` repeated."""
    n = len(s)
    for p in range(1, n + 1):
        if n % p == 0 and s[:p] * (n // p) == s:
            return p
    return n


def decompose_key(
    key: str, words, *, alphabet: str = "KRYPTOS", convention: str = "vigenere",
) -> list[dict]:
    """Recover the ``(word_a, word_b)`` pairs that build ``key`` via :func:`compose_key`.

    The self-validating check: decoding the composed key with one word yields the other
    word repeated. ``words`` is a candidate vocabulary (e.g. thematic words). For
    ``lcm(|a|,|b|) = P = len(key)`` both lengths must divide ``P``, so only words whose
    length divides ``P`` are tried as ``word_b``; ``word_a`` is then *derived* (not guessed)
    and checked for being a clean repeat of a word in ``words``.
    """
    alpha = _alphabet(alphabet)
    pos = {ch: i for i, ch in enumerate(alpha)}
    k = only_letters(key)
    P = len(k)
    if P == 0:
        return []
    kidx = [pos[c] for c in k]
    vocab = {only_letters(w) for w in words if only_letters(w)}
    by_len: dict[int, list[str]] = {}
    for w in vocab:
        by_len.setdefault(len(w), []).append(w)

    divisors = [d for d in range(1, P + 1) if P % d == 0]
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for lb in divisors:
        for b in by_len.get(lb, []):
            bidx = [pos[c] for c in b]
            a_full = "".join(alpha[_uncombine(kidx[i], bidx[i % lb], convention)] for i in range(P))
            la = _minimal_period(a_full)
            a = a_full[:la]
            if (la * lb) // gcd(la, lb) != P:
                continue
            if a in vocab and (a, b) not in seen:
                seen.add((a, b))
                found.append({"word_a": a, "word_b": b, "period": P,
                              "alphabet": alpha, "convention": convention})
    return found
