"""Keyword / key-square recovery — CryptoCrack's "Keyword Finder" family.

The inverse of the standard K1-style keyed-alphabet construction used by
:mod:`buttcrack.ciphers.substitution` and :mod:`buttcrack.ciphers.squares`:

    keyed = dedupe(keyword) + remaining alphabet letters in straight order

A keyed alphabet (or Polybius square laid out row-by-row) is just that string.
Given the keyed alphabet, this module recovers the leading keyword(s): the
prefix of distinct letters that precede the point where the alphabet collapses
into the remaining letters in plain A..Z order. Every candidate is *validated*
by re-running the forward construction and checking it reproduces the exact
input, so a returned keyword is guaranteed to regenerate the given alphabet.

The construction is many-to-one (e.g. keyword "AB" and "ABC" can build the same
alphabet when C already follows in order), so several prefix lengths can be
valid; all validating keywords are returned, shortest first.
"""

from __future__ import annotations

from .ciphers.squares import ALPHABET_5, ALPHABET_6
from .text import only_letters

_FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _build_keyed(keyword: str, alphabet: str) -> str:
    """Forward construction: dedupe ``keyword``, then append remaining letters.

    Mirrors ``PolybiusSquare.__init__`` / the substitution keyed-alphabet build.
    Letters outside ``alphabet`` (and duplicates) are ignored, exactly as the
    forward builders ignore them.
    """
    seq: list[str] = []
    for ch in keyword.upper() + alphabet:
        if ch in alphabet and ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _validate_alphabet(alphabet: str, *, expected_len: int) -> str:
    """Uppercase, keep A-Z/0-9, and length/permutation-check ``alphabet``."""
    cleaned = "".join(ch for ch in alphabet.upper() if "A" <= ch <= "Z" or "0" <= ch <= "9")
    if len(cleaned) != expected_len:
        raise ValueError(f"alphabet must be {expected_len} characters, got {len(cleaned)}")
    if len(set(cleaned)) != expected_len:
        raise ValueError("alphabet must contain no repeated characters")
    return cleaned


def _recover(keyed: str, alphabet: str) -> list[str]:
    """Recover validating keyword prefixes for ``keyed`` over ``alphabet``.

    A prefix ``keyed[:n]`` is a valid keyword exactly when the remaining suffix
    ``keyed[n:]`` lists the not-yet-used letters in natural ``alphabet`` order —
    i.e. the construction has fallen through to the straight remaining letters.
    Once that holds at some position it holds for every larger ``n`` too, so the
    candidates are the prefixes from the *first* such fall-through point up to
    one short of the full length. The minimal point is 0 only for a straight,
    unkeyed alphabet, which therefore yields no keyword.

    Each candidate is validated by re-running the forward construction.
    """
    pos = {ch: i for i, ch in enumerate(alphabet)}
    # ``start`` = smallest n such that keyed[n:] is strictly increasing in
    # alphabet index: the rightmost descent fixes it. No descent at all means
    # the whole string is in natural order (an unkeyed alphabet), start == 0.
    start = 0
    for n in range(len(keyed) - 1):
        if pos[keyed[n]] > pos[keyed[n + 1]]:
            start = n + 1

    if start == 0:
        return []  # straight, unkeyed alphabet: no keyword

    candidates: list[str] = []
    for n in range(start, len(keyed)):
        prefix = keyed[:n]
        if _build_keyed(prefix, alphabet) == keyed:
            candidates.append(prefix)
    return candidates


def keyword_from_alphabet(alphabet: str) -> list[str]:
    """Recover candidate keyword(s) for a 26-letter K1-style keyed alphabet.

    ``alphabet`` is the mixed alphabet produced by
    ``dedupe(keyword) + remaining A..Z``. Returns every leading keyword whose
    forward construction reproduces ``alphabet`` exactly, shortest first. A
    straight A..Z alphabet (no keyword) returns an empty list.
    """
    keyed = _validate_alphabet(alphabet, expected_len=26)
    if any(ch not in _FULL_ALPHABET for ch in keyed):
        raise ValueError("a 26-letter alphabet must be a permutation of A-Z")
    return _recover(keyed, _FULL_ALPHABET)


def keysquare_candidates(square: str, *, size: int = 5) -> list[str]:
    """Recover candidate keyword(s) for a Polybius square laid out row-by-row.

    ``square`` is the grid read left-to-right, top-to-bottom: 25 chars for a 5x5
    square (over ``ALPHABET_5``, J merged into I) or 36 chars for a 6x6 square
    (over ``ALPHABET_6``, A-Z plus 0-9). Returns every leading keyword whose
    forward construction reproduces the square exactly, shortest first; an
    unkeyed square (alphabet in natural order) returns an empty list.
    """
    if size == 5:
        alphabet = ALPHABET_5
    elif size == 6:
        alphabet = ALPHABET_6
    else:
        raise ValueError("size must be 5 or 6")

    expected_len = size * size
    keyed = _validate_alphabet(square, expected_len=expected_len)
    if any(ch not in alphabet for ch in keyed):
        raise ValueError(f"square must be a permutation of {alphabet!r}")
    return _recover(keyed, alphabet)


def find_keyword_in(text: str, *, size: int | None = None) -> list[str]:
    """Best-effort keyword recovery from a keyed alphabet or square string.

    Convenience dispatcher: cleans ``text`` to alphanumerics, then routes by
    length (26 -> alphabet, 25 -> 5x5 square, 36 -> 6x6 square). ``size`` forces
    the square interpretation. Returns the recovered keyword candidates, or an
    empty list when ``text`` is unkeyed or not a recognisable key shape.
    """
    cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
    if size is not None:
        return keysquare_candidates(cleaned, size=size)
    n = len(cleaned)
    if n == 26:
        return keyword_from_alphabet(cleaned)
    if n == 25:
        return keysquare_candidates(cleaned, size=5)
    if n == 36:
        return keysquare_candidates(cleaned, size=6)
    raise ValueError(
        "text must be a 26-letter keyed alphabet, a 25-char 5x5 square, "
        "or a 36-char 6x6 square (or pass size= explicitly)"
    )


# --------------------------------------------------------- transposition-key inversion
def keyword_from_order(order, wordlist, *, tie: str = "stable") -> list[str]:
    """Recover the keyword(s) that induce a columnar read-``order`` (the inverse of
    :func:`buttcrack.ciphers.columnar._read_order`).

    A columnar transposition key is a keyword whose letters, argsorted (ties broken
    left-to-right), give the column read-order. This turns a *recovered numeric order*
    back into the thematic keyword a human likely chose — the search only ever reaches
    orders that spell a word, so the reverse map lets a blindly-recovered order self-report
    whether it is a word (and which). Many-to-one: several words can induce one order.
    Returns validated matches, empty if the order is not keyword-inducible from the list.
    """
    from .ciphers.columnar import _read_order

    order = list(order)
    n = len(order)
    out: list[str] = []
    seen: set[str] = set()
    for word in wordlist:
        letters = only_letters(str(word).upper())
        if len(letters) != n or letters in seen:
            continue
        try:
            if _read_order(letters) == order:
                out.append(letters)
                seen.add(letters)
        except ValueError:
            continue
    return out


def describe_permutation(perm) -> list[str]:
    """Label a permutation with the human, *recognisable* generator(s) that produce it.

    A recognisable-permutation wall is a low-entropy, human-recognisable permutation that resists
    blind n-gram search (flat gradient) yet is not a dictionary keyword. This checks a recovered
    read-order against named non-keyword generators — identity, reversal, rotation, odd/even
    interleave, and the out-faro (riffle) shuffle — so a recovered transposition can
    self-report "this is a reverse" / "rotate-3" instead of reading as random noise. The
    order and its inverse are both checked (the read-order convention is arbitrary). Returns
    every matching label; empty when it matches no known generator.
    """
    perm = list(perm)
    n = len(perm)
    if sorted(perm) != list(range(n)):
        return []
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    labels: list[str] = []

    def match(name: str, seq: list[int]) -> None:
        if perm == seq or inv == seq:
            labels.append(name)

    match("identity", list(range(n)))
    match("reverse", list(range(n - 1, -1, -1)))
    for k in range(1, n):
        seq = [(i + k) % n for i in range(n)]
        if perm == seq or inv == seq:
            labels.append(f"rotate-{k}")
            break
    evens = [i for i in range(n) if i % 2 == 0]
    odds = [i for i in range(n) if i % 2 == 1]
    match("evens-then-odds", evens + odds)
    match("odds-then-evens", odds + evens)
    # out-faro (riffle): interleave the two halves, top card stays on top.
    half = (n + 1) // 2
    top, bot = list(range(half)), list(range(half, n))
    riffle: list[int] = []
    for a, b in zip(top, bot, strict=False):
        riffle += [a, b]
    if len(top) > len(bot):
        riffle.append(top[-1])
    match("out-riffle", riffle)
    return labels
