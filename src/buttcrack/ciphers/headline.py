"""Headline cipher (ACA "HEADLINES": K3 keyed simple substitution at a setting).

The ACA *Headlines* puzzle presents five newspaper headlines, all enciphered by
simple substitution with the *same* mixed alphabet, but each headline uses a
different *setting* (a rotation of that mixed alphabet) -- this is the K3 keying
idea, the mixed alphabet substituted against a rotation of itself.

Three keywords drive the construction:

* **Hat** -- a keyword that fixes a columnar-transposition column order
  (the alphabetical rank of its letters, ties broken left to right). Its length
  is the block width.
* **Key** -- a keyword that builds a keyed alphabet (deduped keyword followed by
  the unused A-Z letters). The keyed alphabet is written row-wise into a block
  of ``len(Hat)`` columns; reading the columns off top-to-bottom in the Hat's
  column order produces the 26-letter **mixed alphabet**.
* **Setting** -- a keyword whose letters give the starting letter of each
  headline's cipher row. For headline ``n`` the cipher row is the mixed alphabet
  rotated so it begins at the ``n``-th setting letter.

Encipherment of one headline (a single letter stream) therefore uses a single
setting letter ``s``: the plaintext header row is the mixed alphabet, the cipher
row is the mixed alphabet rotated to start at ``s``, and a plaintext letter ``P``
is enciphered to the cipher-row letter sitting directly under ``P`` in the
header (i.e. at ``P``'s position in the mixed alphabet).

KEY FORMAT (one ``--key`` string, slash-separated)::

    "HAT/KEY/SETTING"

``HAT`` and ``KEY`` are keywords. ``SETTING`` is the setting for *this* headline:
a single letter that the cipher row starts at (e.g. ``D``). For convenience a
multi-letter setting word may be given with a 1-based row index appended, as in
``"APOTHECARY/CHEMIST/DRUGS:3"`` (headline 3 -> starts at the 3rd letter ``U``);
without an index the first setting letter is used.

This is a monoalphabetic substitution per headline, so it is reciprocal only in
the trivial identity case; encode/decode are inverses but not the same map.
"""

from __future__ import annotations

import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import ALPHABET, only_letters, reflow
from .base import Cipher


def _keyed_alphabet(keyword: str) -> str:
    """Deduped keyword letters followed by the remaining A-Z letters in order."""
    seq: list[str] = []
    for ch in keyword.upper():
        if "A" <= ch <= "Z" and ch not in seq:
            seq.append(ch)
    for ch in ALPHABET:
        if ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _column_order(keyword: str) -> list[int]:
    """``order[col]`` = 0-based alphabetical rank of that column (ties L-to-R)."""
    kw = keyword.upper()
    ranked = sorted(range(len(kw)), key=lambda i: (kw[i], i))
    order = [0] * len(kw)
    for rank, idx in enumerate(ranked):
        order[idx] = rank
    return order


def _mixed_alphabet(hat: str, key: str) -> str:
    """Mixed alphabet: KEY's keyed alphabet read off in HAT's column order.

    Write the keyed alphabet of ``key`` row-wise into a block whose width is the
    number of letters in ``hat``, then read columns top-to-bottom in the
    alphabetical-rank order of ``hat`` (the columnar "Hat" transposition).
    """
    hat_letters = only_letters(hat)
    key_letters = only_letters(key)
    if not hat_letters:
        raise ValueError("headline hat keyword must contain letters")
    if not key_letters:
        raise ValueError("headline key keyword must contain letters")
    width = len(hat_letters)
    keyed = _keyed_alphabet(key_letters)
    rows = [keyed[i : i + width] for i in range(0, len(keyed), width)]
    order = _column_order(hat_letters)
    cols_by_rank = sorted(range(width), key=lambda c: order[c])
    out: list[str] = []
    for col in cols_by_rank:
        for row in rows:
            if col < len(row):
                out.append(row[col])
    return "".join(out)


def _parse_key(key: str) -> tuple[str, str, str]:
    """Parse ``"HAT/KEY/SETTING"`` into (hat, key, setting_letter).

    ``SETTING`` is either a single letter or a setting word with an optional
    ``:n`` 1-based row index (``"DRUGS:3"`` -> 3rd letter); default index 1.
    """
    parts = str(key).split("/")
    if len(parts) < 3:
        raise ValueError("headline key must be 'HAT/KEY/SETTING'")
    hat = only_letters(parts[0])
    key_kw = only_letters(parts[1])
    setting_part = parts[2]
    index = 1
    if ":" in setting_part:
        word, _, idx_str = setting_part.partition(":")
        idx_str = idx_str.strip()
        if not idx_str.isdigit():
            raise ValueError("headline setting index must be a positive integer")
        index = int(idx_str)
        setting_word = only_letters(word)
    else:
        setting_word = only_letters(setting_part)
    if not hat:
        raise ValueError("headline hat keyword must contain letters")
    if not key_kw:
        raise ValueError("headline key keyword must contain letters")
    if not setting_word:
        raise ValueError("headline setting must contain at least one letter")
    if index < 1 or index > len(setting_word):
        raise ValueError(
            f"headline setting index {index} out of range for setting {setting_word!r}"
        )
    return hat, key_kw, setting_word[index - 1]


def _cipher_row(mixed: str, setting_letter: str) -> str:
    """Mixed alphabet rotated to begin at ``setting_letter``."""
    idx = mixed.index(setting_letter)
    return mixed[idx:] + mixed[:idx]


def _encode_letters(letters: str, mixed: str, cipher_row: str) -> str:
    pos = {ch: i for i, ch in enumerate(mixed)}
    return "".join(cipher_row[pos[ch]] for ch in letters)


def _decode_letters(letters: str, mixed: str, cipher_row: str) -> str:
    cpos = {ch: i for i, ch in enumerate(cipher_row)}
    return "".join(mixed[cpos[ch]] for ch in letters)


class Headline(Cipher):
    name = "headline"
    aliases = ("headlines",)
    description = "ACA Headlines: K3 keyed simple substitution at a chosen setting."
    key_format = "HAT/KEY/SETTING ('/'-separated keywords; SETTING is a letter or word[:n])"
    key_example = "APOTHECARY/CHEMIST/DRUGS"
    complexity = 6

    def encode(self, text: str, key: str) -> str:
        hat, key_kw, setting_letter = _parse_key(key)
        mixed = _mixed_alphabet(hat, key_kw)
        cipher_row = _cipher_row(mixed, setting_letter)
        return _encode_letters(only_letters(text), mixed, cipher_row)

    def decode(self, text: str, key: str) -> str:
        hat, key_kw, setting_letter = _parse_key(key)
        mixed = _mixed_alphabet(hat, key_kw)
        cipher_row = _cipher_row(mixed, setting_letter)
        return reflow(text, _decode_letters(only_letters(text), mixed, cipher_row))

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
        """Best-effort crack of a single Headline (one setting).

        A single headline at one setting is just a monoalphabetic substitution
        whose cipher alphabet happens to be a rotated mixed alphabet. With the
        keywords unknown we cannot reconstruct Hat/Key/Setting from one short
        headline, but we *can* recover the plaintext: every reachable cipher
        alphabet is some 26-letter permutation, so we hill-climb a general
        substitution map on the quadgram score (identical machinery to the
        ``substitution`` solver). We return the recovered plaintext; we do not
        report a Hat/Key/Setting key because many keyword triples yield the same
        alphabet and they are not identifiable from a single headline.
        """
        rng = rng or random.Random()
        letters = only_letters(text)
        if len(letters) < 12:
            return []
        deadline = (time.monotonic() + timeout) if timeout else None
        restarts = int(opts.get("restarts", 20))

        def decrypt(dec: list[str]) -> str:
            return "".join(dec[ord(c) - 65] for c in letters)

        results: list[tuple[float, list[str]]] = []
        for r in range(restarts):
            if deadline and time.monotonic() > deadline:
                break
            parent = _freq_seed(letters) if r == 0 else _shuffled_dec(rng)
            parent_score = scorer.score(decrypt(parent))
            improved = True
            timed_out = False
            while improved and not timed_out:
                improved = False
                for i in range(25):
                    if deadline and time.monotonic() > deadline:
                        timed_out = True
                        break
                    for j in range(i + 1, 26):
                        child = parent[:]
                        child[i], child[j] = child[j], child[i]
                        s = scorer.score(decrypt(child))
                        if s > parent_score:
                            parent_score, parent, improved = s, child, True
            results.append((parent_score, parent))

        results.sort(key=lambda rs: rs[0], reverse=True)
        seen: set[str] = set()
        candidates: list[Candidate] = []
        for score, dec in results:
            plain = decrypt(dec)
            if plain in seen:
                continue
            seen.add(plain)
            candidates.append(
                Candidate(
                    plaintext=reflow(text, plain),
                    cipher=self.name,
                    key=None,
                    score=score,
                    confidence=scorer.confidence(plain),
                    meta={"restarts": restarts},
                )
            )
            if len(candidates) >= top:
                break
        return candidates


# --- crack internals -----------------------------------------------------

_ENGLISH_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def _freq_seed(letters: str) -> list[str]:
    """Seed a decrypt map (cipher index -> plain) by frequency alignment."""
    from collections import Counter

    counts = Counter(letters)
    cipher_by_freq = sorted((chr(65 + i) for i in range(26)), key=lambda c: -counts.get(c, 0))
    dec = ["A"] * 26
    for cipher_letter, plain_letter in zip(cipher_by_freq, _ENGLISH_ORDER, strict=True):
        dec[ord(cipher_letter) - 65] = plain_letter
    return dec


def _shuffled_dec(rng: random.Random) -> list[str]:
    letters = [chr(65 + i) for i in range(26)]
    rng.shuffle(letters)
    return letters
