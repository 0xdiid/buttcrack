"""Chaocipher (John F. Byrne, 1918) — two alphabets that permute after every letter.

``identify`` has always been able to name this cipher as a failure mode: it is the
worked example in ``identify.py`` of a keystream whose period is not recoverable,
because the alphabets never repeat a state. Until now the tool could name it and not
apply it, so a Chaocipher hypothesis could be neither confirmed nor encoded.

ALGORITHM
---------
Two 26-letter alphabets: LEFT (ciphertext) and RIGHT (plaintext).

To encipher one letter:

1. Find the plaintext letter at index ``i`` of RIGHT; the ciphertext letter is
   ``LEFT[i]``.
2. Permute LEFT: rotate it so ``LEFT[i]`` sits at position 0 (the "zenith"), then take
   the letter now at position 1 out and re-insert it at position 13 (the "nadir"),
   sliding positions 2..13 down one.
3. Permute RIGHT: rotate it so ``RIGHT[i]`` sits at position 0, rotate one place
   further, then take the letter at position 2 out and re-insert it at position 13,
   sliding positions 3..13 down one.

Deciphering finds the ciphertext letter in LEFT and reads RIGHT at the same index; the
permutation step is identical, so both directions stay in lockstep.

KEY FORMAT
----------
``LEFT/RIGHT`` — two 26-letter permutations, e.g.
``HXUCZVAMDSLKPEFJRIGTWOBNYQ/PTLNBQDEOYSFAVZKGJRIHWXUMC``.

Reference: Moshe Rubin, "Chaocipher Revealed: The Algorithm" (2010). Exhibit 1 of the
Byrne papers enciphers ``WELLDONEISBETTERTHANWELLSAID`` to
``OAHQHCNYNXTSZJRRHJBYHQKSOUJY`` under the key above.
"""

from __future__ import annotations

from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

ZENITH = 0
NADIR = 13


def _parse_key(key: str) -> tuple[list[str], list[str]]:
    left, sep, right = str(key).partition("/")
    if not sep:
        raise ValueError("chaocipher key must be 'LEFT/RIGHT' (two 26-letter alphabets)")
    out = []
    for name, half in (("left", left), ("right", right)):
        letters = only_letters(half).upper()
        if len(letters) != 26 or len(set(letters)) != 26:
            raise ValueError(f"chaocipher {name} alphabet must be a 26-letter permutation of A-Z")
        out.append(list(letters))
    return out[0], out[1]


def _permute_left(alphabet: list[str], index: int) -> list[str]:
    """Rotate the ciphertext letter to the zenith, then move zenith+1 to the nadir."""
    rotated = alphabet[index:] + alphabet[:index]
    moved = rotated.pop(ZENITH + 1)
    rotated.insert(NADIR, moved)
    return rotated


def _permute_right(alphabet: list[str], index: int) -> list[str]:
    """As the left, but shifted one further and taking zenith+2 rather than zenith+1."""
    rotated = alphabet[index:] + alphabet[:index]
    rotated = rotated[1:] + rotated[:1]
    moved = rotated.pop(ZENITH + 2)
    rotated.insert(NADIR, moved)
    return rotated


def _run(text: str, left: list[str], right: list[str], *, decrypt: bool) -> str:
    # Look-up alphabet and output alphabet swap between the two directions; the
    # permutation step does not, which is what keeps the machines synchronised.
    out: list[str] = []
    for ch in only_letters(text).upper():
        source, target = (left, right) if decrypt else (right, left)
        try:
            index = source.index(ch)
        except ValueError:  # pragma: no cover - only_letters guarantees A-Z
            continue
        out.append(target[index])
        left = _permute_left(left, left.index(out[-1] if not decrypt else ch))
        right = _permute_right(right, right.index(ch if not decrypt else out[-1]))
    return "".join(out)


class Chaocipher(Cipher):
    name = "chaocipher"
    aliases = ("chao",)
    description = "Byrne's two dynamic alphabets, permuted after every letter enciphered."
    key_format = "two 26-letter permutations, 'LEFT/RIGHT'"
    key_example = "HXUCZVAMDSLKPEFJRIGTWOBNYQ/PTLNBQDEOYSFAVZKGJRIHWXUMC"
    complexity = 9
    # No keyless attack exists; see `crack`. Running it under `auto` would only burn
    # the time budget of the ciphers that can be solved.
    auto_crackable = False

    def encode(self, text: str, key: str) -> str:
        left, right = _parse_key(key)
        return _run(text, left, right, decrypt=False)

    def decode(self, text: str, key: str) -> str:
        left, right = _parse_key(key)
        return _run(text, left, right, decrypt=True)

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ) -> list[Candidate]:
        """Returns nothing, deliberately — there is no keyless attack to run.

        The keyspace is two independent 26-letter permutations (~2^177), and unlike
        every other square or alphabet in this package it cannot be climbed: the
        alphabets permute after each letter, so one wrong cell corrupts the entire
        remaining decrypt rather than one position of it. The n-gram score of a nearly
        correct key is therefore indistinguishable from that of a random one — there is
        no gradient for hill-climbing or annealing to follow.

        Published cryptanalysis (Deavours and Kruh) is a *known-plaintext* attack
        needing on the order of 50-100 aligned characters. Returning ``[]`` is the
        honest result; a search here would spend the budget and report noise.
        """
        return []
