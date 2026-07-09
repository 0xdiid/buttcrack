"""Syllabary cipher (ACA): homophonic substitution onto a stream of digit pairs.

The Syllabary cipher (G-MAN, *The Cryptogram* 2012; after Friedman & Callimahos,
*Military Cryptanalytics* Part I, 1956) uses a 10x10 "syllabary square" whose 100
cells hold the 26 letters, the 10 digits, and 64 frequent digraphs/trigraphs. The
plaintext is parsed into those tokens and each token is replaced by the two-digit
``(row)(column)`` coordinate of its cell, so a single plaintext can be spelled
many ways (its *isologs*) -- this suppresses letter frequencies and word patterns.

STANDARD SQUARE (unmixed), rows/cols numbered 0-9 (ACA appendix order)::

        0    1    2    3    4    5    6    7    8    9
    0   A    1    AL   AN   AND  AR   ARE  AS   AT   ATE
    1   ATI  B    2    BE   C    3    CA   CE   CO   COM
    2   D    4    DA   DE   E    5    EA   ED   EN   ENT
    3   ER   ERE  ERS  ES   EST  F    6    G    7    H
    4   8    HAS  HE   I    9    IN   ING  ION  IS   IT
    5   IVE  J    0    K    L    LA   LE   M    ME   N
    6   ND   NE   NT   O    OF   ON   OR   OU   P    Q
    7   R    RA   RE   RED  RES  RI   RO   S    SE   SH
    8   ST   STO  T    TE   TED  TER  TH   THE  THI  THR
    9   TI   TO   U    V    VE   W    WE   X    Y    Z

KEY FORMAT
    Up to three whitespace- or ``/``-separated fields::

        "[keyword] [leftkey] [topkey]"

    * ``keyword`` -- mixes the square. The keyword is split into syllabary tokens
      (longest-match) and they are placed first, row by row, followed by the
      remaining standard tokens in order. Use ``-`` (or omit) for the standard
      unmixed square.
    * ``leftkey`` -- the 10 row-coordinate digits, top to bottom (a permutation of
      ``0123456789``). Defaults to ``0123456789``.
    * ``topkey``  -- the 10 column-coordinate digits, left to right. Defaults to
      ``0123456789``.

    The three ACA variants are expressed by which fields differ from the default:
    keyword only (Unknown Keysquare), coordinate keys only (Unknown Coordinates),
    or both. An empty key is the standard square with sequential coordinates.

ENCRYPT parses the plaintext greedily into the longest available token at each
position and emits the coordinate pair of each. DECRYPT reads digit pairs, maps
each through the coordinate keys back to a cell, and emits that cell's token.
Because encryption is one-to-many (a token's spelling is a choice), the cipher is
not reciprocal; round-tripping uses this module's fixed greedy parse.
"""

from __future__ import annotations

import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

#: The 100 cells of the standard unmixed syllabary square, in row-major order.
STANDARD_TOKENS: tuple[str, ...] = (
    "A",
    "1",
    "AL",
    "AN",
    "AND",
    "AR",
    "ARE",
    "AS",
    "AT",
    "ATE",
    "ATI",
    "B",
    "2",
    "BE",
    "C",
    "3",
    "CA",
    "CE",
    "CO",
    "COM",
    "D",
    "4",
    "DA",
    "DE",
    "E",
    "5",
    "EA",
    "ED",
    "EN",
    "ENT",
    "ER",
    "ERE",
    "ERS",
    "ES",
    "EST",
    "F",
    "6",
    "G",
    "7",
    "H",
    "8",
    "HAS",
    "HE",
    "I",
    "9",
    "IN",
    "ING",
    "ION",
    "IS",
    "IT",
    "IVE",
    "J",
    "0",
    "K",
    "L",
    "LA",
    "LE",
    "M",
    "ME",
    "N",
    "ND",
    "NE",
    "NT",
    "O",
    "OF",
    "ON",
    "OR",
    "OU",
    "P",
    "Q",
    "R",
    "RA",
    "RE",
    "RED",
    "RES",
    "RI",
    "RO",
    "S",
    "SE",
    "SH",
    "ST",
    "STO",
    "T",
    "TE",
    "TED",
    "TER",
    "TH",
    "THE",
    "THI",
    "THR",
    "TI",
    "TO",
    "U",
    "V",
    "VE",
    "W",
    "WE",
    "X",
    "Y",
    "Z",
)

#: Tokens longest-first, for greedy maximal-munch parsing of a letter/digit stream.
_TOKENS_BY_LEN: tuple[str, ...] = tuple(sorted(STANDARD_TOKENS, key=lambda tok: -len(tok)))


def _clean(text: str) -> str:
    """Uppercase, keeping only A-Z and 0-9 (the syllabary alphabet)."""
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _tokenize(stream: str) -> list[str]:
    """Greedily split a clean letter/digit stream into syllabary tokens."""
    out: list[str] = []
    i = 0
    n = len(stream)
    while i < n:
        for tok in _TOKENS_BY_LEN:
            if stream.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                break
        else:  # pragma: no cover - every single letter/digit is a token
            raise ValueError(f"no syllabary token at {stream[i:]!r}")
    return out


def _parse_digits(field: str, *, what: str) -> list[int]:
    """Parse a 10-digit coordinate key into a list of ints (a 0-9 permutation)."""
    digits = [int(ch) for ch in field if ch.isdigit()]
    if len(digits) != 10 or sorted(digits) != list(range(10)):
        raise ValueError(f"syllabary {what} must be a permutation of 0123456789")
    return digits


class _Square:
    """A parsed syllabary key: the mixed square plus coordinate digit maps."""

    def __init__(self, key: str):
        parts = str(key).replace("/", " ").split()
        keyword = parts[0] if parts and parts[0] not in ("-", "") else ""
        leftkey = parts[1] if len(parts) > 1 else "0123456789"
        topkey = parts[2] if len(parts) > 2 else "0123456789"

        tokens: list[str] = []
        seen: set[str] = set()
        for tok in _tokenize(_clean(keyword)):
            if tok not in seen:
                tokens.append(tok)
                seen.add(tok)
        for tok in STANDARD_TOKENS:
            if tok not in seen:
                tokens.append(tok)
                seen.add(tok)

        self.tokens: tuple[str, ...] = tuple(tokens)  # 100 tokens, row-major
        self.pos: dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.left: list[int] = _parse_digits(leftkey, what="leftkey")
        self.top: list[int] = _parse_digits(topkey, what="topkey")
        self.inv_left: dict[int, int] = {d: i for i, d in enumerate(self.left)}
        self.inv_top: dict[int, int] = {d: i for i, d in enumerate(self.top)}

    def code(self, token: str) -> str:
        row, col = divmod(self.pos[token], 10)
        return f"{self.left[row]}{self.top[col]}"

    def token(self, pair: str) -> str:
        row = self.inv_left[int(pair[0])]
        col = self.inv_top[int(pair[1])]
        return self.tokens[row * 10 + col]


class Syllabary(Cipher):
    name = "syllabary"
    aliases = ("syllabary-square",)
    description = "Homophonic substitution onto two-digit codes from a 10x10 syllabary square."
    key_format = "keyword [leftkey] [topkey] (space/'/'-separated; coord keys are 0-9 perms)"
    key_example = "CRYPTO 1234567890 0987654321"
    complexity = 5

    def encode(self, text: str, key: str) -> str:
        square = _Square(key)
        codes = [square.code(tok) for tok in _tokenize(_clean(text))]
        return " ".join(codes)

    def decode(self, text: str, key: str) -> str:
        square = _Square(key)
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        out: list[str] = []
        for i in range(0, len(digits) - 1, 2):
            out.append(square.token(digits[i : i + 2]))
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
        """Not implemented as a keyless solver.

        Recovering a Syllabary requires jointly searching a 100-cell keyword
        square and two 10-digit coordinate permutations while the variant-spelling
        (isolog) parse of the plaintext is itself unknown -- an enormous keyspace
        that the ACA recommends attacking with a dictionary/crib search this
        package does not carry. We return no candidates rather than a misleading
        guess.
        """
        _ = (text, scorer, top, rng, opts)
        deadline = None if timeout is None else time.monotonic() + timeout
        _ = deadline
        return []
