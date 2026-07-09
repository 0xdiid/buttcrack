"""Complete columnar transposition.

Key may be a keyword (columns ordered by the alphabetical rank of its letters)
or an explicit 0-based read order such as ``1,2,0``. ``crack`` reports the
numeric read order, which feeds straight back into ``decode``.
"""

from __future__ import annotations

import random
import time
from itertools import permutations

from .. import search
from ..result import Candidate
from ..scoring import NgramScorer
from ..text import only_letters
from .base import Cipher

#: 8! = 40320 permutations is the ceiling for exhaustive column-order search; wider
#: keys are recovered by simulated annealing instead (the order has a real n-gram
#: gradient — swapping toward the right order improves the score — so SA converges
#: where brute force can't).
BRUTE_MAX_WIDTH = 8


def _read_order(key: str) -> list[int]:
    s = str(key).strip()
    if s and all(ch.isdigit() or ch in ", " for ch in s):
        order = [int(x) for x in s.replace(",", " ").split()]
    else:
        letters = only_letters(s)
        if not letters:
            raise ValueError("columnar key must be a keyword or numeric read order")
        order = [idx for _, idx in sorted((ch, i) for i, ch in enumerate(letters))]
    if sorted(order) != list(range(len(order))):
        raise ValueError(f"columnar read order must be a permutation of 0..{len(order) - 1}")
    return order


def _column_lengths(n: int, width: int) -> list[int]:
    full_rows, extra = divmod(n, width)
    return [full_rows + (1 if c < extra else 0) for c in range(width)]


def _units(letters: str, unit: int) -> list[str]:
    """Split into transposition atoms of ``unit`` letters (the last may be short)."""
    return [letters[i : i + unit] for i in range(0, len(letters), unit)]


def _encode_units(letters: str, order: list[int], unit: int = 1) -> str:
    """Columnar transposition that moves ``unit``-letter atoms instead of single letters.

    With ``unit=1`` this is an ordinary letter columnar; with ``unit=3`` it transposes
    3-letter blocks as indivisible tokens (so trigrams survive intact) — the
    "trigraph-granular" transposition that block-transposition analysis needs and that
    no single-letter transposition primitive could express.
    """
    toks = _units(letters, unit)
    width = len(order)
    columns: list[list[str]] = [[] for _ in range(width)]
    for i, tok in enumerate(toks):
        columns[i % width].append(tok)
    return "".join("".join(columns[c]) for c in order)


def _decode_units(cipher: str, order: list[int], unit: int = 1) -> str:
    """Inverse of :func:`_encode_units`. Handles a short final atom unambiguously."""
    width = len(order)
    n = len(cipher)
    n_tok = (n + unit - 1) // unit
    tok_len = [unit] * n_tok
    if n_tok:
        tok_len[-1] = n - unit * (n_tok - 1)
    # column c (in fill order) holds token indices c, c+width, c+2*width, ...
    col_tokens: list[list[int]] = [[] for _ in range(width)]
    for t in range(n_tok):
        col_tokens[t % width].append(t)
    col_chars = [sum(tok_len[t] for t in col_tokens[c]) for c in range(width)]
    pieces: list[str] = [""] * width
    idx = 0
    for c in order:
        pieces[c] = cipher[idx : idx + col_chars[c]]
        idx += col_chars[c]
    # split each column's slice back into its atoms
    col_split: list[list[str]] = [[] for _ in range(width)]
    for c in range(width):
        s = pieces[c]
        p = 0
        for t in col_tokens[c]:
            col_split[c].append(s[p : p + tok_len[t]])
            p += tok_len[t]
    pos = [0] * width
    out = []
    for t in range(n_tok):
        c = t % width
        out.append(col_split[c][pos[c]])
        pos[c] += 1
    return "".join(out)


def _encode_letters(letters: str, order: list[int]) -> str:
    return _encode_units(letters, order, 1)


def _decode_letters(cipher: str, order: list[int]) -> str:
    return _decode_units(cipher, order, 1)


class Columnar(Cipher):
    name = "columnar"
    aliases = ("coltrans", "column")
    description = "Complete columnar transposition; key is a keyword or read order."
    key_format = "keyword (letters) or numeric read order e.g. 1,2,0"
    key_example = "CIPHER"
    complexity = 4

    # Transposition cannot preserve word spacing; operate on a clean uppercase
    # letter stream (no reflow, which would leak plaintext word lengths).
    def encode(self, text: str, key: str) -> str:
        return _encode_letters(only_letters(text), _read_order(key))

    def decode(self, text: str, key: str) -> str:
        return _decode_letters(only_letters(text), _read_order(key))

    def crack(
        self, text, scorer: NgramScorer, *, top=5, rng=None, timeout: float | None = None, **opts
    ):
        """Recover the column read-order. Widths up to 8 are searched exhaustively;
        wider keys (``--width N`` / ``--max-width M`` above 8) by simulated annealing.

        The default ``max_width`` is 7, so the default crack — and ``auto`` — stays an
        exact brute force. Pass a larger ``--width``/``--max-width`` to reach long
        keys; a column-order has an n-gram gradient (a swap toward the right order
        improves the score), so SA converges on keys far past the factorial wall.
        """
        letters = only_letters(text)
        if len(letters) < 4:
            return []
        rng = rng or random.Random()
        unit = max(1, int(opts.get("unit", 1)))
        max_width = int(opts.get("max_width", 7))
        widths = [int(opts["width"])] if opts.get("width") else range(2, max_width + 1)
        widths = [w for w in widths if 2 <= w <= len(letters)]
        deadline = (time.monotonic() + timeout) if timeout else None
        sa_remaining = sum(1 for w in widths if w > BRUTE_MAX_WIDTH)

        candidates: list[Candidate] = []
        truncated_at = None
        for width in widths:
            if deadline and time.monotonic() > deadline:
                truncated_at = width
                break
            if width <= BRUTE_MAX_WIDTH:
                for perm in permutations(range(width)):
                    if deadline and time.monotonic() > deadline:
                        truncated_at = width
                        break
                    order = list(perm)
                    plain = _decode_units(letters, order, unit)
                    candidates.append(
                        Candidate(
                            plaintext=plain,
                            cipher=self.name,
                            key=",".join(map(str, order)),
                            score=scorer.score(plain),
                            confidence=scorer.confidence(plain),
                            meta={"width": width, "unit": unit} if unit > 1 else {"width": width},
                        )
                    )
                if truncated_at is not None:
                    break
            else:
                # Split the remaining budget across the SA widths still to run, so an
                # early width can't starve the rest (recomputed each time => a width
                # that finishes early hands its slack to the next).
                sub = deadline
                if deadline is not None and sa_remaining > 0:
                    sub = time.monotonic() + (deadline - time.monotonic()) / sa_remaining
                sa_remaining -= 1
                candidates.append(_anneal_order(letters, width, scorer, rng, sub, opts, unit))
        candidates.sort(key=lambda c: c.score, reverse=True)
        out = candidates[:top]
        if truncated_at is not None and out:
            out[-1].meta["timeout_truncated_at_width"] = truncated_at
        return out


def _anneal_order(
    letters: str,
    width: int,
    scorer: NgramScorer,
    rng,
    deadline: float | None,
    opts: dict,
    unit: int = 1,
) -> Candidate:
    """Recover a wide column order by simulated annealing on the n-gram score.

    State is the read-order permutation; a move swaps two columns. The fitness is the
    *per-quadgram* score so the annealing temperature is independent of message
    length. Returns the best candidate found.
    """
    denom = max(1, len(letters) - 3)
    restarts = int(opts.get("restarts", 10))

    def fitness(order: list[int]) -> float:
        return scorer.score(_decode_units(letters, order, unit)) / denom

    def init_order() -> list[int]:
        return search.shuffled(list(range(width)), rng)

    best, _ = search.anneal(
        init=init_order,
        neighbour=search.swap_neighbour,
        score=fitness,
        rng=rng,
        restarts=restarts,
        iters_per_temp=200,
        temp0=0.3,
        cooling=0.92,
        min_temp=0.02,
        deadline=deadline,
    )
    plain = _decode_units(letters, best, unit)
    meta = {"width": width, "method": "annealing"}
    if unit > 1:
        meta["unit"] = unit
    return Candidate(
        plaintext=plain,
        cipher="columnar",
        key=",".join(map(str, best)),
        score=scorer.score(plain),
        confidence=scorer.confidence(plain),
        meta=meta,
    )
