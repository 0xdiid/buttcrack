"""Digrafid cipher (ACA, KNUTE 1960): digraphic fractionation over two squares.

The Digrafid combines Bifid/Trifid ideas into a *digraphic* fractionation cipher.
It uses two mixed 27-character alphabets (A-Z plus ``#`` as the 27th/padding
symbol) laid into a single tableau around a fixed 3x3x3 numeric bridge:

  * a HORIZONTAL alphabet written ROW-BY-ROW into a 3x9 grid, so every letter has
    a row index 1..3 and a column index 1..9;
  * a VERTICAL alphabet written COLUMN-BY-COLUMN into a 9x3 grid, so every letter
    has a row index 1..9 and a column index 1..3;
  * a fixed bridge whose value at ``[horizontalRow (1..3)][verticalCol (1..3)]``
    is ``(row - 1) * 3 + col`` (the digits 1..9 laid row-major in three panels).

ENCRYPT (per the ACA description sheet):

  1. Pick a period = number of digraphs per group.
  2. Split the plaintext into digraphs (pad a final odd letter with ``#``) and
     group them into period-length groups.
  3. For each digraph ``(L1, L2)`` build a vertical 3-digit number:
       FIRST  = column of L1 in the horizontal alphabet (1..9);
       MIDDLE = bridge[row of L1 in horizontal][col of L2 in vertical] (1..9);
       THIRD  = row of L2 in the vertical alphabet (1..9).
  4. Fractionate exactly as in Trifid within each group: read the whole group's
     top digits left-to-right, then all middle digits, then all bottom digits,
     and regroup the stream into fresh vertical triples (one per digraph).
  5. Map each new triple ``(x, y, z)`` back to a digraph: ``y`` locates a cell
     ``(br, bc)`` in the bridge; the first cipher letter is the horizontal letter
     at ``[br][x]`` and the second is the vertical letter at ``[z][bc]``.

Decryption runs the same structure in reverse: convert each ciphertext digraph to
its triple via the tableau, un-permute the Trifid fractionation within the group,
then recover each plaintext digraph from the original ``(first, middle, third)``.

KEY FORMAT
----------
Three components separated by ``/``: ``HKEYWORD/VKEYWORD/PERIOD``.

  * ``HKEYWORD`` — keyword for the horizontal alphabet. The 27-character mixed
    alphabet (deduplicated keyword letters, then the remaining A-Z, then ``#``)
    is filled ROW-BY-ROW into the 3x9 grid.
  * ``VKEYWORD`` — keyword for the vertical alphabet. Its 27-character mixed
    alphabet is filled COLUMN-BY-COLUMN into the 9x3 grid (the sequence reads
    down column 1, then column 2, then column 3).
  * ``PERIOD`` — the integer period (digraphs per group) for the fractionation.

Example: ``KEYWORD/VERTICAL/3`` reproduces the ACA worked example.
"""

from __future__ import annotations

import math
import random
import time

from ..result import Candidate
from ..scoring import NgramScorer
from .base import Cipher

ALPHABET27 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ#"


def _keyed_alphabet(keyword: str) -> str:
    """27-character mixed alphabet: deduped keyword, then remaining A-Z, then #."""
    seq: list[str] = []
    for ch in keyword.upper() + ALPHABET27:
        if ch in ALPHABET27 and ch not in seq:
            seq.append(ch)
    return "".join(seq)


def _clean(text: str) -> str:
    """Uppercase, keeping only A-Z and the # padding symbol."""
    return "".join(ch for ch in text.upper() if ("A" <= ch <= "Z") or ch == "#")


def _bridge(hrow: int, vcol: int) -> int:
    """Bridge value (1..9) at horizontal row 1..3 and vertical col 1..3."""
    return (hrow - 1) * 3 + vcol


def _bridge_rev(value: int) -> tuple[int, int]:
    """(row 1..3, col 1..3) of bridge cell holding ``value`` (1..9)."""
    return (value - 1) // 3 + 1, (value - 1) % 3 + 1


class _Tableau:
    """The Digrafid tableau: a 3x9 horizontal grid and a 9x3 vertical grid."""

    def __init__(self, h_keyword: str, v_keyword: str):
        h_seq = _keyed_alphabet(h_keyword)
        v_seq = _keyed_alphabet(v_keyword)

        # Horizontal 3x9, filled row-by-row.
        self.h_pos: dict[str, tuple[int, int]] = {}
        self.h_grid: dict[tuple[int, int], str] = {}
        for i, ch in enumerate(h_seq):
            row, col = i // 9 + 1, i % 9 + 1
            self.h_pos[ch] = (row, col)
            self.h_grid[(row, col)] = ch

        # Vertical 9x3, filled column-by-column (sequence reads down each column).
        self.v_pos: dict[str, tuple[int, int]] = {}
        self.v_grid: dict[tuple[int, int], str] = {}
        for i, ch in enumerate(v_seq):
            col, row = i // 9 + 1, i % 9 + 1
            self.v_pos[ch] = (row, col)
            self.v_grid[(row, col)] = ch

    def encode_triple(self, digraph: str) -> tuple[int, int, int]:
        l1, l2 = digraph[0], digraph[1]
        hrow1, hcol1 = self.h_pos[l1]
        vrow2, vcol2 = self.v_pos[l2]
        return hcol1, _bridge(hrow1, vcol2), vrow2

    def output_digraph(self, triple: tuple[int, int, int]) -> str:
        x, y, z = triple
        br, bc = _bridge_rev(y)
        return self.h_grid[(br, x)] + self.v_grid[(z, bc)]

    def input_triple(self, digraph: str) -> tuple[int, int, int]:
        c1, c2 = digraph[0], digraph[1]
        br, x = self.h_pos[c1]
        z, bc = self.v_pos[c2]
        return x, _bridge(br, bc), z

    def decode_digraph(self, triple: tuple[int, int, int]) -> str:
        first, middle, third = triple
        hrow1, vcol2 = _bridge_rev(middle)
        return self.h_grid[(hrow1, first)] + self.v_grid[(third, vcol2)]


def _to_digraphs(letters: str) -> list[str]:
    s = letters
    if len(s) % 2:
        s = s + "#"
    return [s[i : i + 2] for i in range(0, len(s), 2)]


def _encode_digraphs(digraphs: list[str], tab: _Tableau, period: int) -> str:
    out: list[str] = []
    for g in range(0, len(digraphs), period):
        group = digraphs[g : g + period]
        triples = [tab.encode_triple(d) for d in group]
        tops = [t[0] for t in triples]
        mids = [t[1] for t in triples]
        bots = [t[2] for t in triples]
        seq = tops + mids + bots
        for i in range(len(group)):
            out.append(tab.output_digraph((seq[i * 3], seq[i * 3 + 1], seq[i * 3 + 2])))
    return "".join(out)


def _decode_digraphs(digraphs: list[str], tab: _Tableau, period: int) -> str:
    out: list[str] = []
    for g in range(0, len(digraphs), period):
        group = digraphs[g : g + period]
        new_triples = [tab.input_triple(d) for d in group]
        seq: list[int] = []
        for t in new_triples:
            seq.extend(t)
        n = len(group)
        tops = seq[0:n]
        mids = seq[n : 2 * n]
        bots = seq[2 * n : 3 * n]
        for i in range(n):
            out.append(tab.decode_digraph((tops[i], mids[i], bots[i])))
    return "".join(out)


def _parse_key(key: str) -> tuple[str, str, int]:
    """Split ``HKEYWORD/VKEYWORD/PERIOD``."""
    s = str(key)
    parts = s.split("/")
    if len(parts) != 3:
        raise ValueError(
            "digrafid key must be 'HKEYWORD/VKEYWORD/PERIOD', e.g. 'KEYWORD/VERTICAL/3'"
        )
    h_kw, v_kw, per = parts
    per = per.strip()
    if not per.isdigit() or int(per) < 1:
        raise ValueError(
            "digrafid period must be a positive integer (key 'HKEYWORD/VKEYWORD/PERIOD')"
        )
    return h_kw, v_kw, int(per)


class Digrafid(Cipher):
    """Digrafid digraphic fractionation over two 27-letter squares (ACA).

    KEY FORMAT: ``HKEYWORD/VKEYWORD/PERIOD`` (slash-separated). ``HKEYWORD`` keys
    the horizontal 3x9 alphabet (filled row-by-row), ``VKEYWORD`` keys the
    vertical 9x3 alphabet (filled column-by-column), and ``PERIOD`` is the number
    of digraphs per fractionation group. Both alphabets are 27 characters (A-Z
    plus ``#``, which also pads odd-length plaintext). Example:
    ``KEYWORD/VERTICAL/3``.
    """

    name = "digrafid"
    description = (
        "Digrafid digraphic fractionation over two 27-letter squares; "
        "key 'HKEYWORD/VKEYWORD/PERIOD'."
    )
    key_format = "hkeyword/vkeyword/period (two keywords + positive integer period)"
    key_example = "KEYWORD/VERTICAL/3"
    complexity = 7

    def encode(self, text: str, key: str) -> str:
        h_kw, v_kw, period = _parse_key(key)
        tab = _Tableau(h_kw, v_kw)
        return _encode_digraphs(_to_digraphs(_clean(text)), tab, period)

    def decode(self, text: str, key: str) -> str:
        h_kw, v_kw, period = _parse_key(key)
        tab = _Tableau(h_kw, v_kw)
        return _decode_digraphs(_to_digraphs(_clean(text)), tab, period)

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
        """Keyless best-effort: nest a period search outside joint annealing of
        both 27-letter alphabets.

        For each candidate period, two mixed alphabets are recovered by simulated
        annealing against the quadgram score of the decrypt, perturbing one
        alphabet per step (the ACA-recommended approach; ~120-220 letters needed).
        This is the hardest cipher of the set: the joint 27!^2 keyspace makes full
        recovery unreliable, so the routine returns the best decrypts it finds and
        honors ``timeout`` via a ``time.monotonic()`` deadline. Returns ``[]`` for
        inputs too short to fingerprint.
        """
        letters = _clean(text)
        if len(letters) < 60:
            return []
        digraphs = _to_digraphs(letters)
        rng = rng or random.Random()
        deadline = (time.monotonic() + timeout) if timeout else None

        periods = opts.get("periods")
        if periods is None:
            max_p = min(int(opts.get("max_period", 7)), len(digraphs))
            periods = list(range(2, max_p + 1))
        restarts = int(opts.get("restarts", 2))
        temp0 = float(opts.get("temp", 8.0))
        step = float(opts.get("temp_step", 0.5))
        iters = int(opts.get("iters", 800))

        base = list(ALPHABET27)

        def decrypt_with(h_seq: list[str], v_seq: list[str], period: int) -> str:
            tab = _Tableau("".join(h_seq), "".join(v_seq))
            # _Tableau rebuilds from keyword fills, but a full 27-char string is a
            # valid "keyword": dedup leaves it unchanged, so the grids match.
            return _decode_digraphs(digraphs, tab, period)

        candidates: list[Candidate] = []
        for period in periods:
            if deadline and time.monotonic() > deadline:
                break
            best_h = base[:]
            best_v = base[:]
            best_score = float("-inf")
            for _ in range(restarts):
                if deadline and time.monotonic() > deadline:
                    break
                parent_h = base[:]
                parent_v = base[:]
                rng.shuffle(parent_h)
                rng.shuffle(parent_v)
                cur = scorer.score(decrypt_with(parent_h, parent_v, period))
                temp = temp0
                while temp > 0:
                    if deadline and time.monotonic() > deadline:
                        break
                    for _ in range(iters):
                        child_h = parent_h[:]
                        child_v = parent_v[:]
                        if rng.random() < 0.5:
                            i, j = rng.randrange(27), rng.randrange(27)
                            child_h[i], child_h[j] = child_h[j], child_h[i]
                        else:
                            i, j = rng.randrange(27), rng.randrange(27)
                            child_v[i], child_v[j] = child_v[j], child_v[i]
                        s = scorer.score(decrypt_with(child_h, child_v, period))
                        delta = s - cur
                        if delta > 0 or rng.random() < math.exp(delta / temp):
                            parent_h, parent_v, cur = child_h, child_v, s
                            if s > best_score:
                                best_h, best_v, best_score = child_h[:], child_v[:], s
                    temp -= step
            h_str = "".join(best_h)
            v_str = "".join(best_v)
            plain = decrypt_with(best_h, best_v, period)
            candidates.append(
                Candidate(
                    plaintext=plain,
                    cipher=self.name,
                    key=f"{h_str}/{v_str}/{period}",
                    score=best_score,
                    confidence=scorer.confidence(plain),
                    meta={"period": period, "h_square": h_str, "v_square": v_str},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top]
