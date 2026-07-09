"""Blind recovery of a 3x3 Hill cipher (optionally affine) by row decomposition.

A keyless 3x3 Hill attack is usually called hopeless: the key space is ``26**9`` and a
hill-climb has no gradient because a single wrong entry scrambles every block. This
module sidesteps the whole matrix search by a decomposition that *does* have a gradient.

The idea
--------
Decryption is ``p = D c`` on 3-letter column blocks. Row ``i`` of ``D`` acts
**independently**: plaintext coordinate ``i`` is the single linear form
``p_i[j] = D[i] . c[j] (mod 26)`` over ciphertext blocks ``j``. So a candidate *row* is
just a 3-covector, and the row space has only ``26**3`` points — ``1471`` once we quotient
by the invertible scalar that leaves the recovered stream a mono-substitution of itself.
Every covector is enumerable, and a *partially* correct key shows up as one or two
high-scoring rows: that is the gradient.

Scoring a row
-------------
``D[i] . c`` is a decimated English stream: it has English single-letter statistics but
never reads contiguously (it is one coordinate out of three). We score it by the shape of
its letter distribution against English, invariant to the two nuisance freedoms:

* an unknown **scalar** ``u`` (coprime to 26) — the row is only defined up to scale, and
  ``u.p`` is a bin-permutation of ``p``; we minimise over the 12 units, which is exactly
  what disambiguates the scale (a chi-square against the *asymmetric* English profile is
  NOT scale-invariant, so the true scale wins);
* an unknown **additive** offset per block-class — this covers a plain Hill (offset 0) and
  the affine/periodic-additive generalisation (a keyword added to the plaintext before the
  Hill step, giving a per-class shift of period ``q``); we minimise chi-square over the 26
  shifts *within each class*.

Assembling
----------
Rank rows by that chi-square, take the strongest ``top_rows``, and try every invertible
3x3 built from three of them (in all position assignments). For each, resolve the
scalar+shift of every row, interleave the three recovered coordinate streams, and score
the whole thing with quadgrams — the final, unambiguous arbiter that also pins the scale
each stream should take. The genuine matrix wins by a wide quadgram margin.

Validated end to end on a keyed-alphabet + periodic-additive construction recovered from
153 letters (see ``tests/test_hill_recover.py``): true rows rank #0/#1/#15 of 1471.

Public API
----------
``recover(ciphertext, scorer, *, alphabet, q_values, top, top_rows, pair_brute) -> list``
    ranked :class:`Recovered` hypotheses (decrypt matrix, alphabet, offsets, plaintext).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..scoring import ENGLISH_MONOGRAM_FREQ, NgramScorer
from ..text import only_letters

# Residues coprime to 26 (the invertible scalars / "units").
UNITS = tuple(u for u in range(1, 26) if math.gcd(u, 26) == 1)
_INV_UNIT = {u: pow(u, -1, 26) for u in UNITS}


def _projective_rows() -> list[tuple[int, int, int]]:
    """Every non-zero 3-covector, one representative per invertible-scalar class."""
    reps: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for a in range(26):
        for b in range(26):
            for c in range(26):
                if a == 0 and b == 0 and c == 0:
                    continue
                g = (a, b, c)
                if g in seen:
                    continue
                orbit = {((u * a) % 26, (u * b) % 26, (u * c) % 26) for u in UNITS}
                seen |= orbit
                reps.append(min(orbit))
    return reps


_ROWS = _projective_rows()  # ~1471 representatives, computed once


def _det3(m: list[tuple[int, int, int]]) -> int:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)) % 26


@dataclass
class Recovered:
    """One recovered construction hypothesis."""

    plaintext: str
    decrypt_matrix: list[tuple[int, int, int]]  # rows, in the KRYPTOS/STD index of `alphabet`
    alphabet: str
    q: int  # additive-schedule period at the block level (1 = plain Hill)
    offsets: list[list[int]]  # offsets[row][class]; all zero for a plain Hill
    score: float
    rows_used: tuple[int, int, int] = (0, 0, 0)  # ranks of the three rows
    meta: dict = field(default_factory=dict)

    @property
    def is_plain_hill(self) -> bool:
        return self.q == 1 and all(o == 0 for row in self.offsets for o in row)


def _english_weights(alphabet: str) -> list[float]:
    """``1 / freq`` in the index order of ``alphabet`` (chi-square denominator)."""
    return [1.0 / ENGLISH_MONOGRAM_FREQ[ch] for ch in alphabet]


def _blocks(indices: list[int], phase: int) -> list[tuple[int, int, int]]:
    nb = (len(indices) - phase) // 3
    return [
        (indices[phase + 3 * j], indices[phase + 3 * j + 1], indices[phase + 3 * j + 2])
        for j in range(nb)
    ]


def _row_chi(channel: list[int], q: int, winv: list[float]) -> float:
    """Additive-invariant chi-square of a candidate row's decimated stream.

    ``min`` over the 12 scalars of the mean-over-classes of the best-shift chi-square.
    """
    nb = len(channel)
    best = math.inf
    for u in UNITS:
        tot = 0.0
        for cls in range(q):
            hist = [0] * 26
            for j in range(cls, nb, q):
                hist[(u * channel[j]) % 26] += 1
            m = sum(hist)
            if m == 0:
                continue
            sq = [h * h for h in hist]
            # chi(t) = sum_l sq[l] * winv[(l - t) % 26] / m - m ; take min over t
            cmin = math.inf
            for t in range(26):
                acc = 0.0
                base = -t
                for i in range(26):
                    hl = sq[i]
                    if hl:
                        acc += hl * winv[(i + base) % 26]
                c = acc / m - m
                if c < cmin:
                    cmin = c
            tot += cmin
        best = min(best, tot / q)
    return best


def _apply_scalar(
    channel: list[int], q: int, u: int, winv: list[float]
) -> tuple[float, list[int], list[int]]:
    """For a fixed scalar ``u``, resolve the per-class additive shift by chi-square.

    Returns ``(chi, plaintext_indices, offsets)``.
    """
    nb = len(channel)
    out = [0] * nb
    offs = [0] * q
    tot = 0.0
    for cls in range(q):
        hist = [0] * 26
        for j in range(cls, nb, q):
            hist[(u * channel[j]) % 26] += 1
        m = sum(hist)
        if m == 0:
            continue
        sq = [h * h for h in hist]
        cmin, targ = math.inf, 0
        for t in range(26):
            acc = 0.0
            for i in range(26):
                if sq[i]:
                    acc += sq[i] * winv[(i - t) % 26]
            c = acc / m - m
            if c < cmin:
                cmin, targ = c, t
        offs[cls] = targ
        for j in range(cls, nb, q):
            out[j] = ((u * channel[j]) % 26 - targ) % 26
        tot += cmin
    return tot / q, out, offs


def _resolve_stream(
    channel: list[int], q: int, winv: list[float]
) -> tuple[float, int, list[int], list[int]]:
    """Best (scalar, per-class shift) for a channel by chi-square vs English.

    Returns ``(chi, scalar, plaintext_indices, offsets)``. The scalar is the noisy
    freedom at short lengths — :func:`_polish` re-decides it by quadgram later.
    """
    best: tuple[float, int, list[int] | None, list[int] | None] = (math.inf, 1, None, None)
    for u in UNITS:
        chi, out, offs = _apply_scalar(channel, q, u, winv)
        if chi < best[0]:
            best = (chi, u, out, offs)
    assert best[2] is not None and best[3] is not None  # UNITS is non-empty
    return best[0], best[1], best[2], best[3]


def _stream(channel: list[int], q: int, u: int, offs: list[int]) -> list[int]:
    return [((u * channel[j]) % 26 - offs[j % q]) % 26 for j in range(len(channel))]


def _strong(
    triple: tuple[int, int, int],
    channels: list[list[int]],
    q: int,
    winv: list[float],
    scorer: NgramScorer,
    alphabet: str,
    scalar_cap: int = 26,
) -> tuple[float, str, list[tuple[int, int, int]], list[list[int]]]:
    """Exhaustively resolve a candidate row-triple into the best readable plaintext.

    Given three covectors that we *believe* are the decrypt-matrix rows, three freedoms
    remain: which plaintext coordinate each row feeds (the 3! assignment), each row's
    invertible scalar (the projective quotient dropped it), and each row's per-class
    additive shift. The first two are searched jointly by quadgram over every assignment
    and scalar triple (each scalar seeded with monogram shifts); the shifts are then swept
    to convergence by quadgram. Cheap (~0.2s) because it runs only on shortlisted triples,
    and it recovers the exact key even when a true row was a monogram outlier.

    Returns ``(score, plaintext, decrypt_matrix, offsets)`` with the scalar folded into
    each returned matrix row, so ``p[i] = decrypt_matrix[i] . c - offsets[i][block % q]``.
    """
    import itertools

    rows = [_ROWS[r] for r in triple]
    # per (row-slot, scalar): monogram-resolved stream + offsets; keep the cheapest scalars
    smap: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    scalars_for: list[list[int]] = []
    for i, r in enumerate(triple):
        ranked = []
        for u in UNITS:
            chi, st, off = _apply_scalar(channels[r], q, u, winv)
            smap[(i, u)] = (st, off)
            ranked.append((chi, u))
        ranked.sort()
        scalars_for.append([u for _, u in ranked[:scalar_cap]])

    nb = len(channels[triple[0]])

    def interleave(s0, s1, s2):
        return "".join(alphabet[v] for j in range(nb) for v in (s0[j], s1[j], s2[j]))

    # score, perm, scalars
    best: tuple[float, tuple[int, ...] | None, tuple[int, int, int] | None] = (
        -math.inf,
        None,
        None,
    )
    for perm in itertools.permutations(range(3)):
        for u0 in scalars_for[perm[0]]:
            s0 = smap[(perm[0], u0)][0]
            for u1 in scalars_for[perm[1]]:
                s1 = smap[(perm[1], u1)][0]
                for u2 in scalars_for[perm[2]]:
                    sc = scorer.score(interleave(s0, s1, smap[(perm[2], u2)][0]))
                    if sc > best[0]:
                        best = (sc, perm, (u0, u1, u2))

    _, best_perm, us = best
    assert best_perm is not None and us is not None  # at least one permutation is scored
    slots = [best_perm[i] for i in range(3)]  # slots[pos] = which input row feeds coordinate pos
    chans = [channels[triple[slots[pos]]] for pos in range(3)]
    offs = [list(smap[(slots[pos], us[pos])][1]) for pos in range(3)]
    streams = [_stream(chans[pos], q, us[pos], offs[pos]) for pos in range(3)]

    def text():
        return interleave(streams[0], streams[1], streams[2])

    cur = scorer.score(text())
    improved = True
    while improved:
        improved = False
        for pos in range(3):
            for cls in range(q):
                base_t, keep = cur, offs[pos][cls]
                for t in range(26):
                    offs[pos][cls] = t
                    streams[pos] = _stream(chans[pos], q, us[pos], offs[pos])
                    sc = scorer.score(text())
                    if sc > base_t:
                        base_t, keep = sc, t
                offs[pos][cls] = keep
                streams[pos] = _stream(chans[pos], q, us[pos], offs[pos])
                if base_t > cur:
                    cur, improved = base_t, True

    matrix: list[tuple[int, int, int]] = []
    for pos in range(3):
        a, b, c = rows[slots[pos]]
        u = us[pos]
        matrix.append(((u * a) % 26, (u * b) % 26, (u * c) % 26))
    return cur, text(), matrix, offs


def recover(
    ciphertext: str,
    scorer: NgramScorer,
    *,
    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    q_values: tuple[int, ...] = (1,),
    top: int = 5,
    top_rows: int = 40,
    phases: tuple[int, ...] = (0,),
    pair_brute: bool = False,
    scalar_cap: int = 6,
    stop_confidence: float = 0.5,
) -> list[Recovered]:
    """Blind-recover a 3x3 (optionally affine) Hill over ``alphabet``.

    ``alphabet`` is the 26-letter index alphabet (e.g. the plain A-Z, or a keyed
    alphabet such as ``"KRYPTOSABCDEFGHIJLMNQUVWXZ"``). ``q_values`` are the additive
    schedule periods to try (``1`` = plain Hill; ``2`` = a per-parity offset; etc.).
    Returns up to ``top`` hypotheses ranked by quadgram fitness of the recovered text.

    Length note: ranking a decrypt row needs its decimated stream to look English, which
    takes samples. It is reliable from roughly 60+ trigraph blocks (~180 letters); on
    shorter text one or two true rows can rank as monogram outliers. ``pair_brute`` scans
    every row for a third given two strong seed rows, rescuing the *single*-outlier case
    (down to ~50 blocks); a two-outlier case at very short length is below the blind floor.
    """
    letters = only_letters(ciphertext)
    if len(letters) < 30:
        return []
    idx = {ch: i for i, ch in enumerate(alphabet)}
    try:
        stream = [idx[ch] for ch in letters]
    except KeyError as exc:  # a letter outside the alphabet
        raise ValueError(f"ciphertext letter {exc} not in alphabet") from exc
    winv = _english_weights(alphabet)

    results: list[Recovered] = []
    for phase in phases:
        blocks = _blocks(stream, phase)
        if len(blocks) < 10:
            continue
        c0 = [b[0] for b in blocks]
        c1 = [b[1] for b in blocks]
        c2 = [b[2] for b in blocks]
        for q in q_values:
            # 1) score every projective row by additive-invariant chi-square
            channels: list[list[int]] = []
            chis: list[float] = []
            for a, b, c in _ROWS:
                ch = [(a * c0[j] + b * c1[j] + c * c2[j]) % 26 for j in range(len(blocks))]
                channels.append(ch)
                chis.append(_row_chi(ch, q, winv))
            order = sorted(range(len(_ROWS)), key=lambda k: chis[k])
            rank = {k: i for i, k in enumerate(order)}
            shortlist = order[:top_rows]

            # monogram-resolve the shortlist (and, for brute, every row) once
            brute_rows = range(len(_ROWS)) if pair_brute else shortlist
            resolved = {k: _resolve_stream(channels[k], q, winv) for k in brute_rows}

            # 2) cheap assembly of invertible triples, ranked by monogram-resolved fit
            cand: list[tuple[float, tuple[int, int, int]]] = []
            seen_tri: set[tuple[int, int, int]] = set()

            # Bind the per-`q` mutable state as defaults so the closure captures
            # *this* iteration's objects explicitly (avoids B023 late-binding trap).
            def _cheap(
                ra: int,
                rb: int,
                rc: int,
                *,
                seen_tri: set[tuple[int, int, int]] = seen_tri,
                resolved: dict[int, tuple[float, int, list[int], list[int]]] = resolved,
                cand: list[tuple[float, tuple[int, int, int]]] = cand,
            ) -> None:
                key = (ra, rb, rc)
                if key in seen_tri:
                    return
                seen_tri.add(key)
                if math.gcd(_det3([_ROWS[ra], _ROWS[rb], _ROWS[rc]]), 26) != 1:
                    return
                pa, pb, pc = resolved[ra][2], resolved[rb][2], resolved[rc][2]
                text = "".join(alphabet[v] for j in range(len(pa)) for v in (pa[j], pb[j], pc[j]))
                cand.append((scorer.score(text), key))

            n = len(shortlist)
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(n):
                        if k != i and k != j:
                            _cheap(shortlist[i], shortlist[j], shortlist[k])

            # 3) optional pair + brute-third: rescue a row too weak to rank (scans all)
            if pair_brute:
                seeds = shortlist[: min(8, n)]
                for a_i in seeds:
                    for b_i in seeds:
                        if b_i == a_i:
                            continue
                        for rc in brute_rows:
                            if rc != a_i and rc != b_i:
                                _cheap(a_i, b_i, rc)

            # 4) strong-resolve the strongest cheap candidates (dedup unordered triples)
            cand.sort(key=lambda x: x[0], reverse=True)
            best_here: list[tuple[float, Recovered]] = []
            seen_text: set[str] = set()
            seen_set: set[frozenset[int]] = set()
            strong_k = max(30, top * 6)
            for _sc, (ra, rb, rc) in cand:
                fs = frozenset((ra, rb, rc))
                if len(fs) < 3 or fs in seen_set:
                    continue
                seen_set.add(fs)
                sc, text, matrix, offs = _strong(
                    (ra, rb, rc), channels, q, winv, scorer, alphabet, scalar_cap=scalar_cap
                )
                if text not in seen_text:
                    seen_text.add(text)
                    best_here.append(
                        (
                            sc,
                            Recovered(
                                plaintext=text,
                                decrypt_matrix=matrix,
                                alphabet=alphabet,
                                q=q,
                                offsets=offs,
                                score=sc,
                                rows_used=(rank[ra], rank[rb], rank[rc]),
                                meta={"phase": phase},
                            ),
                        )
                    )
                # early exit: a confidently-English recovery is the answer
                if scorer.confidence(text) >= stop_confidence:
                    break
                if len(seen_set) >= strong_k:
                    break

            best_here.sort(key=lambda sr: sr[0], reverse=True)
            results.extend(rec for _, rec in best_here[:top])

    results.sort(key=lambda r: r.score, reverse=True)
    # dedup by plaintext, keep best
    out: list[Recovered] = []
    seen: set[str] = set()
    for r in results:
        if r.plaintext in seen:
            continue
        seen.add(r.plaintext)
        out.append(r)
        if len(out) >= top:
            break
    return out
