"""Crib-anchored constraint solver for sub-INNER complete columnar.

Model
-----
``CT = columnar(width w, read-order)( Sub_p( PT ) )`` with the substitution applied
*inside* the transposition (sub-INNER). Complete columnar only (``N % w == 0``),
so the grid has ``H = N // w`` rows.

Geometry (must match :mod:`buttcrack.ciphers.columnar`)
    Cell ``(r, c)`` of the grid is ``PT[r*w + c]``; it is substituted with
    ``key[(r*w + c) % p]``. The columns are read out in ``read-order``; for a
    complete columnar every column is exactly ``H`` letters, so column ``c`` lands
    in ciphertext block ``slot = inv[c]`` at positions ``slot*H .. slot*H + H - 1``.
    Hence ``CT[slot*H + r] = Sub( PT[r*w + c], key[(r*w + c) % p] )``.

Why a constraint solver
-----------------------
A known plaintext PREFIX (the crib) pins the column read-order AND the period-``p``
substitution key *jointly* by consistency rather than by optimising a flat n-gram
objective. Each crib cell ``PT[j]`` (with ``j = c + k*w``, row ``r = j // w``) forces
the key shift of residue class ``j % p`` to a specific value for the candidate slot
of column ``c``. Backtracking over the column->slot bijection prunes the instant two
crib cells in the same residue class disagree; a consistent full assignment fixes the
order and (usually) the whole key, so the rest of the message decrypts directly.
This sidesteps the small-``N`` information wall that defeats blind objective search.

Public API
----------
``solve(ct, crib, widths=..., periods=..., alphabets=..., variants=...) -> dict``
returns the best-scoring consistent solution as a dict with keys
``score, plaintext, width, period, alphabet, variant, order``
(``order[slot] = source column``), or ``None`` if nothing is consistent.
"""

from __future__ import annotations

from collections.abc import Iterable

from .scoring import get_scorer

#: KRYPTOS-keyed alphabet and the plain standard alphabet.
KRYPTOS_ALPHABET = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
STD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ALPHABETS = {
    "KRYPTOS": KRYPTOS_ALPHABET,
    "KRY": KRYPTOS_ALPHABET,
    "STD": STD_ALPHABET,
    "STANDARD": STD_ALPHABET,
}


def _resolve_alphabet(spec: str) -> tuple[str, str]:
    """Map an alphabet spec to ``(name, alphabet)``; a literal 26-letter permutation
    is accepted verbatim (named ``CUSTOM``)."""
    key = spec.upper()
    if key in _ALPHABETS:
        return key, _ALPHABETS[key]
    if len(spec) == 26 and sorted(spec.upper()) == list(STD_ALPHABET):
        return "CUSTOM", spec.upper()
    raise ValueError(f"unknown alphabet spec: {spec!r}")


def solve(
    ct: str,
    crib: str,
    widths: Iterable[int] = (16, 17),
    periods: Iterable[int] = range(9, 17),
    alphabets: Iterable[str] = ("KRYPTOS", "STD"),
    variants: Iterable[str] = ("vig", "beaufort"),
    *,
    scorer=None,
    leaf_cap: int = 3000,
    node_cap: int = 800000,
) -> dict | None:
    """Crib-anchor a sub-INNER complete columnar and return the best solution.

    Parameters
    ----------
    ct, crib:
        Ciphertext and the known plaintext PREFIX (``crib == PT[:len(crib)]``).
    widths, periods, alphabets, variants:
        Configuration grid to sweep. ``widths`` must divide ``len(ct)`` to be tried
        (others are silently skipped — complete columnar only). ``alphabets`` accepts
        ``"KRYPTOS"``/``"KRY"``, ``"STD"``/``"STANDARD"``, or a literal 26-letter
        permutation. ``variants`` are ``"vig"`` and/or ``"beaufort"``.
    scorer:
        n-gram scorer with a ``.score(text)`` method; defaults to English quadgrams.
    leaf_cap, node_cap:
        Backtracking budgets (consistent orders decrypted / nodes visited) so a weak
        crib cannot explode the search.

    Returns
    -------
    dict with ``score, plaintext, width, period, alphabet, variant, order`` for the
    highest-scoring consistent solution, or ``None`` if nothing is consistent.
    ``order`` is the column read-order (``order[slot] = source column``).
    """
    if scorer is None:
        scorer = get_scorer("quadgrams", "english")
    if not crib:
        raise ValueError("crib must be a non-empty known-plaintext prefix")

    best: dict | None = None
    for width in widths:
        if width <= 0 or len(ct) % width:
            continue
        if len(crib) > len(ct):
            continue
        for alph_spec in alphabets:
            alph_name, alphabet = _resolve_alphabet(alph_spec)
            for variant in variants:
                if variant not in ("vig", "beaufort"):
                    raise ValueError(f"unknown variant: {variant!r}")
                for period in periods:
                    if period <= 0:
                        continue
                    for pt, sc, order in _iter_solutions(
                        ct,
                        crib,
                        width,
                        period,
                        alphabet,
                        variant,
                        scorer,
                        leaf_cap=leaf_cap,
                        node_cap=node_cap,
                    ):
                        if best is None or sc > best["score"]:
                            best = {
                                "score": sc,
                                "plaintext": pt,
                                "width": width,
                                "period": period,
                                "alphabet": alph_name,
                                "variant": variant,
                                "order": order,
                            }
    return best


def _iter_solutions(ct, crib, width, period, alphabet, variant, scorer, *, leaf_cap, node_cap):
    """Like ``_solve_one`` but also reports the recovered read-order per solution."""
    n = len(ct)
    if width <= 0 or n % width:
        return
    H = n // width
    ai = {c: i for i, c in enumerate(alphabet)}
    try:
        cti = [ai[c] for c in ct]
        cribi = [ai[c] for c in crib]
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} not in alphabet {alphabet!r}") from exc
    L = len(crib)
    col_cells = {c: [(j, j // width) for j in range(c, L, width)] for c in range(width)}

    used_slot = [False] * width
    inv = [-1] * width
    caps = {"leaf": leaf_cap, "node": node_cap}

    def desub_idx(ct_idx, shift):
        return (ct_idx - shift) % 26 if variant == "vig" else (shift - ct_idx) % 26

    def implied_shift(ct_idx, pt_idx):
        return (ct_idx - pt_idx) % 26 if variant == "vig" else (ct_idx + pt_idx) % 26

    out_results: list[tuple[str, float, list[int]]] = []

    def emit(key):
        order = [0] * width
        for col, slot in enumerate(inv):
            order[slot] = col
        S = [0] * n
        for slot in range(width):
            col = order[slot]
            base = slot * H
            for r in range(H):
                S[r * width + col] = cti[base + r]
        chars = []
        for i in range(n):
            sh = key.get(i % period, 0)
            chars.append(alphabet[desub_idx(S[i], sh)])
        pt = "".join(chars)
        out_results.append((pt, scorer.score(pt), order))

    def backtrack(c, key):
        if caps["leaf"] <= 0 or caps["node"] <= 0:
            return
        caps["node"] -= 1
        if c == width:
            caps["leaf"] -= 1
            emit(key)
            return
        for slot in range(width):
            if used_slot[slot]:
                continue
            base = slot * H
            newkey: dict[int, int] = {}
            ok = True
            for j, r in col_cells[c]:
                sh = implied_shift(cti[base + r], cribi[j])
                cls = j % period
                cur = key.get(cls)
                if cur is None:
                    cur = newkey.get(cls)
                if cur is None:
                    newkey[cls] = sh
                elif cur != sh:
                    ok = False
                    break
            if not ok:
                continue
            used_slot[slot] = True
            inv[c] = slot
            merged = dict(key)
            merged.update(newkey)
            backtrack(c + 1, merged)
            used_slot[slot] = False
            inv[c] = -1

    backtrack(0, {})
    yield from out_results


# --------------------------------------------------------------------------- #
# Self-test: plant a synthetic of the exact target structure and assert recovery.
# --------------------------------------------------------------------------- #
def _sub_encode(pt, shifts, alphabet, variant):
    idx = {c: i for i, c in enumerate(alphabet)}
    p = len(shifts)
    out = []
    for i, ch in enumerate(pt):
        j = idx[ch]
        s = shifts[i % p]
        k = (j + s) % 26 if variant == "vig" else (s - j) % 26
        out.append(alphabet[k])
    return "".join(out)


def _self_test():
    import random

    from buttcrack.ciphers.columnar import _encode_letters

    rng = random.Random(3)
    # Known English text, 272 chars, opening with a recognizable prefix.
    base = (
        "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEINLEDGER"
        "WHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHERISINGSUNOUTSIDE"
        "ABROADRIVERWOUNDPASTTHEOLDSTONEBRIDGEWHEREFARMERSCARRIEDBASKETSOFFRESHFRUITTOTOWN"
        "ANDCHILDRENPLAYEDALONGTHEGRASSYBANKSLAUGHINGASTHEYCHASEDONEANOTHERTHROUGHTHEFIELDS"
    )
    pt = (base * 2)[:272]
    width = 16
    period = 11
    shifts = [rng.randrange(26) for _ in range(period)]
    S = _sub_encode(pt, shifts, KRYPTOS_ALPHABET, "vig")  # sub INNER
    order = list(range(width))
    rng.shuffle(order)
    ct = _encode_letters(S, order)  # columnar OUTER
    crib = pt[:34]  # true 34-char opening (covers >2 rows of width 16)

    print("=== SELF-TEST cribbing.solve (w16, p11, KRYPTOS/vig, 34-char crib) ===", flush=True)
    print(f"    planted order = {order}", flush=True)
    res = solve(
        ct,
        crib,
        widths=(16,),
        periods=(11,),
        alphabets=("KRYPTOS",),
        variants=("vig",),
    )
    assert res is not None, "no consistent solution found"
    match = sum(a == b for a, b in zip(res["plaintext"], pt, strict=False)) / len(pt)
    print(f"    recovered order = {res['order']}", flush=True)
    print(
        f"    score={res['score']:.0f} match={match:.0%} "
        f"cfg=(w{res['width']} p{res['period']} {res['alphabet']}/{res['variant']})",
        flush=True,
    )
    print(f"    plaintext[:80] = {res['plaintext'][:80]}", flush=True)
    assert match == 1.0, f"expected 100% recovery, got {match:.0%}"

    # Also confirm the full sweep (defaults) finds it among many configs.
    res2 = solve(ct, crib)
    assert res2 is not None, "full default sweep found no consistent solution"
    match2 = sum(a == b for a, b in zip(res2["plaintext"], pt, strict=False)) / len(pt)
    print(
        f"    [full default sweep] best score={res2['score']:.0f} match={match2:.0%} "
        f"cfg=(w{res2['width']} p{res2['period']} {res2['alphabet']}/{res2['variant']})",
        flush=True,
    )
    assert match2 == 1.0, f"full sweep failed to recover: {match2:.0%}"

    print("SELF-TEST PASSED: 100% recovery (targeted + full sweep)", flush=True)


if __name__ == "__main__":  # pragma: no cover
    # Run as a package module so the relative imports resolve:
    #   PYTHONPATH=src python3 -m buttcrack.cribbing
    _self_test()
