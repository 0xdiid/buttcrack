"""Crib-anchored inner-periodic-sub-under-columnar solver (cribbing.solve), now CLI-exposed.

It jointly recovers the columnar read-order AND the period-p Quagmire key by consistency
backtracking from a known plaintext prefix — the lever that sidesteps the flat blind
objective. This pins the recovery as a regression (the module previously had no test file).
"""

import random

from buttcrack.ciphers.columnar import _encode_letters
from buttcrack.cribbing import KRYPTOS_ALPHABET, _sub_encode, solve
from buttcrack.text import only_letters

BASE = (
    "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEINLEDGER"
    "WHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHERISINGSUNOUTSIDE"
    "ABROADRIVERWOUNDPASTTHEOLDSTONEBRIDGEWHEREFARMERSCARRIEDBASKETSOFFRESHFRUITTOTOWN"
)


def _make(width, period, variant, seed):
    rng = random.Random(seed)
    pt = (BASE * 2)[: width * (len(BASE * 2) // width)]
    shifts = [rng.randrange(26) for _ in range(period)]
    S = _sub_encode(pt, shifts, KRYPTOS_ALPHABET, variant)  # substitution INNER
    order = list(range(width))
    rng.shuffle(order)
    ct = _encode_letters(S, order)  # columnar OUTER
    return pt, ct


def test_crib_recovers_inner_sub_under_columnar():
    pt, ct = _make(width=16, period=11, variant="vig", seed=3)
    res = solve(ct, pt[:34], widths=(16,), periods=range(9, 13))
    assert res is not None
    assert only_letters(res["plaintext"]) == only_letters(pt)
    assert res["width"] == 16 and res["period"] == 11


def test_crib_handles_beaufort_variant():
    pt, ct = _make(width=12, period=9, variant="beaufort", seed=8)
    res = solve(ct, pt[:30], widths=(12,), periods=range(8, 11), variants=("vig", "beaufort"))
    assert res is not None and only_letters(res["plaintext"]) == only_letters(pt)
    assert res["variant"] == "beaufort"


def test_wrong_crib_does_not_reconstruct_english():
    """The crib is anchored, so any solution starts with it — but a WRONG crib cannot
    reconstruct coherent English in the rest of the text (it forces an inconsistent order)."""
    from buttcrack.words import long_word_coverage

    _, ct = _make(width=16, period=11, variant="vig", seed=3)
    res = solve(ct, "ZZZZZZZZZZZZ", widths=(16,), periods=range(9, 13))
    assert res is None or long_word_coverage(only_letters(res["plaintext"])) < 0.4
