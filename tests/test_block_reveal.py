"""Block-granular (unit>1) reveal search, exact-binomial block-alignment, and transposition-key
inversion — the trigraph-block transposition over a periodic Quagmire capabilities."""
from __future__ import annotations

import random

from buttcrack import keyfinder
from buttcrack.analysis import block_transposition_signal, search_aware_null
from buttcrack.ciphers.columnar import _encode_units, _read_order
from buttcrack.ciphers.quagmire3 import QuagmireIII
from buttcrack.diagnose import diagnose
from buttcrack.scoring import get_scorer
from buttcrack.transsub import crack_columnar_reveal_enum, reveal_spectrum, sweep_known_alphabet

_PLAIN = (
    "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARROWSTAIRSEACHEVENINGTOTRIMTHELAMPANDPOLISH"
    "THEGREATGLASSLENSHEHASCOUNTEDTHESTEPSFORTHIRTYYEARSANDKNOWSEVERYWORNSTONEBYH"
    "EARTFROMTHEGALLERYHEWATCHESTHEFISHINGBOATSRET"
)
_ORDER = [3, 0, 5, 1, 4, 2]  # a NON-dictionary width-6 read-order


def _block_over_q3(order=_ORDER, indicator="MEADOW", unit=3) -> str:
    inner = QuagmireIII().encode(_PLAIN, f"KRYPTOS/{indicator}")
    return _encode_units(inner, order, unit=unit)


# ---- item 2: unit= threaded through the reveal enumeration ----
def test_reveal_enum_unit3_recovers_nondictionary_block_order():
    """A trigraph-block columnar over a Quagmire III is cracked at unit=3 and NOT at unit=1 —
    the letter-only enumeration is blind to the block geometry."""
    ct = _block_over_q3()
    res3 = crack_columnar_reveal_enum(ct, get_scorer("quadgrams"), widths=[6], unit=3, null_samples=12)
    assert res3["recovered"] is True
    assert res3["structure"]["columnar_order"] == _ORDER
    assert res3["structure"]["unit"] == 3
    res1 = crack_columnar_reveal_enum(ct, get_scorer("quadgrams"), widths=[6], unit=1, null_samples=12)
    assert res1["recovered"] is False


def test_reveal_spectrum_prefers_block_granularity():
    spec = reveal_spectrum(_block_over_q3(), widths=[6], units=(1, 3), null_samples=12)
    assert spec["best"] is not None
    assert spec["best"]["unit"] == 3  # the block granularity wins the reveal spectrum


def test_search_aware_null_block_shuffle_is_stricter():
    """On a block construction, shuffling BLOCKS (unit=3) preserves intra-trigraph structure the
    letter-shuffle destroys, so the block-aware null_max is >= the letter-shuffle null_max —
    i.e. it is the harder, correct bar (letter-shuffle would over-credit the reveal)."""
    ct = _block_over_q3()

    def search(s: str) -> float:
        from buttcrack.transsub import _best_reveal_for_width
        return _best_reveal_for_width(s, 6, 3)

    letter_null = search_aware_null(ct, search, samples=12, unit=1)
    block_null = search_aware_null(ct, search, samples=12, unit=3)
    assert block_null["null_max"] >= letter_null["null_max"]


# ---- item 5: exact-binomial, all-residue block-alignment ----
def _plant(positions, gram="QZJ", n=72, seed=7):
    """Text with ``gram`` planted at ``positions`` and diverse filler carrying no >=3x trigram."""
    rng = random.Random(seed)
    for _ in range(200):
        s = [None] * n
        for p in positions:
            for j, ch in enumerate(gram):
                s[p + j] = ch
        for i in range(n):
            if s[i] is None:
                s[i] = chr(65 + rng.randrange(26))
        text = "".join(s)
        from collections import Counter
        tri = Counter(text[i:i + 3] for i in range(n - 2))
        # only the planted gram may recur >=3x
        if all(g == gram or c < 3 for g, c in tri.items()) and tri[gram] == len(positions):
            return text
    raise AssertionError("could not build clean planted text")


def test_block_alignment_detects_nonzero_residue():
    """Phase-offset grid: a trigram repeating at positions all == 1 (mod 3) is detected at
    block 3, residue 1 — the old residue-0-only test missed this entirely. (Positions are
    spaced by 9 so only the planted trigram recurs — no overlap-induced repeats.)"""
    sig = block_transposition_signal(_plant([1, 10, 19, 28, 37]))
    assert sig["best_block"] == 3
    assert sig["alignment"][3]["residue"] == 1
    assert sig["alignment"][3]["p"] < 0.01


def test_block_alignment_exact_p_small_k():
    """k=5 all-aligned: exact tail p = (1/3)^5 ~= 0.0041 — flagged, though the old >=6-count
    normal-approx gate silently dropped it."""
    sig = block_transposition_signal(_plant([0, 9, 18, 27, 36]))
    assert sig["best_block"] == 3
    assert abs(sig["alignment"][3]["p"] - (1 / 3) ** 5) < 1e-6


def test_block_alignment_none_on_unaligned():
    sig = block_transposition_signal(_plant([0, 7, 11, 20, 29]))  # residues 0,1,2,2,2 — not full
    assert sig["best_block"] is None


# ---- item 3: diagnose routes the block signal to --unit ----
def test_diagnose_recommends_unit_on_block_signal():
    ct = _plant([1, 10, 19, 28, 37], n=120)
    info = diagnose(ct)
    assert info["signals"]["block_transposition"]["best_block"] == 3
    assert any("--unit 3" in r for r in info["recommended"])


# ---- item 6: transposition-key inversion ----
def test_keyword_from_order_inverts_read_order():
    order = _read_order("PALIMPSEST")
    assert "PALIMPSEST" in keyfinder.keyword_from_order(order, ["PALIMPSEST", "KRYPTOS", "LAVENDER"])
    assert keyfinder.keyword_from_order(list(range(10)), ["PALIMPSEST"]) == []  # identity: unkeyed


def test_describe_permutation_labels_generators():
    assert "reverse" in keyfinder.describe_permutation([4, 3, 2, 1, 0])
    assert "rotate-3" in keyfinder.describe_permutation([(i + 3) % 8 for i in range(8)])
    assert "identity" in keyfinder.describe_permutation([0, 1, 2, 3])
    # a genuinely non-generator order (neither it nor its inverse is a named generator)
    assert keyfinder.describe_permutation([3, 0, 5, 1, 4, 2]) == []


# ---- item 2/4: the decider ranks a hypothesised true order first ----
def test_sweep_decider_confirms_true_block_order():
    ct = _block_over_q3()
    res = sweep_known_alphabet(ct, [_ORDER, [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]],
                               unit=3, periods=range(6, 9))
    top = res["candidates"][0]
    assert top["order"] == _ORDER
    assert top["recovered"] is True
