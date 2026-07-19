"""Long-key Quagmire III: the 2-opt finisher that makes long keys solve cleanly.

A long Vigenere key (e.g. period 40 on ~280 letters = ~7 letters/column) leaves
single-column coordinate-ascent (``_recover_shifts``) stuck on *coupled* local
optima: two nearby columns are jointly wrong and no single-column move improves
either, so even many random restarts can fail to converge to clean English. A
quadgram window spans only four consecutive positions, so the only column pairs
that can be jointly coupled are those within cyclic distance 3 — refining just
those pairs (:func:`_quagmire_solver._two_opt_polish`) breaks the trap
deterministically and cheaply.

The fixture below is chosen so the deterministic cold (all-zero start) 1-opt pass
traps, which lets the test assert the *value added* by the 2-opt finisher without
depending on RNG.
"""

from __future__ import annotations

import random

from buttcrack.ciphers import _quagmire_solver as qs
from buttcrack.ciphers.quagmire3 import QuagmireIII, keyed_alphabet
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PLAIN = only_letters(
    "INTHEQUIETOFTHELIBRARYTHESCHOLARTURNSTHEBRITTLEPAGESOFANANCIENTATLASTRACINGF"
    "ADEDCOASTLINESWITHAGLOVEDFINGEREVERYMAPTELLSOFHARBORSLONGSILTEDOVERANDROADST"
    "HATVANISHINTOFORESTSHECOPIESEACHLEGENDINTOHERNOTEBOOKANDNUMBERSEVERYPLATEFOR"
    "THEARCHIVESHEWILLREBIND"
)[:280]
# A random 40-letter Vigenere key over the KRYPTOS keyed alphabet (period 40,
# ~7 letters/column) for which the deterministic cold 1-opt pass traps. Tuned for
# the canonical Quagmire-III alignment (keyed alphabet's first letter, KEY has no
# explicit third field), so it must be re-tuned if that default ever changes.
INDICATOR = "JRTKFXMLENPKESJPELGOFVBBILICLZWPWFNRDXTB"
KEY = f"KRYPTOS/{INDICATOR}"
PERIOD = 40


def _setup(ct: str):
    table, _ = qs._fast_table(get_scorer())
    ctn = [ord(c) - 65 for c in ct]
    cols = [[i for i in range(j, len(ct), PERIOD)] for j in range(PERIOD)]
    pre, post = qs._build_pre_post("Q3", ctn, keyed_alphabet("KRYPTOS"))
    return table, cols, pre, post


def test_cold_one_opt_traps_but_two_opt_recovers():
    """Cold 1-opt traps on this long key; the 2-opt finisher recovers it exactly."""
    ct = QuagmireIII().encode(PLAIN, KEY)
    table, cols, pre, post = _setup(ct)

    # Deterministic cold pass (restarts=1 => all-zero start, no rng): trapped.
    _, shifts, plain = qs._recover_shifts(pre, post, PERIOD, cols, table, restarts=1)
    assert "".join(chr(65 + x) for x in plain) != PLAIN

    # 2-opt finisher escapes the coupled local optimum: exact recovery.
    _, _, plain2 = qs._two_opt_polish(pre, post, PERIOD, cols, table, shifts)
    assert "".join(chr(65 + x) for x in plain2) == PLAIN


def test_two_opt_never_worsens():
    """The 2-opt finisher starts from the seed and only ever improves the score."""
    ct = QuagmireIII().encode(PLAIN, KEY)
    table, cols, pre, post = _setup(ct)
    sc1, shifts, _ = qs._recover_shifts(pre, post, PERIOD, cols, table, restarts=1)
    sc2, _, _ = qs._two_opt_polish(pre, post, PERIOD, cols, table, shifts)
    assert sc2 >= sc1


def test_dictionary_attack_solves_long_key_quagmire3():
    """Integration: the keyless dictionary attack recovers a long key cleanly and
    deterministically (the 2-opt finisher removes the old RNG dependence)."""
    ct = QuagmireIII().encode(PLAIN, KEY)
    scorer = get_scorer()
    hit = qs.dictionary_attack(
        ct, scorer, "Q3", keywords=["KRYPTOS"], forced_period=PERIOD, rng=random.Random(0)
    )
    assert hit is not None
    _, plain, period, shifts, kw = hit
    assert plain == PLAIN
    assert period == PERIOD and kw == "KRYPTOS"
    # the recovered key round-trips through decode
    key = qs.build_key("Q3", kw, shifts)
    assert only_letters(QuagmireIII().decode(ct, key)) == PLAIN

    # deterministic: a second run yields the identical plaintext
    hit2 = qs.dictionary_attack(
        ct, scorer, "Q3", keywords=["KRYPTOS"], forced_period=PERIOD, rng=random.Random(0)
    )
    assert hit2 is not None and hit2[1] == plain
