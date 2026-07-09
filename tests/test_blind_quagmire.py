"""Blind/keyword Quagmire recovery: the reliable machinery, honestly scoped.

The *dictionary* attack (a known keyed-alphabet keyword) is the reliable route and
is asserted to solve. Blind recovery of an *arbitrary* keyed alphabet is best-effort
(the alphabet is an isolated optimum with ~no gradient), so we don't assert it
*solves* — only that its deterministic inner machinery is correct and that it never
emits an invalid key or runs below its length floor.
"""

import random
import time

from buttcrack.ciphers import _quagmire_solver as qs
from buttcrack.ciphers.quagmire3 import QuagmireIII, keyed_alphabet
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PLAIN = only_letters(
    "BETWEENSUBTLESHADINGANDTHEABSENCEOFLIGHTLIESTHENUANCEOFIQLUSIONITWASTOTALLY"
    "INVISIBLEHOWSTHATPOSSIBLETHEYUSEDTHEEARTHSMAGNETICFIELDXTHEINFORMATIONWAS"
    "GATHEREDANDTRANSMITTEDUNDERGRUUNDTOANUNKNOWNLOCATIONX"
)


def test_dictionary_attack_solves_kryptos_quagmire3():
    """The reliable route: a famous keyed-alphabet keyword is recovered keyless."""
    cipher = QuagmireIII()
    ct = cipher.encode(PLAIN, "KRYPTOS/PALIMPSEST")
    scorer = get_scorer()
    results = cipher.crack(ct, scorer, timeout=20, rng=random.Random(1))
    assert results, "dictionary attack returned nothing"
    best = results[0]
    assert only_letters(best.plaintext) == PLAIN
    assert best.confidence >= 0.85
    # the published key must round-trip through decode
    assert only_letters(cipher.decode(ct, best.key)) == PLAIN


def test_blind_cycleword_recovery_is_exact_given_true_alphabet():
    """The deterministic step: with the right keyed alphabet, per-column chi-square +
    quadgram shift recovery reconstruct the plaintext exactly (this is the part the
    blind anneal would only need to *find the alphabet* for)."""
    cipher = QuagmireIII()
    ct = cipher.encode(PLAIN, "KRYPTOS/PALIMPSEST")
    scorer = get_scorer()
    table, _ = qs._fast_table(scorer)
    ctn = [ord(x) - 65 for x in ct]
    period = 10
    cols = [[i for i in range(j, len(ct), period)] for j in range(period)]
    pre, post = qs._build_pre_post("Q3", ctn, keyed_alphabet("KRYPTOS"))
    _, _, plain = qs._recover_shifts(pre, post, period, cols, table, restarts=1)
    assert "".join(chr(65 + x) for x in plain) == PLAIN


def test_blind_respects_length_floor():
    cipher = QuagmireIII()
    short = cipher.encode(PLAIN[:80], "KRYPTOS/PALIMPSEST")
    assert qs.blind_candidates(cipher, "Q3", short, get_scorer(), deadline=None) == []


def test_blind_never_emits_invalid_key():
    """Best-effort: it may not converge, but any candidate it returns must round-trip."""
    cipher = QuagmireIII()
    ct = cipher.encode(PLAIN, "KRYPTOS/PALIMPSEST")
    scorer = get_scorer()
    cands = qs.blind_candidates(
        cipher, "Q3", ct, scorer, deadline=time.monotonic() + 2, rng=random.Random(2)
    )
    for cand in cands:
        assert cand.key is None or only_letters(cipher.decode(ct, cand.key)) == only_letters(
            cand.plaintext
        )
