"""Tests for ngram_relation.scan — detecting a linear relation among n-gram positions.

Builds a synthetic homophonic-expansion cipher (the tri-square family's signature): each
plaintext letter becomes a trigraph whose closing letter satisfies a fixed modular sum and
whose other two letters are free homophones. The scanner must recover the planted relation
and report nothing on plain / random text.
"""

import random

from buttcrack.keysources import KRYPTOS
from buttcrack.ngram_relation import combine, scan
from buttcrack.text import only_letters

PLAIN = only_letters(
    "THENAVIGATORPLOTSACOURSEBYTHEWINTERSTARSANDCHECKSITTWICEAGAINSTTHECOMPASSBE"
    "FOREDAWNTHECREWHAULSTHENETSABOARDANDSORTSTHESILVERFISHINTOWOODENCRATESTHECA"
    "P"
)


def _homophonic_expand(plain: str, alphabet: str, seed: int = 0) -> str:
    """Encode: each plaintext letter P -> trigraph (C1,C2,C3) with C1 = P + C2 + C3 (mod 26)
    in ``alphabet`` index space; C2, C3 chosen at random (free homophones).

    Then the relation ``C2 + C3 - C1 == -P`` holds, i.e. coef (-1,1,1) recovers -P (an
    English-distributed channel).
    """
    rng = random.Random(seed)
    idx = {c: i for i, c in enumerate(alphabet)}
    out = []
    for p in plain:
        c2 = rng.randrange(26)
        c3 = rng.randrange(26)
        c1 = (idx[p] + c2 + c3) % 26
        out += [alphabet[c1], alphabet[c2], alphabet[c3]]
    return "".join(out)


def test_scan_recovers_planted_relation():
    ct = _homophonic_expand(PLAIN, KRYPTOS, seed=3)
    result = scan(ct, n=3, samples=800, seed=1)
    best = result["candidates"][0]
    # the planted relation is C2+C3-C1 == coef (-1,1,1) in the KRYPTOS alphabet
    assert best["alphabet"] == "KRYPTOS"
    assert best["coef"] in ([-1, 1, 1], [1, -1, -1])
    assert best["p"] < 0.01
    assert best["ioc"] > result["floor"] + 0.008
    assert "relation found" in result["verdict"]


def test_combine_extracts_negated_plaintext():
    ct = _homophonic_expand(PLAIN, KRYPTOS, seed=7)
    # coef (-1,1,1) gives (C2+C3-C1) = -P; combine renders it in the same alphabet
    channel = combine(ct, [-1, 1, 1], "KRYPTOS")
    idx = {c: i for i, c in enumerate(KRYPTOS)}
    recovered = "".join(KRYPTOS[(-idx[c]) % 26] for c in channel)
    assert recovered == PLAIN


def test_no_relation_on_plain_text():
    # ordinary English trigraphs carry no hidden linear relation above the floor
    result = scan(PLAIN * 3, n=3, samples=600, seed=1)
    assert "no elevated linear relation" in result["verdict"]


def test_no_relation_on_random_text():
    rng = random.Random(11)
    rnd = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(300))
    result = scan(rnd, n=3, samples=600, seed=1)
    assert "no elevated linear relation" in result["verdict"]


def test_searchaware_p_is_matched_count_under_a_larger_search():
    # The search-aware p must scale with the NUMBER of functionals actually evaluated (multiple
    # testing). Widening the coefficient set to {-2..2} multiplies the candidate count by an order
    # of magnitude; the best raw IoC on random text rises with it purely from selection, so a
    # non-matched (fixed) correction would flag a spurious relation. The matched empirical null
    # (per-replicate max over the SAME enlarged candidate set) must not.
    rng = random.Random(7)
    rnd = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(360))
    result = scan(
        rnd, n=3, coeffs=(-2, -1, 0, 1, 2), alphabets=("KRYPTOS", "STD"), samples=600, seed=1
    )
    assert "no elevated linear relation" in result["verdict"]
    best = result["candidates"][0]
    # Even the strongest of the many candidates is not significant once the null is matched-count.
    assert best["p"] > 0.01
    # Laplace-smoothed empirical p is always a proper, bounded, non-zero probability.
    assert all(0.0 < c["p"] <= 1.0 for c in result["candidates"])
