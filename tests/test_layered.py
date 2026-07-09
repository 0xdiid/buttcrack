"""Layered solver: periodic (Quagmire) substitution OVER a columnar transposition.

Covers the hard real-world case (keyed alphabet, long period,
unknown columnar order) — the sibling of engine._layered_additive_crack. The
order-independent chi-square de-sub seed + columnar-order hill-climb + quadgram
shift recovery solve it exactly when columns are long enough to be unambiguous.
"""

from __future__ import annotations

import random
import string

import pytest

from buttcrack.ciphers.columnar import Columnar, _read_order
from buttcrack.ciphers.quagmire3 import QuagmireIII
from buttcrack.layered import (
    _alphabet_header,
    column_alternatives,
    crack_layered,
    crack_quagmire_over_columnar,
    detect_periods,
    solve_inner_periodic,
    solve_inner_periodic_screen,
    substitution_over_transposition_test,
)
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

PLAIN = only_letters(
    "THEARCHIVISTRECORDEDEVERYDETAILOFTHELONGSEARCHFORTHESILVERNEEDLEACROSSMANY"
    "COUNTRIESANDYEARSWITHGREATCAREANDPATIENCEHOPINGTHATSOMEONEWOULDONEDAYFINISH"
    "WHATHEHADSTARTEDINTHEQUIETARCHIVEBEFOREITWASLOSTAGAINTOTIMEANDNEGLECTFOREVER"
)[:240]


def _make(period: int, colkey: str, seed: int = 5) -> str:
    rng = random.Random(seed)
    key = "KRYPTOS/" + "".join(rng.choice(string.ascii_uppercase) for _ in range(period))
    return QuagmireIII().encode(Columnar().encode(PLAIN, colkey), key)


def test_known_order_exact():
    """Given the columnar order, the substitution shifts recover exactly."""
    ct = _make(12, "NEEDLES")  # width 7, 240 letters -> ~20 letters/column
    r = crack_quagmire_over_columnar(
        ct,
        get_scorer(),
        alphabet="KRYPTOS",
        period=12,
        order=_read_order("NEEDLES"),
        shift_restarts=30,
    )
    assert r["plaintext"] == PLAIN
    assert r["structure"]["columnar_order"] == _read_order("NEEDLES")
    assert r["residual"] == []  # long columns -> nothing ambiguous


@pytest.mark.slow
def test_order_search_recovers_width_and_text():
    """Blind columnar order: the width/order are recovered by hill-climb and the
    plaintext comes out clean (columns long enough to be unambiguous)."""
    ct = _make(12, "NEEDLES")
    r = crack_quagmire_over_columnar(
        ct,
        get_scorer(),
        alphabet="KRYPTOS",
        period=12,
        widths=[6, 7, 8],
        order_restarts=25,
        shift_restarts=15,
    )
    assert r["structure"]["columnar_width"] == 7
    assert r["structure"]["columnar_order"] == _read_order("NEEDLES")
    assert r["plaintext"] == PLAIN


def test_detect_periods_finds_the_substitution_period():
    """The outer-substitution period is recoverable from the raw ciphertext."""
    ct = _make(12, "NEEDLES")
    assert 12 in detect_periods(ct, top=4)


def test_crack_layered_autonomous():
    """End-to-end autonomy: no period or order given — detect the period and brute the
    columnar (single-process for test determinism) to an exact solve."""
    ct = _make(12, "NEEDLES")  # width 7, period 12 -> long columns, fast brute
    r = crack_layered(
        ct, get_scorer(), alphabet="KRYPTOS", periods=[12], widths=range(6, 8), workers=1
    )
    assert r["plaintext"] == PLAIN
    assert r["structure"]["period"] == 12
    assert r["structure"]["columnar_width"] == 7
    assert r["structure"]["columnar_order"] == _read_order("NEEDLES")


def test_sott_detects_substitution_over_transposition():
    """A genuine sub-over-columnar: the chi-square peel snaps the residual IoC to language."""
    ct = _make(7, "NEEDLES")  # period 7 over a width-7 columnar of English
    r = substitution_over_transposition_test(ct, period=7, alphabet="KRYPTOS")
    assert r["verdict"] == "substitution-over-transposition"
    assert r["chi_seed_ioc"] >= r["language_ioc"] - 0.012


def test_sott_flags_flattener_and_ioc_overfit():
    """A flattening inner (bifid) under a period-7 sub: the peel does NOT snap to language;
    coset IoC is real (flattener band) and the free IoC-max peel is flagged as an overfit."""
    from buttcrack.ciphers.bifid import bifid_encode
    from buttcrack.ciphers.quagmire3 import keyed_alphabet

    inner = bifid_encode(PLAIN[:154], "BUTTERFLY", 7)
    hdr = keyed_alphabet("KRYPTOS")
    shifts = [3, 17, 0, 9, 22, 5, 14]
    ct = "".join(hdr[(hdr.index(c) + shifts[i % 7]) % 26] for i, c in enumerate(inner))
    r = substitution_over_transposition_test(ct, period=7, alphabet="KRYPTOS")
    assert r["verdict"] == "substitution-over-flattener"
    assert r["chi_seed_ioc"] < r["language_ioc"] - 0.01
    assert r["coset_ioc"] > r["floor"]  # real period-7 structure, just not language
    # ioc_max (the degenerate peel) never undershoots the honest chi-square peel
    assert r["ioc_max"] >= r["chi_seed_ioc"]
    assert isinstance(r["overfit_warning"], bool)


def test_sott_reports_no_structure_on_random():
    rng = random.Random(11)
    rt = "".join(rng.choice(string.ascii_uppercase) for _ in range(150))
    r = substitution_over_transposition_test(rt, period=7, alphabet="STD")
    assert r["verdict"] == "no-structure-or-wrong-period"


def test_full_climb_objective_separates_true_order():
    """Regression for the order-search objective: a DETERMINISTIC full quadgram climb
    (run to convergence) must score the true columnar order clearly above a near-miss
    (a 2-swap perturbation). A truncated objective under-converges and ranks the
    near-miss ~as high — the plateau failure mode. Uses short columns (~5/col)."""
    from buttcrack.ciphers.quagmire3 import keyed_alphabet
    from buttcrack.layered import _chi_seed, _fast_quad_table, _recover_shifts

    period = 40  # 200 letters -> 5 letters/column (the hard, short-column regime)
    rng = random.Random(3)
    key = "KRYPTOS/" + "".join(rng.choice(string.ascii_uppercase) for _ in range(period))
    ct = QuagmireIII().encode(Columnar().encode(PLAIN[:200], "MONARCHY"), key)
    header = keyed_alphabet("KRYPTOS")
    table = _fast_quad_table(get_scorer())
    seed = _chi_seed(ct, period, header, {c: i for i, c in enumerate(header)})
    true = _read_order("MONARCHY")
    near = true[:]
    near[0], near[1] = near[1], near[0]  # one swap away

    def full_climb(order):
        return _recover_shifts(ct, header, period, order, table, seed=seed)[0]  # no passes cap

    assert full_climb(true) > full_climb(near) + 50  # wide, reliable margin


def test_residual_suppressed_when_clean():
    """A clean solve returns no residual report (it would just be noise)."""
    ct = _make(12, "NEEDLES")
    r = crack_quagmire_over_columnar(
        ct,
        get_scorer(),
        alphabet="KRYPTOS",
        period=12,
        order=_read_order("NEEDLES"),
        shift_restarts=30,
    )
    assert r["plaintext"] == PLAIN
    assert r["word_coverage"] >= 0.42
    assert r["residual"] == []


def _encode_periodic(plain: str, alphabet: str, convention: str, shifts: list[int]) -> str:
    """Encrypt ``plain`` with a periodic substitution over a keyed/standard alphabet.

    Inverse of the family-grammar decrypt conventions (index space over the alphabet):
    Vigenere dec ``p=c-k`` -> enc ``c=p+k``; Beaufort ``p=k-c`` -> enc ``c=k-p``;
    variant dec ``p=c+k`` -> enc ``c=p-k``.
    """
    header = _alphabet_header(alphabet)
    hpos = {c: i for i, c in enumerate(header)}
    period = len(shifts)
    out = []
    for i, ch in enumerate(plain):
        p = hpos[ch]
        k = shifts[i % period]
        if convention == "vigenere":
            c = (p + k) % 26
        elif convention == "beaufort":
            c = (k - p) % 26
        elif convention == "variant":
            c = (p - k) % 26
        else:  # pragma: no cover - test guard
            raise ValueError(convention)
        out.append(header[c])
    return "".join(out)


def test_solve_inner_periodic_recovers_period13_beaufort_kryptos():
    """The generic inner solve recovers a planted period-13 Beaufort-in-KRYPTOS stream:
    right convention, right alphabet, right period, exact shifts, clean plaintext."""
    rng = random.Random(7)
    shifts = [rng.randrange(26) for _ in range(13)]
    ct = _encode_periodic(PLAIN, "KRYPTOS", "beaufort", shifts)
    score, plain, meta = solve_inner_periodic(ct, get_scorer(), periods=range(2, 16))
    assert meta["convention"] == "beaufort"
    assert meta["alphabet"] == "KRYPTOS"
    assert meta["period"] == 13
    assert meta["shifts"] == shifts
    assert plain == PLAIN


def test_solve_inner_periodic_recovers_standard_vigenere():
    """It also generalises to a STANDARD-alphabet Vigenere (not just keyed/Q3)."""
    rng = random.Random(11)
    shifts = [rng.randrange(26) for _ in range(7)]
    ct = _encode_periodic(PLAIN, "STD", "vigenere", shifts)
    _, plain, meta = solve_inner_periodic(ct, get_scorer(), periods=range(2, 12))
    assert meta["convention"] == "vigenere"
    assert meta["alphabet"] == "STD"
    assert meta["period"] == 7
    assert plain == PLAIN


def test_solve_inner_periodic_screen_locked_period():
    """The cheap screen variant solves when the period is already locked. Uses Beaufort
    (genuinely distinct from Vigenere/variant, which alias up to shift negation)."""
    rng = random.Random(3)
    shifts = [rng.randrange(26) for _ in range(9)]
    ct = _encode_periodic(PLAIN, "KRYPTOS", "beaufort", shifts)
    _, plain, meta = solve_inner_periodic_screen(ct, get_scorer(), period=9)
    assert meta["convention"] == "beaufort"
    assert meta["period"] == 9
    assert plain == PLAIN


def test_solve_inner_periodic_variant_recovers_plaintext():
    """A variant-Beaufort stream recovers the right PLAINTEXT; the reported convention may
    read 'vigenere' because variant == Vigenere with a negated key (a true ambiguity)."""
    rng = random.Random(9)
    shifts = [rng.randrange(26) for _ in range(8)]
    ct = _encode_periodic(PLAIN, "KRYPTOS", "variant", shifts)
    _, plain, meta = solve_inner_periodic(ct, get_scorer(), periods=range(2, 12))
    assert meta["period"] == 8
    assert plain == PLAIN


def test_column_alternatives_shape():
    """The residual report is the agent hook: per ambiguous column, ranked candidate
    shifts each with the plaintext slots that column controls, shown in context.
    With a wide gap threshold every column is reported; the shape must be valid and
    the current shift must be present among the options."""
    from buttcrack.ciphers.quagmire3 import keyed_alphabet
    from buttcrack.layered import _chi_seed, _fast_quad_table, _recover_shifts

    ct = _make(12, "NEEDLES")
    header = keyed_alphabet("KRYPTOS")
    table = _fast_quad_table(get_scorer())
    order = _read_order("NEEDLES")
    seed = _chi_seed(ct, 12, header, {c: i for i, c in enumerate(header)})
    _, shifts, _ = _recover_shifts(ct, header, 12, order, table, seed=seed, restarts=10)
    report = column_alternatives(ct, header, 12, order, shifts, table, gap=1e9)
    assert len(report) == 12  # every column surfaced under a wide gap
    for entry in report:
        assert 0 <= entry["column"] < 12
        assert entry["options"], "each column must list candidate shifts"
        assert any(opt["current"] for opt in entry["options"])
        first = entry["options"][0]
        assert {"shift", "letters", "contexts", "current"} <= first.keys()
        assert len(first["contexts"]) >= 1
