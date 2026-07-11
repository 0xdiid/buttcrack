"""Forward-simulation model-ID: the right structural class fits, wrong ones are refuted."""

from __future__ import annotations

import random

from buttcrack import construction
from buttcrack.ciphers.columnar import Columnar
from buttcrack.ciphers.vigenere import Vigenere

PLAIN = construction._CORPUS[:220]


def test_dial_vector_shape():
    dv = construction.dial_vector(PLAIN)
    assert set(dv) == {"ic", "dic", "entropy", "cic2", "cic3", "cic5", "cic7"}
    assert 0.05 < dv["ic"] < 0.08  # clean English monogram IoC


def test_vigenere_ciphertext_fits_polyalphabetic_not_language():
    ct = Vigenere().encode(PLAIN, "PALIMPSEST")  # period-10 polyalphabetic
    report = construction.construction_baseline(ct, n_sims=150, rng=random.Random(1))
    top3 = [f.name for f in report.families[:3]]
    # A polyalphabetic family should top the ranking; the true class is vigenere/beaufort.
    assert any(name in top3 for name in ("vigenere", "beaufort"))
    # The language and transposition families cannot manufacture the low monogram IoC.
    english = report.get("english")
    columnar = report.get("columnar")
    assert english is not None and english.verdict == "refuted"
    assert columnar is not None and columnar.verdict == "refuted"


def test_transposition_ciphertext_keeps_english_dials():
    # Columnar preserves monogram IoC, so the language/transposition classes fit and
    # polyalphabetic families are refuted (their low-IoC signature is absent).
    ct = Columnar().encode(PLAIN, "TRIANGLE")
    report = construction.construction_baseline(ct, n_sims=150, rng=random.Random(2))
    vig = report.get("vigenere")
    assert vig is not None and vig.verdict == "refuted"
    # english/columnar/substitution keep English-like monogram IoC -> not refuted on ic.
    for name in ("english", "columnar", "substitution"):
        f = report.get(name)
        assert f is not None
        assert f.per_dial["ic"]["z"] < report.get("vigenere").per_dial["ic"]["z"]


def test_report_helpers():
    ct = Vigenere().encode(PLAIN, "LEMON")
    report = construction.construction_baseline(ct, n_sims=80, rng=random.Random(3))
    assert report.best is report.families[0]
    assert len(report.plausible) + len(report.refuted) == len(report.families)
    assert isinstance(report.summary(), str)
