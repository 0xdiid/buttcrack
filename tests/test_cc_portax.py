"""Portax: validated against the ACA Portax PDF vectors; hill-climb crack."""

import random

import pytest

from buttcrack.ciphers.portax import Portax
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters

# A longer plaintext for round-trip and crack exercises.
PT = (
    "the art of war teaches us to rely not on the likelihood of the enemy not "
    "coming but on our own readiness to receive him not on the chance of his not "
    "attacking but rather on the fact that we have made our position unassailable "
    "all warfare is based on deception when able to attack we must seem unable to move"
)


def test_portax_aca_vector():
    # Source: ACA cipher description PDF "PORTAX",
    # cryptogram.org/downloads/aca.info/ciphers/Portax.pdf.
    # "the early bird gets the worm" under key EASY, block rows
    # THEE/ARLY/BIRD/GETS/THEW/ORMX -> cipher rows NIJA/MPBG/QCWK/HQJE/UIKY/MPAT.
    px = Portax()
    assert px.encode("the early bird gets the worm", "EASY") == "NIJAMPBGQCWKHQJEUIKYMPAT"


def test_portax_pdf_single_pair_vectors():
    # Same PDF: with the slide set for key letter U, 'in'->JL, 'no'->UA, 'na'->DB.
    px = Portax()
    assert px.encode("in", "U") == "JL"
    assert px.encode("no", "U") == "UA"
    assert px.encode("na", "U") == "DB"


def test_portax_roundtrip_prepared():
    px = Portax()
    key = "PORTAX"
    ct = px.encode(PT, key)
    # decode recovers the padded prepared plaintext (== decode of our own encode).
    prepared = px.decode(ct, key)
    assert px.decode(px.encode(PT, key), key) == prepared
    # and the head matches the original letters (no transposition of content).
    assert prepared.startswith(only_letters(PT)[:40])


@pytest.mark.slow
def test_portax_crack_recovers_long_text():
    # Per-column slide hill-climb with auto period detection. Seeded for determinism.
    px = Portax()
    key = "CIPHER"
    ct = px.encode(PT, key)
    prepared = px.decode(ct, key)
    best = px.crack(ct, get_scorer(), rng=random.Random(3), timeout=40)[0]
    recovered = only_letters(best.plaintext)
    matches = sum(a == b for a, b in zip(recovered, prepared, strict=False))
    assert matches / len(prepared) >= 0.9
