"""Look-elsewhere (family-wide) significance of the strongest period in a scan."""

import random

import pytest

from buttcrack.analysis import period_family_significance
from buttcrack.validate import encode_substitution

ENG = (
    "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEIN"
    "HERLEDGERWHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHE"
)


def test_real_period_clears_family_null():
    ct = encode_substitution(ENG, "PORTALS", alphabet="STD")  # true period 7
    r = period_family_significance(ct, statistic="coset_ioc", samples=120)
    assert r["best_period"] == 7
    assert r["beats_null_max"] and r["family_p"] < 0.05


def test_random_text_is_multiplicity_noise():
    rng = random.Random(1)
    rnd = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(150))
    r = period_family_significance(rnd, statistic="coset_ioc", samples=120)
    # the strongest period does NOT clear the family null even if its per-period z looks tempting
    assert not r["beats_null_max"] and r["family_p"] > 0.05


def test_unknown_statistic_raises():
    with pytest.raises(ValueError):
        period_family_significance(ENG, statistic="bogus")
