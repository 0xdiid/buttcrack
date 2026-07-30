"""Autocorrelation + Friedman reads (`butt stats --autocorrelation / --friedman`).

Both statistics existed inside ``analysis`` but were unreachable from the CLI and
un-gated. These tests are the plant gate: a detector that cannot recover a period
it planted itself has no business reporting one on real ciphertext.
"""

import json

import pytest

from buttcrack import registry
from buttcrack.analysis import autocorrelation_report, friedman_test
from buttcrack.cli import main

# Long enough that a 12-letter period still has ~60 letters per column.
CORPUS = (
    "the quick brown fox jumps over the lazy dog while the industrious beaver "
    "constructs a dam across the swift river and the patient heron waits beside "
    "the shallow water for a careless fish to swim within reach of its long and "
    "very sudden beak once again today"
) * 3


def _vig(key: str) -> str:
    return registry.get("vigenere").encode(CORPUS, key)


@pytest.mark.parametrize(
    "key",
    ["XY", "CAT", "LEMON", "SECRET", "KRYPTOS", "PASSWORD", "SECRETKEY", "PALIMPSEST"],
)
def test_autocorrelation_recovers_planted_vigenere_period(key):
    """The true key length is in the reported ladder for every planted period.

    ``candidate_periods`` and not ``best_period``: a key with a repeated letter
    (SECRET has E at 1 and 4) genuinely makes the gap coincide, so lag 3 outranks
    the true 6 on its own merit. The ladder is the honest answer.
    """
    report = autocorrelation_report(_vig(key), max_lag=32)
    assert report["verdict"] == "period"
    assert len(key) in report["candidate_periods"]


def test_autocorrelation_period_is_lowest_or_a_divisor():
    """best_period is the true period or one of its divisors, never a stray lag."""
    report = autocorrelation_report(_vig("PALIMPSEST"), max_lag=32)
    assert 10 % report["best_period"] == 0


@pytest.mark.parametrize("key", ["SECRET", "KRYPTOS", "LEMON"])
def test_autocorrelation_recovers_beaufort_period(key):
    ct = registry.get("beaufort").encode(CORPUS, key)
    report = autocorrelation_report(ct, max_lag=32)
    assert len(key) in report["candidate_periods"]


@pytest.mark.parametrize(
    "cipher,key",
    [
        ("caesar", "5"),
        ("substitution", "QWERTYUIOPASDFGHJKLZXCVBNM"),
        ("columnar", "ZEBRAS"),
        ("railfence", "5"),
    ],
)
def test_autocorrelation_refuses_monoalphabetic_and_transposition(cipher, key):
    """The gate that stops the detector inventing a period out of flat text.

    Monoalphabetic and transposed text coincide at plaintext rate at *every* lag, so
    every lag clears the 1/26 random floor and a harmonic search will always find
    something. The report must decline rather than name a period.
    """
    ct = registry.get(cipher).encode(CORPUS, key)
    report = autocorrelation_report(ct, max_lag=32)
    assert report["best_period"] is None
    assert "cannot resolve a period" in report["verdict"]


def test_autocorrelation_null_on_random_text():
    import random

    rng = random.Random(1)
    text = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(600))
    report = autocorrelation_report(text, max_lag=32)
    assert report["verdict"] != "period"


def test_autocorrelation_short_text_declines():
    report = autocorrelation_report("ABCDE", max_lag=32)
    assert report["best_period"] is None
    assert "too short" in report["verdict"]


def test_friedman_estimate_near_planted_key_length():
    """Friedman is a coarse scalar — assert the order of magnitude, not the integer."""
    report = friedman_test(_vig("KRYPTOS"))
    assert report["reliable"]
    assert 3.0 < report["estimate"] < 12.0


def test_friedman_flags_monoalphabetic_as_unreliable():
    """IoC at plaintext level: the closed form collapses toward 1, which is not a
    key length. It must say so rather than return the number."""
    ct = registry.get("caesar").encode(CORPUS, "5")
    report = friedman_test(ct)
    assert not report["reliable"]
    assert "monoalphabetic level" in report["note"]


def test_friedman_flags_short_text_as_unreliable():
    report = friedman_test(registry.get("vigenere").encode("attack at dawn ok now", "LEMON"))
    assert not report["reliable"]


def test_friedman_accepts_non_english_plaintext_ioc():
    """kappa_plain is a parameter: an English default biases a non-English payload."""
    ct = _vig("KRYPTOS")
    assert friedman_test(ct, kappa_p=0.0738)["estimate"] != friedman_test(ct)["estimate"]


def test_cli_stats_autocorrelation_and_friedman(capsys):
    rc = main(["stats", _vig("KRYPTOS"), "--autocorrelation", "24", "--friedman", "--json"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["autocorrelation"]["max_lag"] == 24
    assert 7 in info["autocorrelation"]["candidate_periods"]
    assert info["friedman"]["estimate"] is not None


def test_cli_stats_autocorrelation_defaults_to_lag_32(capsys):
    rc = main(["stats", _vig("KRYPTOS"), "--autocorrelation", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["autocorrelation"]["max_lag"] == 32


def test_cli_stats_text_mode_renders_both(capsys):
    rc = main(["stats", _vig("KRYPTOS"), "--autocorrelation", "--friedman"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "autocorrelation top lags" in out
    assert "friedman:" in out


def test_cli_stats_unchanged_without_flags(capsys):
    """The new keys are opt-in — no cost or shape change to the default report."""
    rc = main(["stats", _vig("KRYPTOS"), "--json"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert "autocorrelation" not in info
    assert "friedman" not in info
