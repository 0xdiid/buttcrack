"""Edge cases and the hardened error/trust contract."""

import json
import time

import pytest

from buttcrack import engine, registry
from buttcrack.cli import main
from buttcrack.identify import identify as run_identify
from buttcrack.scoring import get_scorer


# -------------------------------------------------------------- error envelope
def test_unknown_cipher_returns_clean_error(capsys):
    rc = main(["crack", "notacipher", "hello there", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["ok"] is False
    assert "notacipher" in out["error"]


def test_bad_key_returns_clean_error(capsys):
    rc = main(["encode", "affine", "hello", "--key", "2,5", "--json"])  # a=2 not coprime
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["ok"] is False
    assert out["error_type"] == "ValueError"


def test_missing_input_is_structured_error(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)  # no piped stdin
    rc = main(["crack", "caesar", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2 and out["ok"] is False


# ------------------------------------------------------------- auto resilience
@pytest.mark.slow
def test_auto_survives_bogus_cipher_in_selection(capsys, plaintext):
    ct = registry.get("caesar").encode(plaintext, "5")
    rc = main(["auto", ct, "--ciphers", "caesar,bogus", "--json", "--seed", "1"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0  # caesar still solves it
    assert out["candidates"][0]["cipher"] == "caesar"
    assert any("bogus" in n for n in out.get("notes", []))


# -------------------------------------------------------------------- identify
def test_identify_empty_is_undetermined():
    info = run_identify("")
    assert info["reliable"] is False
    assert info["likely_families"][0]["family"] == "undetermined"


def test_identify_digits_only_is_undetermined():
    info = run_identify("12345 !!! ---")
    assert info["likely_families"][0]["family"] == "undetermined"


# ------------------------------------------------------------------ confidence
def test_confidence_is_low_on_tiny_input():
    scorer = get_scorer()
    assert scorer.confidence("THEM") < 0.3  # one lucky quadgram
    assert scorer.confidence("THETHETHE") < 0.5  # repetitive
    assert scorer.confidence("") == 0.0


def test_confidence_high_on_real_paragraph(plaintext):
    assert get_scorer().confidence(plaintext) > 0.8


# ----------------------------------------------------------------- ok / verdict
def test_letterless_input_is_not_ok(capsys):
    rc = main(["crack", "caesar", "12345", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["verdict"] == "no-candidates"
    assert rc == 1


def test_verdict_present_and_valid(capsys, plaintext):
    ct = registry.get("caesar").encode(plaintext, "8")
    main(["crack", "caesar", ct, "--json", "--seed", "1"])
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] in {"solved", "likely", "ambiguous", "unlikely", "no-candidates"}
    assert out["verdict"] in {"solved", "likely"}  # a clean Caesar should be trusted


# -------------------------------------------------------------------- timeouts
@pytest.mark.slow
def test_columnar_respects_timeout():
    # max_width 9 would be 9! permutations; a 0.2s budget must bound the wall time.
    letters = "WEAREDISCOVEREDFLEEATONCEBEFORETHEGUARDSCHANGE" * 2
    start = time.perf_counter()
    result = engine.crack("columnar", letters, timeout=0.2, max_width=9)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
    assert isinstance(result.candidates, list)
