"""Structure-triage command (`butt diagnose`)."""

import json

from buttcrack import registry
from buttcrack.cli import main
from buttcrack.diagnose import diagnose


def test_diagnose_periodic_vigenere(plaintext):
    ct = registry.get("vigenere").encode(plaintext, "SECRET")  # period 6
    info = diagnose(ct)
    assert "periodic polyalphabetic" in info["structure"]
    assert info["signals"]["calibrated_periods"][0]["period"] in (6, 12)
    assert any("vigenere" in r or "quagmire" in r for r in info["recommended"])


def test_diagnose_transposition(plaintext):
    ct = registry.get("columnar").encode(plaintext, "ZEBRA")
    info = diagnose(ct)
    assert "transposition" in info["structure"]
    assert any("columnar" in r for r in info["recommended"])


def test_diagnose_non_stationary_keystream():
    # Increasing per-position randomness (larger alphabet per quarter) -> IoC decays with
    # no clean period: reads as an evolving keystream / no recoverable period.
    import random

    rng = random.Random(1)
    sizes = [18, 21, 24, 26]
    q = 320 // 4
    text = "".join(
        "".join(chr(65 + rng.randrange(s)) for _ in range(q if i < 3 else 320 - 3 * q))
        for i, s in enumerate(sizes)
    )
    info = diagnose(text)
    assert info["structure"] in (
        "non-stationary / evolving keystream",
        "polyalphabetic, no recoverable period",
    )
    assert info["recommended"]  # always proposes something to try


def test_diagnose_flags_homophonic_expansion():
    # tri-square-family shape: each plaintext letter -> trigraph with C1 = P + C2 + C3 (mod 26)
    # in the KRYPTOS alphabet; flat IoC, no period, no repeats — only the linear-relation scan
    # sees it.
    import random

    from buttcrack.keysources import KRYPTOS

    plain = (
        "THENAVIGATORPLOTSACOURSEBYTHEWINTERSTARSANDCHECKSITTWICEAGAINSTTHECOMPASSBE"
        "FOREDAWNTHECREWHAULSTHENETSABOARDANDSORTSTHESILVERFISHINTOW"
    )
    rng = random.Random(5)
    idx = {c: i for i, c in enumerate(KRYPTOS)}
    ct = []
    for p in plain:
        c2, c3 = rng.randrange(26), rng.randrange(26)
        c1 = (idx[p] + c2 + c3) % 26
        ct += [KRYPTOS[c1], KRYPTOS[c2], KRYPTOS[c3]]
    info = diagnose("".join(ct))
    assert "LINEAR RELATION" in info["summary"]
    assert any(r.startswith("butt relation") for r in info["recommended"])
    rel = info["signals"]["linear_relation_n3"]
    assert rel["candidates"][0]["coef"] in ([-1, 1, 1], [1, -1, -1])


def test_diagnose_infers_outer_substitution_over_fractionation():
    # All 26 letters (incl J), flat IoC, no linear channel, no block alignment: the emitted
    # alphabet itself implies an OUTER SUBSTITUTION over a <=25-cell fractionation (a bifid square
    # cannot emit J; a trifid needs filler symbols, absent here).
    import random

    rng = random.Random(2)
    flat = "".join(chr(65 + rng.randrange(26)) for _ in range(320))
    info = diagnose(flat)
    assert "inferences" in info
    assert any("OUTER SUBSTITUTION" in s for s in info["inferences"])
    # The whole set of 26 letters must actually be present for the inference to be sound.
    assert len(set(flat)) == 26
    # Layer-count humility rule also fires on any flattened reading.
    assert any(">=1 flattening layer" in s for s in info["inferences"])


def test_diagnose_reports_period7_as_real_not_small_sample(plaintext):
    # A genuine period-7 Vigenere on a long message is a REAL period: reported as periodic, with
    # no small-sample caveat (its cosets hold plenty of letters).
    ct = registry.get("vigenere").encode(plaintext, "RAINBOW")  # period 7
    info = diagnose(ct)
    assert "periodic polyalphabetic" in info["structure"]
    assert info["signals"]["calibrated_periods"][0]["period"] in (7, 14)
    assert not any("small-sample" in s for s in info.get("inferences", []))


def test_diagnose_cli_json(capsys, plaintext):
    ct = registry.get("vigenere").encode(plaintext, "SECRET")
    main(["diagnose", ct, "--json"])
    info = json.loads(capsys.readouterr().out)
    for field in ("length", "structure", "summary", "recommended", "signals"):
        assert field in info
    assert isinstance(info["recommended"], list) and info["recommended"]
