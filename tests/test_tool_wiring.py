"""CLI wiring for the previously library-only power modules: `butt keysource`
(cross-document + composed word-pair keys), `butt validate` (synthetic control),
and `butt hillkpa` (Hill known-plaintext attack)."""

import json

from buttcrack.cli import main


def _json(capsys):
    return json.loads(capsys.readouterr().out)


# ----- keysource: compose / decompose word-pair keys -----
def test_keysource_compose_then_decompose_round_trips(capsys):
    rc = main(["keysource", "--compose", "WATERMELON", "LAVENDER", "--json", "--compact"])
    out = _json(capsys)
    assert rc == 0
    # lcm(10, 8) = 40
    assert out["period"] == 40 and len(out["key"]) == 40
    key = out["key"]

    rc = main(
        [
            "keysource",
            "--decompose",
            key,
            "--words",
            "WATERMELON,LAVENDER,MAPLE,RIVERBANK",
            "--json",
            "--compact",
        ]
    )
    dec = _json(capsys)
    assert rc == 0
    pairs = {(p["word_a"], p["word_b"]) for p in dec["pairs"]}
    assert ("WATERMELON", "LAVENDER") in pairs


# ----- keysource: derive candidate keys from a prior-solution corpus -----
def test_keysource_derive_from_corpus_emits_running_key(capsys):
    rc = main(
        [
            "keysource",
            "--corpus",
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
            "--json",
            "--compact",
        ]
    )
    out = _json(capsys)
    assert rc == 0 and out["mode"] == "derive"
    kinds = {c["kind"] for c in out["candidates"]}
    assert "full" in kinds  # the whole text as a running key
    assert any(c["kind"] == "word" for c in out["candidates"])


# ----- keysource: screen the corpus running-keys against a target ciphertext -----
def test_keysource_screens_prior_plaintext_as_running_key(capsys):
    # A running-key Vigenere over KRYPTOS whose key IS a "prior solution" text: desub with
    # the prior plaintext (p = c - k) must snap the IoC toward English.
    kry = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    idx = {c: i for i, c in enumerate(kry)}
    gen_key = (
        "THEHARBORMASTERKEEPSALOGOFEVERYVESSELTHATPASSESTHEBREAKWATERANDNOTESTHEWEATHERINABLUEINK"
    )
    gen_pt = (
        "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGEACHTREEFORRIPEFRUITBYWALL"
    )
    n = 76
    prior, plain = gen_key[:n], gen_pt[:n]
    ct = "".join(kry[(idx[p] + idx[k]) % 26] for p, k in zip(plain, prior, strict=True))
    main(["keysource", ct, "--corpus", prior, "--json", "--compact"])
    out = _json(capsys)
    assert out["mode"] == "screen"
    w = out["screen"].get("winner")
    assert w is not None and w["ioc"] > 0.055  # prior text snaps the IoC


# ----- validate: build a same-structure synthetic control -----
def test_validate_builds_layered_synthetic(capsys):
    rc = main(
        [
            "validate",
            "--structure",
            "substitution-over-columnar",
            "--sub-key",
            "SILVER",
            "--alphabet",
            "KRYPTOS",
            "--columnar-keyword",
            "WATERFALL",
            "--length",
            "120",
            "--json",
            "--compact",
        ]
    )
    out = _json(capsys)
    assert rc == 0
    assert out["structure"] == "substitution-over-columnar" and out["length"] == 120
    assert len(out["ciphertext"]) == 120 and out["ciphertext"] != out["plaintext"]
    assert "qscore_per_char" in out["signature"]


# ----- hillkpa: recover a planted Hill from a crib -----
def test_hillkpa_recovers_planted_hill(capsys):
    from buttcrack.ciphers.hill import Hill

    pt = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGX"  # 36 letters = 12 trigraph blocks
    ct = Hill().encode(pt, "GYBNQKURP")  # classic invertible 3x3 key
    rc = main(
        ["hillkpa", ct, "--crib", pt[:12], "-n", "3", "--alphabet", "STD", "--json", "--compact"]
    )
    out = _json(capsys)
    assert rc == 0 and out["count"] >= 1
    assert out["results"][0]["plaintext"] == pt


# ----- compare: sibling-pair verdict is emitted -----
def test_compare_cli_emits_verdict(capsys):
    from buttcrack.ciphers.vigenere import Vigenere

    vig = Vigenere()
    pt_a = "THEHARBORMASTERKEEPSALOGOFEVERYVESSELTHATPASSESTHEBREAKWATERANDNOTES"
    pt_b = "EARLYINTHEMORNINGTHEGARDENERWALKSTHELONGROWSOFTHEORCHARDCHECKINGTREE"
    ct_a = vig.encode(pt_a, "LEMON")
    ct_b = vig.encode(pt_b, "MELON")
    main(["compare", ct_a, "--with", ct_b, "--json", "--compact"])
    out = _json(capsys)
    assert "freq_profile_l1" in out
    assert "shared_construction" in out["verdict"]


# ----- stats --family: look-elsewhere-corrected period significance -----
def test_stats_family_flags_real_period(capsys):
    from buttcrack.validate import encode_substitution

    eng = (
        "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLE"
        "INHERLEDGERWHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHT"
    )
    ct = encode_substitution(eng, "PORTALS", alphabet="STD")  # period 7
    main(["stats", ct, "--family", "--family-samples", "120", "--json", "--compact"])
    fam = _json(capsys)["period_family"]
    assert fam["coset_ioc"]["best_period"] == 7
    assert fam["coset_ioc"]["family_p"] < 0.05


# ----- nonprose: route vs prose lean -----
def test_nonprose_cli_flags_route_vs_prose(capsys):
    main(
        [
            "nonprose",
            "NORTHSEVENPACESLEFTATTHEOAKEASTTHREEPACESTURNRIGHTPASTWELL",
            "--json",
            "--compact",
        ]
    )
    assert _json(capsys)["leans_nonprose"] is True
    main(
        [
            "nonprose",
            "THEHARBORMASTERKEEPSALOGOFEVERYVESSELTHATPASSESTHEBREAKWATER",
            "--json",
            "--compact",
        ]
    )
    assert _json(capsys)["leans_nonprose"] is False
