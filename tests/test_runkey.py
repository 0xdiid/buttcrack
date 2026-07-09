"""Tests for the running-key screen (buttcrack.runkey).

Synthetic, control-gated: a running key (a long, non-repeating text) used as the key of a
periodic substitution applied OVER a columnar. The screen must rank
the true key text as a lone IoC outlier and peel the columnar to exact English."""
from buttcrack.runkey import desub_ioc, running_desub, screen_running_keys
from buttcrack.scoring import get_scorer
from buttcrack.text import only_letters
from buttcrack.validate import encode_columnar, encode_substitution

# A long running key (no repeat within the message) and two decoy key-texts.
KEY = only_letters(
    "WHENINTHECOURSEOFHUMANEVENTSITBECOMESNECESSARYFORONEPEOPLETODISSOLVETHEPOLITICAL"
    "BANDSWHICHHAVECONNECTEDTHEMWITHANOTHERANDTOASSUMEAMONGTHEPOWERSOFTHEEARTHTHESEPA"
)
DECOYS = [
    only_letters("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGAGAINANDAGAINUNTILTHEMORNINGLIGHT"
                 "FELLACROSSTHEQUIETFIELDSWHERETHEHARVESTHADLONGSINCEBEENGATHEREDIN"),
    only_letters("LOREMIPSUMDOLORSITAMETCONSECTETURADIPISCINGELITSEDDOEIUSMODTEMPORINCI"
                 "DIDUNTUTLABOREETDOLOREMAGNAALIQUAENIMADMINIMVENIAMQUISNOSTRUD"),
]


def _synth(plaintext, keyword="ZEBRAS", convention="vigenere", alphabet="KRYPTOS"):
    """CT = Sub(Columnar(PT)) — substitution OUTER over a columnar."""
    pt = only_letters(plaintext)[:200]
    ct = encode_substitution(encode_columnar(pt, keyword), KEY,
                             substitution=convention, alphabet=alphabet)
    return pt, ct


def test_desub_ioc_outlier_gap(plaintext):
    _, ct = _synth(plaintext)
    good, _ = desub_ioc(ct, KEY, alphabet="KRYPTOS", convention="vigenere")
    bad, _ = desub_ioc(ct, DECOYS[0], alphabet="KRYPTOS", convention="vigenere")
    assert good > 0.060  # de-sub with the true key snaps to transposed-English
    assert bad < 0.050   # a wrong key stays near the random floor


def test_screen_ranks_true_keytext_and_peels(plaintext):
    pt, ct = _synth(plaintext)
    res = screen_running_keys(ct, [KEY, *DECOYS], labels=["true", "d0", "d1"],
                              scorer=get_scorer())
    assert res["recovered"] is True
    w = res["winner"]
    assert w["label"] == "true"
    assert w["alphabet"] == "KRYPTOS"
    assert w["convention"] == "vigenere"
    assert w["transposed_english"] is True
    assert w["z_outlier"] >= 3.0                       # a lone outlier vs the decoys
    assert res["structure"]["transposition"] == "columnar"
    assert pt[:32] in res["plaintext"]                 # exact-span recovery


def test_rejects_when_true_key_absent(plaintext):
    _, ct = _synth(plaintext)
    res = screen_running_keys(ct, DECOYS, labels=["d0", "d1"], scorer=get_scorer())
    assert res["recovered"] is False                   # no decoy reads as English


def test_pure_running_key_no_transposition(plaintext):
    """A pure running-key substitution (no columnar) is read directly (width 1)."""
    pt = only_letters(plaintext)[:200]
    ct = encode_substitution(pt, KEY, substitution="vigenere", alphabet="KRYPTOS")
    res = screen_running_keys(ct, [KEY, *DECOYS], labels=["true", "d0", "d1"],
                              scorer=get_scorer())
    assert res["recovered"] is True
    assert res["winner"]["label"] == "true"
    assert res["structure"] == {"transposition": None}
    assert pt[:40] in res["plaintext"]


def test_beaufort_alphabet_variants_are_found(plaintext):
    """The screen sweeps Beaufort and the standard alphabet too."""
    pt, ct = _synth(plaintext, convention="beaufort", alphabet="STD")
    res = screen_running_keys(ct, [KEY, *DECOYS], labels=["true", "d0", "d1"],
                              scorer=get_scorer())
    assert res["recovered"] is True
    assert res["winner"]["convention"] == "beaufort"
    assert res["winner"]["alphabet"] == "STD"
    assert pt[:32] in res["plaintext"]


def test_single_keytext_z_outlier_is_none(plaintext):
    """With one candidate key there is nothing to be an outlier against -> z_outlier None,
    no divide-by-zero blow-up; recovery is still judged by word coverage."""
    pt, ct = _synth(plaintext)
    res = screen_running_keys(ct, [KEY], labels=["only"], scorer=get_scorer())
    assert res["winner"]["z_outlier"] is None
    assert res["recovered"] is True
    assert pt[:32] in res["plaintext"]


def test_empty_alphabets_returns_clean_envelope(plaintext):
    """Empty alphabets/conventions must not raise (regression: ZeroDivisionError)."""
    _, ct = _synth(plaintext)
    res = screen_running_keys(ct, [KEY], alphabets=(), scorer=get_scorer())
    assert res["ok"] is False and res["recovered"] is False
    assert res["winner"] is None and res["trials"] == 0


def test_key_index_survives_no_letter_drop(plaintext):
    """A dropped no-letter key-text must not shift the surviving keys' reported key_index."""
    _, ct = _synth(plaintext)
    res = screen_running_keys(ct, ["12345", DECOYS[0], KEY],  # entry 0 has no letters -> dropped
                              labels=["empty", "decoy", "true"], scorer=get_scorer())
    assert res["winner"]["label"] == "true"
    assert res["winner"]["key_index"] == 2          # original position, not post-filter index 1


def test_running_desub_matches_validate_decode(plaintext):
    """running_desub is semantically identical to validate.decode_substitution."""
    from buttcrack.validate import decode_substitution
    _, ct = _synth(plaintext)
    for alph in ("KRYPTOS", "STD"):
        for conv in ("vigenere", "beaufort", "variant"):
            a = running_desub(ct, KEY, alphabet=alph, convention=conv)
            b = decode_substitution(ct, KEY, substitution=conv, alphabet=alph)
            assert a == b
