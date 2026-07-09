"""Layered-cipher (superencipherment) decode pipeline.

Vector: a KRYPTOS-keyed Vigenere (Quagmire III) over a double columnar
transposition. encrypt = columnar(TELESCOPE) ->
columnar(HURRICANE) -> quagmire3(KRYPTOS/MEADOW/K); decrypt reverses it.
"""

from buttcrack import engine
from buttcrack.text import only_letters

PLAINTEXT = (
    "THEOLDLIGHTHOUSEKEEPERCLIMBSTHENARROWSTAIRSEACHEVENINGTOTRIMTHELAMPANDPOLISHTH"
    "EGREATGLASSLENSHEHASCOUNTEDTHESTEPSFORTHIRTYYEARSANDKNOWSEVERYWORNSTONEBYHEART"
    "FROMTHEGALLERYHEWATCHESTHEFISHINGBOATSRETURNBEFOREDARKANDMARKSEACHSAILINASMALL"
    "LEDGERTHATHISFATHERKEPTBEFOREHIMWHENTHEFOGROLLSINHERINGSTHEGREATBELLTWICEANDWA"
    "ITSFORTHEANSWERINGHORN"
)

ENCRYPT_STEPS = [
    ("columnar", "TELESCOPE"),
    ("columnar", "HURRICANE"),
    ("quagmire3", "KRYPTOS/MEADOW/K"),
]
DECRYPT_STEPS = [
    ("quagmire3", "KRYPTOS/MEADOW/K"),
    ("columnar", "HURRICANE"),
    ("columnar", "TELESCOPE"),
]


def test_pipeline_round_trip_layered():
    ciphertext, enc_trace = engine.pipeline(PLAINTEXT, ENCRYPT_STEPS, op="encode")
    assert len(enc_trace) == 3
    plain, dec_trace = engine.pipeline(ciphertext, DECRYPT_STEPS, op="decode")
    assert plain == PLAINTEXT
    assert [t["cipher"] for t in dec_trace] == ["quagmire3", "columnar", "columnar"]


_LONG_PT = (
    "THEEXPEDITIONREACHEDTHEHIGHPASSJUSTBEFOREDAWNANDFOUNDTHEANCIENTMARKERSEXACTLY"
    "WHERETHEOLDMAPHADPROMISEDEACHSTONEWASCARVEDWITHTHESAMESPIRALEMBLEMANDWECOPIED"
    "THEMCAREFULLYBYLANTERNLIGHTBEFORETHESUNROSEANDTHEMISTBURNEDAWAYREVEALINGTHE"
    "VALLEYFARBELOWWHERETHELOSTCITYWASSAIDTOLIESILENTBENEATHCENTURIESOFDRIFTINGSAND"
)


def test_auto_cracks_additive_substitution_over_transposition():
    """The tractable layered case: Vigenere outer + columnar inner is auto-cracked."""
    inner = engine.encode("columnar", _LONG_PT, "CARGO").best().plaintext
    ct = engine.encode("vigenere", inner, "LEMONS").best().plaintext
    result = engine.auto(ct, seed=1)
    best = result.candidates[0]
    assert best.cipher == "vigenere+columnar"
    assert only_letters(best.plaintext) == _LONG_PT
    assert best.meta.get("layered") is True


def test_auto_does_not_mislabel_pure_transposition_as_layered():
    ct = engine.encode("columnar", _LONG_PT, "CARGO").best().plaintext
    result = engine.auto(ct, seed=1)
    assert "+" not in result.candidates[0].cipher  # plain columnar, not "vigenere+..."


def test_pipeline_decodes_known_ciphertext():
    ciphertext = (
        "DBUHNTNLHHJJWIYRJJTZMLJBALCXDRNVUREHPTKQMPATGUBRTNMWXEXMMREXVZQHNJZVGACBMIUISBZI"
        "VLSXZYQJSTVFIZSYPFZXXBGVJHCBPITQFBZXVEJYUTHQJJAQMPORXVBESXVJJUEISVMEXXQJMHCYSFEZ"
        "QBRIFUCTAFMYCTUZXHQZRYJJFDENZLDNXNPQFEZFLPNBZVVHQEMLGVVHCWKRFHTYLXYREBAIMHXKBXFP"
        "DJKPCYXBKAFBAGNXEPKIHVSDUSMUDEVHQRNBNLEQSAQMVXSTAVAPJQSFXJNEKZHWNBSWQHDBCLEUJBAJ"
        "ZEJTPKKJEYPJUE"
    )
    plain, _ = engine.pipeline(ciphertext, DECRYPT_STEPS, op="decode")
    assert plain == PLAINTEXT
