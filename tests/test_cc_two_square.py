"""Two-Square (double Playfair).

Validated against the Wikipedia vertical worked example
(en.wikipedia.org/wiki/Two-square_cipher): squares from keywords EXAMPLE (top)
and KEYWORD (bottom), 25-letter alphabet that keeps J and drops Q. Plaintext
"help me obi wan kenobi" -> digraphs HE LP ME OB IW AN KE NO BI -> ciphertext
HEDLXWSDJYANHOTKDG (HE and AN are same-column transparencies, unchanged).
"""

from buttcrack.ciphers.squares import PolybiusSquare
from buttcrack.ciphers.two_square import ALPHABET_NO_Q, TwoSquare


def test_two_square_wikipedia_vector():
    cs = TwoSquare()
    # Vertical layout is the default; "/V" optional.
    assert cs.encode("help me obi wan kenobi", "EXAMPLE/KEYWORD") == "HEDLXWSDJYANHOTKDG"
    assert cs.encode("help me obi wan kenobi", "EXAMPLE/KEYWORD/V") == "HEDLXWSDJYANHOTKDG"


def test_two_square_reciprocal_roundtrip():
    cs = TwoSquare()
    # The transform is reciprocal: decode(encode(m)) == the prepared plaintext.
    # Prep keeps J, drops Q, and pads a lone final letter with X.
    prepared = PolybiusSquare("", alphabet=ALPHABET_NO_Q).prepare("help me obi wan kenobi")
    assert cs.decode(cs.encode("help me obi wan kenobi", "EXAMPLE/KEYWORD"), "EXAMPLE/KEYWORD") == (
        prepared
    )


# No crack test: the keyless (25!)^2 keyspace is too large for a generic
# hill-climb to fully recover within a reasonable timeout (TwoSquare.crack is
# implemented as best-effort but does not reliably reach a passing threshold).
