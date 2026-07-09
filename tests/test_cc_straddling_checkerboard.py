"""Tests for the Straddling Checkerboard cipher."""

from __future__ import annotations

from buttcrack.ciphers.straddling_checkerboard import StraddlingCheckerboard

# The classic Wikipedia board (blank columns 2 and 6):
#       0 1 2 3 4 5 6 7 8 9
#         E T   A O N   R I S
#     2 | B C D F G H J K L M
#     6 | P Q / U V W X Y Z .
# Top-row + row2 + row6 reading order, with "/." as the last two cells.
WIKI_KEY = "ETAONRISBCDFGHJKLMPQ/UVWXYZ./26"


def test_vector():
    # Source: Wikipedia "Straddling checkerboard" worked example.
    # Plaintext ATTACK AT DAWN -> 3113212731223655
    # (A=3, T=1, C=21, K=27, D=22, W=65, N=5).
    c = StraddlingCheckerboard()
    out = c.encode("ATTACK AT DAWN", WIKI_KEY)
    assert out.replace(" ", "") == "3113212731223655"


def test_roundtrip_letters():
    c = StraddlingCheckerboard()
    msg = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    assert c.decode(c.encode(msg, WIKI_KEY), WIKI_KEY) == msg


def test_roundtrip_with_digits():
    c = StraddlingCheckerboard()
    msg = "MEETATDAWNON9"
    assert c.decode(c.encode(msg, WIKI_KEY), WIKI_KEY) == msg


def test_roundtrip_keyword_board():
    c = StraddlingCheckerboard()
    key = "ARABESQUE/37"
    msg = "ATTACKATDAWN"
    assert c.decode(c.encode(msg, key), key) == msg


# NOTE: no crack test. crack() is implemented as best-effort random-restart
# hill climbing but does NOT reliably recover the plaintext: the fractionated
# digit stream means a single mis-placed board cell mis-aligns the whole
# downstream parse, so the n-gram fitness landscape is too deceptive to climb.
