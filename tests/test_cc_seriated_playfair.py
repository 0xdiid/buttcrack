"""Tests for the Seriated Playfair cipher.

VECTOR source: CryptoCrack user guide, "Seriated Playfair"
(sites.google.com/site/cryptocrackprogram/user-guide/cipher-types/other/
seriated-playfair). Keyword ``SERIATEDPLAYFAIR`` (square S E R I A / T D P L Y /
F B C G H / K M N O Q / U V W X Z), period 7. The plaintext is Babbage's Rule
from *The Codebreakers* by David Kahn -- "No man's cipher is worth looking at
unless the inventor has himself solved a very difficult cipher" -- prefixed by
the attribution "Babbage's Rule" and suffixed by "X The Codebreakers by Kahn X",
giving the seriated stream below. It enciphers to the published groups
``FSFGSCI EIVDROM QSWEFRL BRPARXN IPFYKKM AKHILXO GREEPFI LMURKYM IBITFLO
EAYKAXD ZDLSURP DBEHBAN VPLFADF SIUPGRG QBRFEAS MDIECDQ RGQZ``.
"""

import random

import pytest

from buttcrack.ciphers.seriated_playfair import SeriatedPlayfair
from buttcrack.scoring import get_scorer

# The prepared seriation stream (letters only, J->I, with the author's X word
# separator after CIPHER and the trailing X null already present).
_VECTOR_PT = (
    "BABBAGESRULENOMANSCIPHERISWORTHLOOKINGATUNLESSTHEINVENTORHASHIMSELF"
    "SOLVEDAVERYDIFFICULTCIPHERXTHECODEBREAKERSBYKAHNX"
)
_VECTOR_CT = (
    "FSFGSCIEIVDROMQSWEFRLBRPARXNIPFYKKMAKHILXOGREEPFILMURKYMIBITFLO"
    "EAYKAXDZDLSURPDBEHBANVPLFADFSIUPGRGQBRFEASMDIECDQRGQZ"
)


def test_vector_cryptocrack_seriated_playfair():
    c = SeriatedPlayfair()
    assert c.encode(_VECTOR_PT, "SERIATEDPLAYFAIR/7") == _VECTOR_CT


def test_round_trip():
    # decode(encode(msg)) recovers the prepared seriated stream: letters only,
    # J->I, with X nulls inserted to break vertical doubles and pad the final
    # block. Encryption (right/down) and decryption (left/up) differ.
    c = SeriatedPlayfair()
    key = "MONARCHY/5"
    msg = "Attack the eastern junction at dawn before the fog lifts, old chap!"
    ct = c.encode(msg, key)
    # decode recovers the seriated plaintext stream; re-encoding it is stable.
    recovered = c.decode(ct, key)
    assert c.encode(recovered, key) == ct
    assert recovered  # non-empty


def test_round_trip_doubles_and_padding():
    # A double-letter-heavy message exercises the X-null insertion and the short
    # final-block padding; the recovered seriated stream must re-encipher identically.
    c = SeriatedPlayfair()
    key = "KEYWORD/4"
    msg = "BALLOON FELL OFF THE MISSISSIPPI BLUFF SLOWLY"
    ct = c.encode(msg, key)
    recovered = c.decode(ct, key)
    assert c.encode(recovered, key) == ct


@pytest.mark.slow
def test_crack_recovers():
    c = SeriatedPlayfair()
    scorer = get_scorer()
    pt = (
        "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOF"
        "FOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASON"
        "OFLIGHTITWASTHESEASONOFDARKNESSITWASTHESPRINGOFHOPEITWASTHEWINTEROFDESPAIR"
        "WEHADEVERYTHINGBEFOREUSWEHADNOTHINGBEFOREUSWEWEREALLGOINGDIRECTTOHEAVEN"
    )
    key = "CHARLES/7"
    prepared = c.encode(pt, key)  # ciphertext
    recovered_pt = c.decode(prepared, key)  # the seriated stream crack targets
    ct = c.encode(pt, key)
    res = c.crack(ct, scorer, rng=random.Random(7), timeout=240)
    assert res, "crack returned no candidates"
    assert res[0].plaintext == recovered_pt
