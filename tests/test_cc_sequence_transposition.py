"""Tests for the Sequence Transposition cipher.

Two independent authoritative worked examples are reproduced exactly:

1. ACA cipher sheet (cryptogram.org aca.info ciphers/SequenceTransposition.pdf):
   pt "THE EARLY BIRD GETS THE WORM.", keyphrase GUMMYBEARS (-> 4956023178),
   primer 69315, sequence 6931552460760673663092..., ciphertext (regrouped)
   "69315 YHOMA RTBDE THIGW LRESE ERT".

2. CryptoCrack user guide, Sequence Transposition page
   (sites.google.com/site/cryptocrackprogram .../other/sequence-transposition):
   pt "Only the mediocre are always at their best.", keyword CRYPTOGRAM
   (-> 2706953814), primer 31752, ciphertext (regrouped)
   "31752 TMIAE TELEO LDREC RTYSO RYIEE BNAWA SAHTH".

Both use the same numbering scheme (alphabetical rank 1..9 then 0).
"""

from __future__ import annotations

import random

import pytest

from buttcrack.ciphers.sequence_transposition import SequenceTransposition
from buttcrack.scoring import get_scorer

# --- ACA cipher-sheet vector ---
ACA_PLAINTEXT = "THEEARLYBIRDGETSTHEWORM"
ACA_KEY = "69315/GUMMYBEARS"
ACA_CIPHERTEXT = "YHOMARTBDETHIGWLRESEERT"

# --- CryptoCrack user-guide vector ---
CC_PLAINTEXT = "ONLYTHEMEDIOCREAREALWAYSATTHEIRBEST"
CC_KEY = "31752/CRYPTOGRAM"
CC_CIPHERTEXT = "TMIAETELEOLDRECRTYSORYIEEBNAWASAHTH"


def test_vector_encode_aca():
    assert SequenceTransposition().encode(ACA_PLAINTEXT, ACA_KEY) == ACA_CIPHERTEXT


def test_vector_encode_cryptocrack():
    assert SequenceTransposition().encode(CC_PLAINTEXT, CC_KEY) == CC_CIPHERTEXT


def test_vector_decode_aca():
    assert SequenceTransposition().decode(ACA_CIPHERTEXT, ACA_KEY) == ACA_PLAINTEXT


def test_vector_decode_cryptocrack():
    assert SequenceTransposition().decode(CC_CIPHERTEXT, CC_KEY) == CC_PLAINTEXT


def test_roundtrip():
    c = SequenceTransposition()
    msg = "ATTACKATDAWNTHEENEMYISNEARANDWEMUSTHOLDTHELINEUNTILREINFORCEMENTSARRIVE"
    key = "58032/CRYPTOGRAM"
    assert c.decode(c.encode(msg, key), key) == msg


def test_roundtrip_header_key_form():
    # The "<primer>/<header digits>" form decodes identically to the keyword form
    # whose ranking produced those headers (GUMMYBEARS -> 4956023178).
    c = SequenceTransposition()
    assert c.decode(ACA_CIPHERTEXT, "69315/4,9,5,6,0,2,3,1,7,8") == ACA_PLAINTEXT


@pytest.mark.slow
def test_crack_with_primer_hint():
    scorer = get_scorer()
    c = SequenceTransposition()
    rng = random.Random(11)
    plaintext = (
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILETHESLEEPINGHOUNDDREAMS"
        "OFCHASINGRABBITSACROSSTHEMEADOWUNDERAPALEMORNINGSKY"
    )
    key = "47185/CRYPTOGRAM"
    ct = c.encode(plaintext, key)
    out = c.crack(ct, scorer, top=5, rng=rng, timeout=60.0, primer="47185")
    assert out, "expected at least one candidate"
    assert out[0].plaintext.upper().replace(" ", "") == plaintext
