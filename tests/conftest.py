import pytest

from buttcrack.text import only_letters

# A long, ordinary-English plaintext — enough statistical signal for cracking.
PLAINTEXT = (
    "the analysis routines need a healthy stretch of perfectly ordinary english "
    "prose so that the frequency statistics and quadgram fitness scores can lock "
    "onto the underlying message and recover the original text without any prior "
    "knowledge of the secret key that was chosen to encipher it in the beginning"
)


@pytest.fixture
def plaintext():
    return PLAINTEXT


def letters(s: str) -> str:
    return only_letters(s)
