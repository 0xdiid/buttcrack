"""superimposition_periods: a Kerckhoffs column-alignment period detector.

Fires when a period is a per-column additive shift over an alignable (peaked) inner — a
periodic shift cipher, or a shift laid over a mildly-peaked inner — and stays near the shuffle
floor for a flattener or for no period.
"""

from __future__ import annotations

from buttcrack.analysis import superimposition_periods
from buttcrack.ciphers.bifid import Bifid
from buttcrack.text import only_letters

PROSE = only_letters(
    "THEHARBORMASTERKEEPSALOGOFEVERYVESSELTHATPASSESTHEBREAKWATERANDNOTESTHEWEATH"
    "ERINAMARGINWITHBLUEINKHISDAUGHTERPAINTSSMALLPORTRAITSOFTHECAPTAINSWHILETHEYW"
    "AITFORTHETIDEANDSELLSTHEMFROMABASKETNEARTHECUSTOMSHOUSEONMARKETDAYSSHEEARNSM"
    "ORETHANHEDOES"
)
_A = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _vigenere(pt: str, key: str) -> str:
    return "".join(_A[(ord(c) - 65 + ord(key[i % len(key)]) - 65) % 26] for i, c in enumerate(pt))


def test_fires_at_vigenere_period():
    ct = _vigenere(PROSE, "MEADOW")  # period 6
    res = superimposition_periods(ct, alphabet="STANDARD", max_period=12)
    top = res[0]
    assert top["period"] == 6
    assert top["z"] > 6.0  # a clean per-column shift superimposes decisively
    assert top["merged_ioc"] > 0.06  # columns realign to a language-like pile


def test_quiet_on_plaintext_no_period():
    res = superimposition_periods(PROSE, alphabet="STANDARD", max_period=12)
    assert max(r["z"] for r in res) < 5.0  # no periodic shift structure to find


def test_flattener_stays_near_floor():
    """A bifid flattener has flat columns: no peak to align, so z stays low everywhere."""
    ct = Bifid().encode(PROSE, "GREENHOUSE/13")
    res = superimposition_periods(ct, alphabet="STANDARD", max_period=15)
    assert max(r["z"] for r in res) < 5.0
    # decisively below a Vigenere of the same text
    vig = superimposition_periods(_vigenere(PROSE, "MEADOW"), alphabet="STANDARD", max_period=15)
    assert vig[0]["z"] - max(r["z"] for r in res) > 5.0


def test_ring_argument_and_short_input():
    assert superimposition_periods("SHORT", alphabet="STANDARD") == []
    # a KRYPTOS-ring Vigenere superimposes best when measured in the KRYPTOS ring
    kry = "KRYPTOSABCDEFGHIJLMNQUVWXZ"
    ct = "".join(kry[(kry.index(c) + kry.index("MEADOW"[i % 6])) % 26] for i, c in enumerate(PROSE))
    r_kry = superimposition_periods(ct, alphabet="KRYPTOS", max_period=12)
    assert r_kry[0]["period"] == 6 and r_kry[0]["z"] > 6.0
