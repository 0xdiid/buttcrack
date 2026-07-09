"""Periodic-bigram transposition-layer detection (conservative: no false spikes)."""

from buttcrack.analysis import transposition_periods

PLAIN_ENGLISH = (
    "THEEXPEDITIONREACHEDTHEHIGHPASSJUSTBEFOREDAWNANDFOUNDTHEANCIENTMARKERSEXACTLY"
    "WHERETHEOLDMAPHADPROMISEDEACHSTONEWASCARVEDWITHTHESAMESPIRALEMBLEMANDWECOPIED"
    "THEMCAREFULLYBYLANTERNLIGHTBEFORETHESUNROSEANDTHEMISTBURNEDAWAYREVEALINGTHE"
)


def test_no_false_spike_on_plain_english():
    # Plain English has only low-count incidental bigram recurrences -> filtered out.
    assert transposition_periods(PLAIN_ENGLISH) == []


def test_detects_strong_periodic_structure():
    periodic = "ABCDEFGHIJKLMNOPQRS" * 60  # a hard period-19 repeat
    res = transposition_periods(periodic)
    assert res, "expected a detected period"
    assert any(r["period"] % 19 == 0 for r in res)
