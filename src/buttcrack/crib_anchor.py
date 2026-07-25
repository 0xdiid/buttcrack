"""Crib-anchored scoring for register-resistant cracks.

When a keyed cipher's plaintext is terse, name-heavy, numeric, or otherwise **not
flowing prose**, a pure n-gram objective loses its grip: the true key's decode
scores only middling, and an SA / hill-climb ranks it *below* over-fit junk. This
is the "register hole" — every fluency-based attack walks past the answer.

If you can guess a **contiguous crib** — a word or phrase almost certain to appear
(a recurring proper noun, a stock diary opener, a set phrase, a spelled year) —
you can anchor the search on it instead of on fluency. This module scores the
best-over-position appearance of the crib in a *decoded candidate*, to be **added
to** (and weighted to dominate) the n-gram term as the SA objective. For the true
key the crib lands in full and the score leaps clear of the plateau even when the
plaintext is unfluent; wrong keys cannot place the whole crib.

The primitive is decoder-agnostic: hand it any candidate plaintext string, from a
Bifid square-climb, a Quagmire solve, a columnar undo — whatever produced it.

Design rules (measured empirically on a short fractionation cipher whose plaintext
defeated n-gram scoring):

* **One contiguous crib, not scattered fragments.** A multi-anchor objective
  (place word A *and* word B *and* word C) is too rugged — the true key's basin
  is tiny and the climb almost never finds it, even at large restart budgets.
* **Length ~16-20 letters is the sweet spot.** Shorter (< ~13) *floods*: a short
  crib is placeable somewhere in junk by many wrong keys, so it fails to separate
  the true key (e.g. a 10-letter name is out-ranked by ~10^5 junk keys). Longer
  (> ~24) *hurts convergence*: the climber cannot place all of it at a modest
  restart budget, and the true key sinks in the ranking. See :data:`SWEET_SPOT`.
* **Weight the crib term to dominate.** Choose ``weight`` so a full placement
  outweighs the n-gram spread between the true and junk keys; the n-gram term then
  only breaks ties among keys that place the crib equally.
"""

from __future__ import annotations

from .scoring import NgramScorer, get_scorer
from .text import only_letters

#: Contiguous-crib length window that both separates (long enough that junk keys
#: cannot place it by chance) and converges (short enough for a hill-climb to
#: place in full at a modest restart budget).
SWEET_SPOT = (16, 20)


def best_position_match(text: str, crib: str) -> tuple[int, int]:
    """Best-over-position letter match of ``crib`` against ``text``.

    Slides ``crib`` across ``text`` and returns ``(matches, position)``: the
    maximum number of coincident letters over all offsets, and the offset that
    achieves it. Both strings are reduced to A-Z first. Returns ``(0, 0)`` when
    the crib is empty or longer than the text.
    """
    t = only_letters(text.upper())
    c = only_letters(crib.upper())
    n = len(c)
    if n == 0 or n > len(t):
        return 0, 0
    best, best_pos = -1, 0
    for p in range(len(t) - n + 1):
        m = sum(t[p + j] == c[j] for j in range(n))
        if m > best:
            best, best_pos = m, p
    return best, best_pos


def crib_bonus(text: str, crib: str, *, weight: float = 1.0) -> float:
    """``weight`` times the best-over-position match count of ``crib`` in ``text``."""
    return weight * best_position_match(text, crib)[0]


def crib_length_advice(crib: str) -> str:
    """Human-readable verdict on a crib's length for anchored SA (see module docs)."""
    n = len(only_letters(crib.upper()))
    if n < 13:
        return "too short: will flood (a short crib is placeable in junk by many keys)"
    if n > 24:
        return "too long: hurts hill-climb convergence at a modest restart budget"
    if SWEET_SPOT[0] <= n <= SWEET_SPOT[1]:
        return "ideal"
    return "usable"


class CribAnchoredScorer:
    """SA objective = n-gram score + ``weight`` * best crib placement.

    Wrap the n-gram scorer a solver already uses and feed it candidate *decoded
    plaintexts*; use :meth:`score` as the climb objective and :meth:`placement`
    to gate / report survivors. See the module docstring for how to choose the
    crib and ``weight``.
    """

    def __init__(
        self,
        crib: str,
        *,
        weight: float = 30.0,
        scorer: NgramScorer | None = None,
    ) -> None:
        self.crib = only_letters(crib.upper())
        self.weight = float(weight)
        self.scorer = scorer or get_scorer()

    def score(self, text: str) -> float:
        """n-gram fitness of ``text`` plus the weighted crib placement bonus."""
        return self.scorer.score(text) + crib_bonus(text, self.crib, weight=self.weight)

    def placement(self, text: str) -> tuple[int, int]:
        """``(matches, position)`` of the crib in ``text`` — for gating/reporting."""
        return best_position_match(text, self.crib)

    @property
    def full(self) -> int:
        """Match count that constitutes a *full* placement of the crib."""
        return len(self.crib)
