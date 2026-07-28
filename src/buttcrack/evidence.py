"""Findings that refuse to be reported until they have earned it.

``nulls`` and ``power`` already provide matched nulls, look-elsewhere correction and
power curves. Nothing makes you *use* them, and that is the whole problem: the
failure mode in long cryptanalytic programs is not a bad search, it is a **clean,
fast, confident number produced by an instrument that was never checked against a
case with a known answer**. Such a number is indistinguishable from a real negative,
so it enters the record as a closure and everything downstream inherits it.

Real examples this module exists to prevent, all observed in practice:

* a solver whose node cap fired on **every** instance, including the true one, with
  the timeouts tallied as rejections — reported as "0 accepts in 20,000";
* a decode with a transposed digit stream, so every score was arithmetic on the
  wrong plaintext — reported as a clean null;
* a driver that only searched block-aligned crib offsets, silently excluding most
  true placements — would have produced a confident null on every phrase forever;
* a call-signature error that turned a worker crash into an empty result list,
  which read as "nothing found";
* a family-MEAN gate reported as a membership test, pruning live hypotheses;
* a per-gauge probability quoted as a per-rule probability, falsely closing three
  hypothesis families at once.

Every one produced plausible output. The only thing that caught any of them was
running the instrument on a case whose answer was already known.

So a :class:`Finding` cannot be rendered without:

1. a **plant gate** — the instrument recovered a synthetic of the *same
   construction and the same plaintext register* (a null inherits the register of
   its plants and does not transfer to registers never tested);
2. a **matched null** — preserving what the objective is given, destroying only what
   is under test;
3. **coverage** — how much of the intended space was actually evaluated, with
   timeouts and caps counted separately from negatives, never as evidence;
4. a **family size** — how many hypotheses the reported best was taken over.

Missing or failing any of these raises :class:`Unverified` rather than printing a
number. That is the point: the type makes the unverified claim *unsayable*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Unverified",
    "PlantGate",
    "Coverage",
    "Finding",
    "MIN_PLANT_RECALL",
]

#: A plant gate below this recall means the instrument cannot reliably find what it
#: is looking for, so its silence carries no information.
MIN_PLANT_RECALL = 0.5


class Unverified(RuntimeError):
    """Raised when a finding is asked to render before it has earned it."""


@dataclass(frozen=True)
class PlantGate:
    """Did the instrument recover synthetics of the same construction and register?

    ``register`` is not decoration. Recall is register-specific (prose ≫ wordlist ≫
    coded), so a null inherits the register of its plants and does not transfer.
    Record what you actually planted.
    """

    recovered: int
    trials: int
    construction: str
    register: str

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("plant gate needs at least one trial")
        if not 0 <= self.recovered <= self.trials:
            raise ValueError("recovered must be within [0, trials]")
        if not self.construction or not self.register:
            raise ValueError("plant gate must name its construction and register")

    @property
    def recall(self) -> float:
        return self.recovered / self.trials

    @property
    def passed(self) -> bool:
        return self.recall >= MIN_PLANT_RECALL

    def summary(self) -> str:
        return (f"plant {self.recovered}/{self.trials} recall={self.recall:.2f} "
                f"[{self.construction}; register={self.register}]")


@dataclass(frozen=True)
class Coverage:
    """What fraction of the intended space was actually evaluated.

    ``timeouts`` and ``capped`` are tracked separately from evaluated cases and are
    NEVER counted as negatives — conflating "the solver gave up" with "no solution
    exists" is the single most common way a search manufactures a false closure.
    """

    evaluated: int
    intended: int
    timeouts: int = 0
    capped: int = 0
    exhaustive: bool = False

    def __post_init__(self) -> None:
        if self.intended <= 0:
            raise ValueError("intended must be positive")
        if self.evaluated < 0 or self.evaluated > self.intended:
            raise ValueError("evaluated must be within [0, intended]")

    @property
    def fraction(self) -> float:
        return self.evaluated / self.intended

    @property
    def complete(self) -> bool:
        """Complete means everything intended was evaluated AND nothing was truncated."""
        return self.evaluated == self.intended and not self.timeouts and not self.capped

    def summary(self) -> str:
        extra = ""
        if self.timeouts or self.capped:
            extra = f", {self.timeouts} timed out, {self.capped} hit caps (NOT negatives)"
        return (f"covered {self.evaluated:,}/{self.intended:,} "
                f"({100 * self.fraction:.1f}%){extra}")


@dataclass
class Finding:
    """A claim plus the attestations required to state it.

    Build it, attach evidence, then call :meth:`render` or :meth:`verdict`. Either
    raises :class:`Unverified` listing exactly what is missing.

    ``family_size`` is how many hypotheses the reported best was maximised over; the
    corrected p-value is the Šidák bound ``1 - (1 - p)**family_size``, which is why
    a best-of-20 at p = 0.03 is not a finding.
    """

    claim: str
    observed: float | None = None
    p_value: float | None = None
    plant: PlantGate | None = None
    coverage: Coverage | None = None
    null_description: str | None = None
    family_size: int = 1
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ builders

    def with_plant(self, recovered: int, trials: int, construction: str,
                   register: str) -> Finding:
        self.plant = PlantGate(recovered, trials, construction, register)
        return self

    def with_null(self, description: str, p_value: float | None = None) -> Finding:
        """``description`` must say what the null PRESERVES and what it DESTROYS."""
        if not description.strip():
            raise ValueError("state what the null preserves and what it destroys")
        self.null_description = description
        if p_value is not None:
            self.p_value = p_value
        return self

    def with_coverage(self, evaluated: int, intended: int, *, timeouts: int = 0,
                      capped: int = 0, exhaustive: bool = False) -> Finding:
        self.coverage = Coverage(evaluated, intended, timeouts, capped, exhaustive)
        return self

    def over_family(self, size: int) -> Finding:
        if size < 1:
            raise ValueError("family_size must be >= 1")
        self.family_size = size
        return self

    # ------------------------------------------------------------------ verdict

    @property
    def corrected_p(self) -> float | None:
        """Šidák look-elsewhere correction for taking a best over `family_size`."""
        if self.p_value is None:
            return None
        p = min(max(self.p_value, 0.0), 1.0)
        return 1.0 - (1.0 - p) ** self.family_size

    def missing(self) -> list[str]:
        """Everything standing between this finding and being reportable."""
        gaps: list[str] = []
        if self.plant is None:
            gaps.append(
                "no plant gate: the instrument was never shown to recover a synthetic "
                "of this construction, so its silence carries no information")
        elif not self.plant.passed:
            gaps.append(
                f"plant gate FAILED ({self.plant.summary()}): recall below "
                f"{MIN_PLANT_RECALL:.0%} means a null is uninformative, not negative")
        if self.null_description is None:
            gaps.append("no matched null: state what it preserves and what it destroys")
        if self.coverage is None:
            gaps.append("no coverage: report evaluated/intended, timeouts and caps")
        elif self.coverage.evaluated == 0:
            gaps.append("coverage is zero — nothing was actually evaluated")
        return gaps

    def verdict(self) -> str:
        """One of: 'closed', 'null (partial)', 'inconclusive', 'positive'."""
        gaps = self.missing()
        if gaps:
            raise Unverified(
                f"cannot state {self.claim!r}:\n  - " + "\n  - ".join(gaps))
        assert self.coverage is not None
        cp = self.corrected_p
        if cp is not None and cp < 0.05:
            return "positive"
        if self.coverage.complete or self.coverage.exhaustive:
            return "closed"
        if self.coverage.timeouts or self.coverage.capped:
            return "inconclusive"
        return "null (partial)"

    def render(self) -> str:
        v = self.verdict()          # raises Unverified if unearned
        assert self.plant is not None and self.coverage is not None
        lines = [f"{self.claim}", f"  verdict     : {v}"]
        if self.observed is not None:
            lines.append(f"  observed    : {self.observed:.4f}")
        if self.p_value is not None:
            lines.append(f"  p           : {self.p_value:.4g} raw, "
                         f"{self.corrected_p:.4g} corrected over "
                         f"{self.family_size} hypotheses")
        lines.append(f"  {self.plant.summary()}")
        lines.append(f"  null        : {self.null_description}")
        lines.append(f"  {self.coverage.summary()}")
        for n in self.notes:
            lines.append(f"  note        : {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        v = self.verdict()
        assert self.plant is not None and self.coverage is not None
        return {
            "claim": self.claim,
            "verdict": v,
            "observed": self.observed,
            "p_value": self.p_value,
            "corrected_p": self.corrected_p,
            "family_size": self.family_size,
            "plant": {
                "recovered": self.plant.recovered, "trials": self.plant.trials,
                "recall": self.plant.recall, "construction": self.plant.construction,
                "register": self.plant.register,
            },
            "null": self.null_description,
            "coverage": {
                "evaluated": self.coverage.evaluated, "intended": self.coverage.intended,
                "timeouts": self.coverage.timeouts, "capped": self.coverage.capped,
                "complete": self.coverage.complete,
            },
            "notes": list(self.notes),
        }

    def __str__(self) -> str:  # rendering an unverified finding must not silently work
        return self.render()
