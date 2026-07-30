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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Unverified",
    "PlantGate",
    "Coverage",
    "Finding",
    "MIN_PLANT_RECALL",
    "MIN_POWER_Z",
    "searched_fraction",
]

#: A plant gate below this recall means the instrument cannot reliably find what it
#: is looking for, so its silence carries no information.
MIN_PLANT_RECALL = 0.5

#: A stated power below this z means the statistic could not have seen the signal it
#: was pointed at, so its null is SILENT (uninformative), not evidence of absence.
MIN_POWER_Z = 3.0


def searched_fraction(axes: Mapping[str, int], completed: int) -> dict[str, Any]:
    """Multiply out the declared free axes of a search and report the fraction done.

    A campaign repeatedly said "the search is finished" while quietly holding several
    axes fixed (an orientation here, a gauge there); multiplying the *declared* axes
    out overturned one such verdict by 2.5 orders of magnitude. This makes that
    bookkeeping one call: ``axes`` maps each free axis to its cardinality (e.g.
    ``{"orientation": 2, "period": 5, "square": 38_300_000}``), ``completed`` is how
    many cells were actually evaluated. Returns ``{declared, completed, fraction,
    axes, summary}``. Feed ``declared`` to :meth:`Finding.with_coverage` as
    ``intended`` so the coverage claim and the axis arithmetic cannot drift apart.
    """
    if not axes:
        raise ValueError("declare at least one axis")
    for name, size in axes.items():
        if size < 1:
            raise ValueError(f"axis {name!r} must have cardinality >= 1, got {size}")
    declared = math.prod(axes.values())
    if completed < 0 or completed > declared:
        raise ValueError(f"completed must be within [0, {declared}], got {completed}")
    fraction = completed / declared
    dims = " x ".join(f"{name}={size:,}" for name, size in axes.items())
    return {
        "declared": declared,
        "completed": completed,
        "fraction": fraction,
        "axes": dict(axes),
        "summary": f"{dims} = {declared:,} cells; {completed:,} done ({100 * fraction:.2g}%)",
    }


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

    @classmethod
    def of_axes(cls, evaluated: int, *, timeouts: int = 0, capped: int = 0,
                exhaustive: bool = False, **axes: int) -> Coverage:
        """Coverage whose ``intended`` is the product of the declared free axes.

        ``Coverage.of_axes(500_000, orientation=2, square=38_300_000)`` cannot
        understate the space the way a hand-typed ``intended`` can — see
        :func:`searched_fraction`.
        """
        sf = searched_fraction(axes, evaluated)
        return cls(evaluated, sf["declared"], timeouts, capped, exhaustive)


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
    power_z: float | None = None
    not_closed: list[str] = field(default_factory=list)
    reopen_delta: str | None = None
    void_reason: str | None = None

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

    def with_power(self, z: float) -> Finding:
        """Attach the measured power of the statistic on same-shape plants.

        ``z`` is how many null standard deviations the statistic sits above its null
        on synthetics *carrying the signal under test* (:func:`buttcrack.power.separation`).
        A negative with power below :data:`MIN_POWER_Z` is reported as ``silent`` —
        the statistic could not have seen the signal, so its quiet means nothing.
        Without this attestation a negative keeps its legacy verdicts, but stating a
        measured power is what upgrades "we saw nothing" to "there is nothing".
        """
        self.power_z = float(z)
        return self

    def scoped(self, not_closed: list[str], reopen_delta: str | None = None) -> Finding:
        """Declare what this negative deliberately does NOT close, and what reopens it.

        A narrow null cited as a family kill is how live hypotheses die in the record.
        ``not_closed`` names the adjacent cells left open (e.g. ``["non-standard ring",
        "terse register"]``); ``reopen_delta`` states the observation that would justify
        reopening. A closed finding with a non-empty ``not_closed`` renders its verdict
        as ``closed (scoped)`` so it can never be quoted as broader than it is.
        """
        self.not_closed = list(not_closed)
        self.reopen_delta = reopen_delta
        return self

    def voided(self, reason: str) -> Finding:
        """Mark the finding VOID: the instrument itself was broken, so the result is
        meaningless — which is a different thing from a trusted negative.

        The distinction exists because a broken harness produces output
        indistinguishable from a clean null (a solver handed a file *path* instead of
        ciphertext once scored 47k cells of heap garbage and passed its own screen).
        A void finding renders — stating WHY it is void — but can never be cited as
        evidence for or against anything.
        """
        if not reason.strip():
            raise ValueError("state what broke the instrument")
        self.void_reason = reason
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
        """One of: 'void', 'positive', 'silent', 'closed', 'closed (scoped)',
        'inconclusive', 'null (partial)'.

        ``void`` — the instrument was broken; the result means nothing either way.
        ``silent`` — a negative whose stated power is below :data:`MIN_POWER_Z`; the
        statistic could not have seen the signal, so the null is not evidence.
        ``closed (scoped)`` — a complete negative that has declared adjacent cells it
        does NOT close (:meth:`scoped`).
        """
        if self.void_reason is not None:
            return "void"
        gaps = self.missing()
        if gaps:
            raise Unverified(
                f"cannot state {self.claim!r}:\n  - " + "\n  - ".join(gaps))
        assert self.coverage is not None
        cp = self.corrected_p
        if cp is not None and cp < 0.05:
            return "positive"
        if self.power_z is not None and self.power_z < MIN_POWER_Z:
            return "silent"
        if self.coverage.complete or self.coverage.exhaustive:
            return "closed (scoped)" if self.not_closed else "closed"
        if self.coverage.timeouts or self.coverage.capped:
            return "inconclusive"
        return "null (partial)"

    def render(self) -> str:
        v = self.verdict()          # raises Unverified if unearned
        lines = [f"{self.claim}", f"  verdict     : {v}"]
        if v == "void":
            lines.append(f"  void        : {self.void_reason}")
            lines.append("  (instrument broken — citable as neither positive nor negative)")
            return "\n".join(lines)
        assert self.plant is not None and self.coverage is not None
        if self.observed is not None:
            lines.append(f"  observed    : {self.observed:.4f}")
        if self.p_value is not None:
            lines.append(f"  p           : {self.p_value:.4g} raw, "
                         f"{self.corrected_p:.4g} corrected over "
                         f"{self.family_size} hypotheses")
        if self.power_z is not None:
            lines.append(f"  power       : z={self.power_z:+.2f} on same-shape plants"
                         + ("" if self.power_z >= MIN_POWER_Z
                            else f" (below {MIN_POWER_Z:.0f} — null is SILENT)"))
        lines.append(f"  {self.plant.summary()}")
        lines.append(f"  null        : {self.null_description}")
        lines.append(f"  {self.coverage.summary()}")
        for cell in self.not_closed:
            lines.append(f"  not closed  : {cell}")
        if self.reopen_delta:
            lines.append(f"  reopen if   : {self.reopen_delta}")
        for n in self.notes:
            lines.append(f"  note        : {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        v = self.verdict()
        if v == "void":
            return {"claim": self.claim, "verdict": v, "void_reason": self.void_reason}
        assert self.plant is not None and self.coverage is not None
        return {
            "claim": self.claim,
            "verdict": v,
            "power_z": self.power_z,
            "not_closed": list(self.not_closed),
            "reopen_delta": self.reopen_delta,
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
