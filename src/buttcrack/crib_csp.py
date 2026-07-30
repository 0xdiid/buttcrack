"""Known-plaintext (crib) recovery for grid ciphers by constraint propagation.

Classical-cipher tools — CryptoCrack, AZdecrypt, and this package's own solvers —
attack keys with hill-climbing against a fitness function. That works when the
plaintext is fluent English and fails structurally when it is not: a wrong key can
score *higher* than the true one, so the true key is verifiable but not searchable.
No amount of restarts fixes it, because the objective itself is wrong.

A crib changes the problem from optimisation to **constraint satisfaction**. Under
known plaintext, a Polybius-grid cipher becomes a system of equations over a
bijection (letter -> cell), and a candidate assignment is either arc-consistent or
it is not. There is no scorer, so:

  * the plaintext register is irrelevant — coded, foreign and terse payloads are
    handled exactly like prose;
  * short cribs cannot "flood" (that is a fitness pathology, not a CSP one);
  * longer cribs are strictly better, and several short cribs compose — the
    opposite of the fitness-based rule of thumb;
  * UNSAT is a *proof* that no key of this shape produces the crib, which is a far
    stronger negative than "the search found nothing".

Backed by OR-Tools CP-SAT, which propagates AllDifferent and Element natively.
Install with ``pip install buttcrack[csp]`` (or ``pip install ortools``).

Currently modelled: bifid with a 5x5 (25-letter) square and an optional
period-p additive over the ciphertext.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Crib", "BifidCribProblem", "solve_bifid_crib", "CribSolution"]

_CP_HINT = (
    "crib CSP solving needs OR-Tools. Install with `pip install buttcrack[csp]` "
    "or `pip install ortools`."
)


def _cp_model():
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - exercised only without ortools
        raise ImportError(_CP_HINT) from exc
    return cp_model


@dataclass(frozen=True)
class Crib:
    """A known plaintext fragment at a known ciphertext offset."""

    offset: int
    text: str

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("crib offset must be >= 0")
        if not self.text:
            raise ValueError("crib text must be non-empty")


@dataclass(frozen=True)
class CribSolution:
    """One consistent key. `square` is the 25-letter grid, row-major."""

    square: str
    strip: tuple[int, ...]
    plaintext: str


@dataclass
class BifidCribProblem:
    """Bifid-over-additive crib recovery.

    ciphertext : the ciphertext, in `alphabet`'s letters
    cribs      : known plaintext fragments
    alphabet   : 26-letter ring the additive works in (e.g. the KRYPTOS ring)
    grid       : the 25 letters that may appear in the square (alphabet minus one)
    period     : bifid seriation period, and the additive's period
    phase      : where block boundaries sit, i.e. positions == phase (mod period).
                 When period does not divide len(ciphertext) this is a free
                 convention and every value must be tried; it is NOT cosmetic.
    additive   : True if a period-`period` additive is layered over the bifid
    orientation: "vig" (ct = inter + strip) or "beau" (ct = strip - inter)
    """

    ciphertext: str
    cribs: list[Crib]
    alphabet: str
    grid: str
    period: int = 7
    phase: int = 0
    additive: bool = True
    orientation: str = "vig"
    _idx: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.alphabet) != 26:
            raise ValueError("alphabet must have 26 letters")
        if len(self.grid) != 25:
            raise ValueError("grid must have 25 letters")
        if set(self.grid) - set(self.alphabet):
            raise ValueError("grid letters must all be in alphabet")
        if self.orientation not in ("vig", "beau"):
            raise ValueError("orientation must be 'vig' or 'beau'")
        if not 0 <= self.phase < self.period:
            raise ValueError("phase must be in [0, period)")
        self._idx = {c: i for i, c in enumerate(self.alphabet)}

    # ---------------------------------------------------------------- blocks

    def blocks(self) -> list[tuple[int, int]]:
        """(start, length) of each seriation block, honouring `phase`."""
        n, p = len(self.ciphertext), self.period
        out = []
        if self.phase:
            out.append((0, min(self.phase, n)))
        b = self.phase
        while b < n:
            out.append((b, min(p, n - b)))
            b += p
        return [(s, ln) for s, ln in out if ln > 0]

    @staticmethod
    def _pairs(length: int) -> list[tuple[tuple[int, str], tuple[int, str]]]:
        """Which (letter, coordinate) pair feeds each output cell of a block.

        A block of L letters writes rows then columns, then re-pairs consecutively:
        seq = [r_0..r_{L-1}, c_0..c_{L-1}], and output k reads (seq[2k], seq[2k+1]).
        """
        seq = [(i, "r") for i in range(length)] + [(i, "c") for i in range(length)]
        return [(seq[2 * k], seq[2 * k + 1]) for k in range(length)]


def solve_bifid_crib(
    problem: BifidCribProblem,
    max_solutions: int = 1,
    max_seconds: float = 60.0,
    workers: int = 8,
) -> tuple[list[CribSolution], str]:
    """Recover (square, strip) consistent with the cribs.

    Returns (solutions, status) where status is one of CP-SAT's
    "OPTIMAL"/"FEASIBLE"/"INFEASIBLE"/"UNKNOWN". **INFEASIBLE is a proof** that no
    key of this shape yields the cribs; UNKNOWN means the time limit was hit and
    nothing is proved.
    """
    cp_model = _cp_model()
    m = cp_model.CpModel()
    P = problem
    G = P.grid
    gidx = {c: i for i, c in enumerate(G)}

    # cell[L] = where grid letter L sits; letter_at is its inverse.
    cell = [m.NewIntVar(0, 24, f"cell{i}") for i in range(25)]
    letter_at = [m.NewIntVar(0, 24, f"at{c}") for c in range(25)]
    m.AddAllDifferent(cell)
    m.AddInverse(cell, letter_at)

    row = [m.NewIntVar(0, 4, f"row{i}") for i in range(25)]
    col = [m.NewIntVar(0, 4, f"col{i}") for i in range(25)]
    for i in range(25):
        m.AddDivisionEquality(row[i], cell[i], 5)
        m.AddModuloEquality(col[i], cell[i], 5)

    strip = [m.NewIntVar(0, 25, f"s{k}") for k in range(P.period)]
    if not P.additive:
        for k in range(P.period):
            m.Add(strip[k] == 0)

    # NB: there is NO gauge symmetry to quotient out here. Shifting every strip value
    # Caesar-shifts the intermediate letters, which would have to be absorbed by
    # relabelling the square -- but the grid holds only 25 of the 26 ring letters, so a
    # shift would need to place the excluded letter. The exclusion breaks the symmetry
    # and different gauges are genuinely different keys. Pinning strip[0] loses solutions.

    blocks = P.blocks()
    for crib in P.cribs:
        for j, ch in enumerate(crib.text):
            pos = crib.offset + j
            if ch not in gidx:
                raise ValueError(f"crib letter {ch!r} cannot occur in the grid")
            if pos >= len(P.ciphertext):
                raise ValueError("crib runs past the end of the ciphertext")
        for b, ln in blocks:
            lo, hi = b, b + ln
            if not (crib.offset <= lo and hi <= crib.offset + len(crib.text)):
                continue  # only fully-covered blocks give equations
            pt = [crib.text[i - crib.offset] for i in range(lo, hi)]
            for k, ((ia, ca), (ib, cb)) in enumerate(P._pairs(ln)):
                va = (row if ca == "r" else col)[gidx[pt[ia]]]
                vb = (row if cb == "r" else col)[gidx[pt[ib]]]
                target = m.NewIntVar(0, 24, f"t{b}_{k}")
                m.Add(target == 5 * va + vb)
                out = m.NewIntVar(0, 24, f"o{b}_{k}")
                m.AddElement(target, letter_at, out)
                # out is a grid index; tie it to the ciphertext through the strip
                ct_i = P._idx[P.ciphertext[b + k]]
                ring = m.NewIntVar(0, 25, f"r{b}_{k}")
                # ring index of the intermediate letter
                m.AddElement(out, [P._idx[g] for g in G], ring)
                sk = strip[(b + k) % P.period]
                # ct = (inter + strip) mod 26  (vig) / (strip - inter) mod 26 (beau).
                # Encode the single wrap explicitly: the sum lies in [0, 51], so it is
                # either ct_i or ct_i + 26.
                tot = m.NewIntVar(0, 51, f"sum{b}_{k}")
                if P.orientation == "vig":
                    m.Add(tot == ring + sk)
                else:
                    m.Add(tot == sk - ring + 26)
                wrap = m.NewBoolVar(f"w{b}_{k}")
                m.Add(tot == ct_i).OnlyEnforceIf(wrap.Not())
                m.Add(tot == ct_i + 26).OnlyEnforceIf(wrap)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_seconds
    solver.parameters.num_search_workers = workers

    sols: list[CribSolution] = []

    class _Collect(cp_model.CpSolverSolutionCallback):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def on_solution_callback(self) -> None:
            sq = [""] * 25
            for i, c in enumerate(G):
                sq[self.Value(cell[i])] = c
            st = tuple(self.Value(s) for s in strip)
            sols.append(CribSolution("".join(sq), st, ""))
            self.n += 1
            if self.n >= max_solutions:
                self.StopSearch()

    status = solver.SearchForAllSolutions(m, _Collect()) if max_solutions > 1 else solver.Solve(m)
    if max_solutions == 1 and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sq = [""] * 25
        for i, c in enumerate(G):
            sq[solver.Value(cell[i])] = c
        sols.append(CribSolution("".join(sq), tuple(solver.Value(s) for s in strip), ""))
    return sols, solver.StatusName(status)
