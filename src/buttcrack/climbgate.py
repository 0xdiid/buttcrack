"""Solver-as-detector — read what a cracker's *achievable score band* says about a
target, and refuse to read anything from a solver that cannot crack its own plant.

Fingerprint statistics rank cipher families; they do not identify them (a wrong
family can match eight statistics at mean |z| < 0.5 and still be excluded by
decode with 100% power). The stronger instrument is the solver itself, used as a
detector: run each family's cracker at the target's exact length on

* **GATE** — a genuine plant of that family (English encoded with a real key): the
  score band the solver reaches when the answer IS there;
* **CTRL** — random letters: the band its search reaches by selection bias alone
  (structureless text scores surprisingly well — never compare a target to an
  absolute bar);
* **TARGET** — the real ciphertext.

A target scoring in the GATE band is evidence *for* the family; in the CTRL band,
evidence against. And a family whose GATE band never separates from its CTRL band
at this length is **UNGATED** — the solver cannot find its own planted answer, so
it says nothing about the target either way and its row is suppressed rather than
believed (the plant-gate discipline, applied to detection).

:func:`length_threshold` separates "this cipher is unclimbable" from "this text is
too short": sweep n with the same gate and find where recovery turns on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import registry
from .power import separation
from .scoring import NgramScorer, get_scorer
from .text import only_letters
from .validate import _FILLER

#: GATE/CTRL bands must separate at least this cleanly (z of gate above ctrl) for
#: the solver to count as a detector at this length.
MIN_GATE_Z = 2.0


def _plant_text(n: int, offset: int) -> str:
    base = _FILLER
    while len(base) < n + offset:
        base += _FILLER
    return base[offset : offset + n]


def _random_text(n: int, rng: random.Random, alphabet: str | None) -> str:
    alpha = alphabet or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(alpha) for _ in range(n))


def _best_score(cipher, text: str, scorer: NgramScorer, timeout: float | None) -> float | None:
    """Per-window average score of the cracker's best candidate (None = no output)."""
    try:
        cands = cipher.crack(text, scorer, top=1, timeout=timeout)
    except Exception:
        return None
    if not cands:
        return None
    pt = only_letters(cands[0].plaintext)
    if not pt:
        return None
    return scorer.average(pt)


@dataclass
class GateBand:
    """A cipher's achievable-score bands at one length: its own plant vs noise."""

    cipher: str
    n: int
    gate_scores: list[float]
    ctrl_scores: list[float]
    gate_z: float
    gated: bool

    def summary(self) -> str:
        state = "gated" if self.gated else "UNGATED (says nothing about any target)"
        gm = sum(self.gate_scores) / len(self.gate_scores) if self.gate_scores else float("nan")
        cm = sum(self.ctrl_scores) / len(self.ctrl_scores) if self.ctrl_scores else float("nan")
        return f"{self.cipher} @ n={self.n}: gate {gm:.3f} vs ctrl {cm:.3f} (z={self.gate_z:+.1f}) — {state}"


def solver_band(
    name: str,
    n: int,
    *,
    trials: int = 3,
    scorer: NgramScorer | None = None,
    timeout: float | None = 10.0,
    rng: random.Random | None = None,
) -> GateBand:
    """Measure one cipher's GATE and CTRL score bands at length ``n``.

    GATE plants are English (distinct slices of the reference filler) encoded with
    the cipher's documented example key; CTRL inputs are random letters over the
    cipher's ciphertext alphabet. ``gated`` requires the gate band to sit
    :data:`MIN_GATE_Z` null-sd above the ctrl band — the "can this solver find its
    own answer at this length" bar.
    """
    rng = rng or random.Random(20250615)
    scorer = scorer or get_scorer()
    cipher = registry.get(name)
    gate: list[float] = []
    ctrl: list[float] = []
    for t in range(trials):
        pt = _plant_text(n, offset=17 * t)
        try:
            ct = cipher.encode(pt, cipher.key_example) if cipher.needs_key else cipher.encode(pt, "")
        except Exception:
            ct = None
        if ct:
            s = _best_score(cipher, ct, scorer, timeout)
            if s is not None:
                gate.append(s)
        s = _best_score(cipher, _random_text(n, rng, cipher.ciphertext_alphabet), scorer, timeout)
        if s is not None:
            ctrl.append(s)
    if gate and ctrl:
        gate_z = separation(gate, ctrl).z
    else:
        gate_z = 0.0
    return GateBand(
        cipher=cipher.name,
        n=n,
        gate_scores=gate,
        ctrl_scores=ctrl,
        gate_z=gate_z,
        gated=bool(gate and ctrl) and gate_z >= MIN_GATE_Z,
    )


@dataclass
class DetectorRow:
    """One cipher's read of the target: where does it sit between CTRL and GATE?"""

    cipher: str
    band: GateBand
    target_score: float | None
    position: float | None  # 0.0 = ctrl band, 1.0 = gate band (anchored fraction)
    verdict: str  # "in-gate-band" | "above-ctrl" | "noise" | "ungated" | "no-output"

    def summary(self) -> str:
        pos = "n/a" if self.position is None else f"{self.position:+.2f}"
        return f"{self.cipher}: target position {pos} -> {self.verdict}"


def detector_sweep(
    target: str,
    ciphers: list[str] | None = None,
    *,
    trials: int = 3,
    scorer: NgramScorer | None = None,
    timeout: float | None = 10.0,
    rng: random.Random | None = None,
) -> list[DetectorRow]:
    """Run the solver-as-detector over ``ciphers`` (default: every registered cipher).

    Each row anchors the target's achieved score between that cipher's CTRL (0.0)
    and GATE (1.0) bands. Verdicts: ``in-gate-band`` (position ≥ 0.75 — the solver
    reads the target like its own plant: strong family evidence), ``above-ctrl``
    (0.35–0.75 — more structure than noise, not a clean read), ``noise``
    (below the ctrl band's reach), ``ungated`` (row suppressed: the solver failed
    its own plant at this length, so its silence is not evidence), ``no-output``.
    Rows are sorted gated-first by position. The verdict is per-cipher evidence,
    not a ranking to trust blindly — confirm any hit by decoding.
    """
    letters = only_letters(target)
    n = len(letters)
    rows: list[DetectorRow] = []
    scorer = scorer or get_scorer()
    for name in ciphers or registry.names():
        band = solver_band(name, n, trials=trials, scorer=scorer, timeout=timeout, rng=rng)
        cipher = registry.get(name)
        if cipher.ciphertext_alphabet and any(
            c not in cipher.ciphertext_alphabet for c in letters
        ):
            rows.append(DetectorRow(cipher.name, band, None, None, "noise"))
            continue
        if not band.gated:
            rows.append(DetectorRow(cipher.name, band, None, None, "ungated"))
            continue
        score = _best_score(cipher, letters, scorer, timeout)
        if score is None:
            rows.append(DetectorRow(cipher.name, band, None, None, "no-output"))
            continue
        gate_mean = sum(band.gate_scores) / len(band.gate_scores)
        ctrl_mean = sum(band.ctrl_scores) / len(band.ctrl_scores)
        span = gate_mean - ctrl_mean
        position = (score - ctrl_mean) / span if span > 0 else None
        if position is None:
            verdict = "no-output"
        elif position >= 0.75:
            verdict = "in-gate-band"
        elif position >= 0.35:
            verdict = "above-ctrl"
        else:
            verdict = "noise"
        rows.append(DetectorRow(cipher.name, band, score, position, verdict))
    rows.sort(key=lambda r: (not r.band.gated, -(r.position if r.position is not None else -9.0)))
    return rows


def length_threshold(
    name: str,
    lengths: list[int],
    *,
    trials: int = 3,
    scorer: NgramScorer | None = None,
    timeout: float | None = 10.0,
    rng: random.Random | None = None,
) -> dict:
    """Where does this solver's gate turn ON as text length grows?

    Separates "the cipher is unclimbable" from "the text is too short": sweep
    ``lengths``, measure the gate at each, and report the smallest gated length.
    A family whose threshold sits far above the target's length cannot be excluded
    *by search* at that length — only by structural/algebraic arguments — and any
    blind negative there is SILENT.
    """
    curve = []
    threshold = None
    for n in sorted(lengths):
        band = solver_band(name, n, trials=trials, scorer=scorer, timeout=timeout, rng=rng)
        curve.append({"n": n, "gate_z": round(band.gate_z, 2), "gated": band.gated})
        if threshold is None and band.gated:
            threshold = n
    return {"cipher": name, "curve": curve, "threshold": threshold}
