"""Contamination sensitivity — how much embedded non-language can a statistic absorb
before its exclusion dissolves?

Exclusions computed on a "pure payload" model quietly assume the payload IS pure.
If the true plaintext embeds a non-language block — an inserted key, a serial, a
grid column of coordinates — every statistic moves, and a language exclusion that
held at insert size 0 can be void at insert size 17. The honest report is the
sensitivity curve: embed k characters of non-language in each plausible *shape*,
re-measure the statistic, and state the k at which your gate breaks.

:func:`embed` plants an insert of a given shape (the shapes correspond to real
write-in mechanisms):

* ``contiguous`` — one solid block (a key or header pasted into the stream);
* ``scattered`` — k characters at independent positions (interrupters, nulls);
* ``column`` — one character every ``width`` positions (an extra column in a
  width-``width`` grid write-in — the shape a columnar route embeds);
* ``coset`` — every ``period``-th position from a random phase (an insert riding
  one residue class of a periodic construction).

:func:`sensitivity_sweep` measures a statistic across insert sizes and shapes and
reports, per shape, the smallest k that moves the statistic past your tolerance —
the insert budget your exclusion actually has.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from .text import only_letters

MODES = ("contiguous", "scattered", "column", "coset")

Statistic = Callable[[str], float]


def _filler(k: int, rng: random.Random, filler: str | None) -> str:
    if filler is not None:
        letters = only_letters(filler)
        if len(letters) < k:
            raise ValueError(f"filler has {len(letters)} letters, need {k}")
        return letters[:k]
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(k))


def embed(
    text: str,
    k: int,
    mode: str = "contiguous",
    *,
    rng: random.Random | None = None,
    width: int | None = None,
    period: int | None = None,
    filler: str | None = None,
) -> tuple[str, list[int]]:
    """Insert ``k`` non-language characters into ``text`` in the given shape.

    Returns ``(contaminated, positions)`` where ``positions`` are the insert's
    indices in the *contaminated* string (length ``len(text) + k``). ``filler``
    supplies the insert characters (e.g. an actual key string); default is uniform
    random letters. ``column`` requires ``width``; ``coset`` requires ``period``
    (they are the same shape — one stride, two vocabularies).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if k < 0:
        raise ValueError("k must be >= 0")
    letters = only_letters(text)
    rng = rng or random.Random()
    ins = _filler(k, rng, filler)
    if k == 0:
        return letters, []

    if mode == "contiguous":
        j = rng.randrange(len(letters) + 1)
        positions = list(range(j, j + k))
        return letters[:j] + ins + letters[j:], positions

    if mode == "scattered":
        n_out = len(letters) + k
        positions = sorted(rng.sample(range(n_out), k))
        out: list[str] = []
        src = iter(letters)
        pos_set = dict(zip(positions, ins, strict=True))
        for i in range(n_out):
            out.append(pos_set[i] if i in pos_set else next(src))
        return "".join(out), positions

    stride = width if mode == "column" else period
    if not stride or stride < 2:
        raise ValueError(f"mode {mode!r} requires {'width' if mode == 'column' else 'period'} >= 2")
    phase = rng.randrange(stride)
    out = []
    positions = []
    src = iter(letters)
    ins_iter = iter(ins)
    remaining = k
    i = 0
    consumed = 0
    while consumed < len(letters) or remaining > 0:
        if remaining > 0 and i % stride == phase:
            out.append(next(ins_iter))
            positions.append(i)
            remaining -= 1
        elif consumed < len(letters):
            out.append(next(src))
            consumed += 1
        else:  # letters exhausted; place any remaining insert chars at the tail
            out.append(next(ins_iter))
            positions.append(i)
            remaining -= 1
        i += 1
    return "".join(out), positions


def sensitivity_sweep(
    text: str,
    statistic: Statistic,
    ks: Sequence[int],
    *,
    modes: Sequence[str] = MODES,
    trials: int = 20,
    tolerance: float | None = None,
    width: int | None = None,
    period: int | None = None,
    filler: str | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Measure how far each insert shape/size moves ``statistic`` off its clean value.

    For every ``(mode, k)`` pair, ``trials`` random embeddings are measured.
    Records report the mean shift from the clean value and the shift in units of
    the trial spread ("z"). With ``tolerance`` given (in the statistic's own
    units), ``budget[mode]`` is the smallest k whose |mean shift| exceeds it —
    i.e. the insert size at which any exclusion built on this statistic at this
    tolerance DISSOLVES. Quote the budget next to the exclusion: "payload-IoC
    excludes every language (p=0.0000) *for inserts up to k=12 in any shape*"
    is a different — and honest — claim from the unconditional one.
    """
    letters = only_letters(text)
    rng = rng or random.Random(20250615)
    clean = float(statistic(letters))
    records: list[dict] = []
    budget: dict[str, int | None] = {}
    for mode in modes:
        budget[mode] = None
        for k in sorted(ks):
            if k == 0:
                continue
            vals = []
            for _ in range(trials):
                contaminated, _ = embed(
                    letters, k, mode, rng=rng, width=width, period=period, filler=filler
                )
                vals.append(float(statistic(contaminated)))
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            shift = mean - clean
            records.append(
                {
                    "mode": mode,
                    "k": k,
                    "mean": round(mean, 6),
                    "sd": round(sd, 6),
                    "shift": round(shift, 6),
                    "z": round(shift / sd, 2) if sd > 0 else 0.0,
                }
            )
            if tolerance is not None and budget[mode] is None and abs(shift) > tolerance:
                budget[mode] = k
    return {"clean": round(clean, 6), "records": records, "budget": budget}
