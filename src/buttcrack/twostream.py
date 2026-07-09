"""Blind two-stream additive split (Reddy-Knight running-key separation).

Some constructions add *two natural-language streams together* rather than a
message and a random key: ``c_i = (x_i + y_i) mod 26`` where BOTH ``x`` (the
plaintext) and ``y`` (a running key that is itself English -- a book passage, a
prior solution) are ordinary English. Because neither stream is random, the sum
is not a one-time pad: the pair is jointly far more English than any random
decomposition, and can be recovered *blind*, without knowing either stream.

The recovery is a left-to-right **beam search** (this module's
:func:`_beam_decode`). At each ciphertext position it enumerates the 26 possible
values of ``x_i`` -- each of which pins ``y_i = (c_i - x_i) mod 26`` -- and
scores every partial pair by a character language-model next-char log-prob on
*both* streams simultaneously, keeping the top ``beam`` partial hypotheses. This
is the classic Reddy-Knight two-stream / running-key attack.

The language model is an interpolated, additively-smoothed character n-gram
model (:class:`_CharLM`) that reuses the interpolation math of
:class:`buttcrack.nonprose.GenreModel`. By default it is an order-5 model built
from buttcrack's bundled English n-gram count tables (mono- through quintgrams --
rich corpus statistics); if those data files are missing it falls back to an
order-3 model trained on a short embedded English paragraph. No puzzle material
is used anywhere.

Caveats to read honestly:

* **This method wants length.** Separation is reliable at roughly >= 150
  characters; on short spans the beam is under-constrained and the recovered
  streams degrade. The power test in :func:`metric` (a true EN+EN sum scoring
  far above the same statistic on shuffled ciphertext) still discriminates well
  before the *exact* streams are fully recoverable.
* **The split is swap-symmetric and approximate.** ``x + y == y + x``, so which
  recovered stream corresponds to which original is arbitrary, and a genuine
  decomposition is recovered up to a modest per-character error rate rather than
  exactly. Verify a hit by eye against long contiguous spans, not char-for-char.

Public API:

* :func:`split` -- recover the two maximally-English streams from a sum.
* :func:`metric` -- the best two-stream score, for a shuffle-null power test.
* :func:`encode` -- add two English passages, for building test plants.
"""

from __future__ import annotations

import functools
from importlib import resources

from .keysources import _alphabet
from .nonprose import SMOOTHING, GenreModel
from .text import only_letters

#: Default beam width. Wider is slower but less likely to prune the true pair.
DEFAULT_BEAM = 400

#: Highest n-gram order of the default model. Two-stream separation needs a long
#: local context to break the many rival English decompositions apart; the
#: bundled tables go up to hexagrams, and order 5 matches the reference attack.
DEFAULT_ORDER = 5

#: Names of the bundled ``english_<name>.txt`` count tables by n-gram order.
_TABLE_NAMES = {
    1: "monograms",
    2: "bigrams",
    3: "trigrams",
    4: "quadgrams",
    5: "quintgrams",
    6: "hexagrams",
}


def _lambdas_for_order(order: int) -> dict[int, float]:
    """Interpolation weights per order: geometric 0.5 decay, highest order heaviest."""
    raw = {order - i: 0.5**i for i in range(order)}
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}


#: A short original English paragraph (NOT puzzle text) used only when the bundled
#: n-gram count tables are unavailable, so the module still works self-contained.
_FALLBACK_ENGLISH = (
    "When the two streams are added together nothing in the sum looks like either "
    "part, and yet both halves are ordinary sentences written in plain english. A "
    "reader who knows only the total can still pull the pieces apart, because real "
    "language leaves a heavy trace that random noise never does. Every letter that "
    "follows another follows it for a reason, and those reasons pile up until the "
    "true reading stands far above every rival guess. The search moves from left to "
    "right, keeping the best partial answers alive and letting the weak ones fall "
    "away, so that by the final letter only the sentences that make sense remain."
)


def _read_ngram_table(name: str) -> dict[str, int]:
    """Load a bundled ``english_<name>.txt`` count table as ``{gram: count}``."""
    fname = f"english_{name}.txt"
    raw = resources.files("buttcrack.data").joinpath(fname).read_text(encoding="ascii")
    table: dict[str, int] = {}
    for line in raw.splitlines():
        if not line:
            continue
        gram, _, count = line.partition(" ")
        table[gram] = int(count)
    return table


def _build_gm_from_tables(order: int = DEFAULT_ORDER) -> GenreModel:
    """Construct a :class:`GenreModel` from buttcrack's bundled n-gram count tables.

    Reuses ``GenreModel``'s interpolation/smoothing math but seeds it with real
    corpus counts (orders 1..``order``) instead of training on a short paragraph.
    Context counts are the within-order marginals of the loaded (possibly pruned)
    grams; interpolation down to lower orders covers any gram missing from a
    truncated high-order table.
    """
    gram_counts: dict[int, dict[str, int]] = {}
    ctx_counts: dict[int, dict[str, int]] = {}
    for m in range(1, order + 1):
        table = _read_ngram_table(_TABLE_NAMES[m])
        ctxs: dict[str, int] = {}
        for gram, cnt in table.items():
            ctx = gram[:-1]
            ctxs[ctx] = ctxs.get(ctx, 0) + cnt
        gram_counts[m] = table
        ctx_counts[m] = ctxs
    return GenreModel(order, SMOOTHING, _lambdas_for_order(order), gram_counts, ctx_counts)


class _CharLM:
    """Interpolated character n-gram LM exposing incremental next-char scoring.

    Wraps a :class:`GenreModel` (whose ``_logprob`` already computes an
    interpolated, additively-smoothed next-char log-probability) and memoizes the
    step log-prob per ``(context, char)``, so the beam search's inner loop is an
    amortized O(1) dict lookup. For low orders (context length <= 2) the whole
    step table is precomputed up front; for higher orders -- where the full table
    (26**(order-1) contexts) is infeasible -- it is filled lazily on demand, which
    is cheap because the beam only ever visits a small slice of all contexts.
    :meth:`score` delegates to the model's mean-log-prob-per-character.
    """

    def __init__(self, model: GenreModel) -> None:
        self._model = model
        self.order = model.order
        self._ctxlen = max(0, model.order - 1)
        self._step: dict[tuple[str, str], float] = {}
        if self._ctxlen <= 2:
            self._precompute_step_cache()

    def _precompute_step_cache(self) -> None:
        """Precompute ``logp_step`` for all contexts of length ``0.._ctxlen``."""
        alpha = [chr(65 + i) for i in range(26)]
        contexts: list[str] = [""]
        level: list[str] = [""]
        for _ in range(self._ctxlen):
            level = [c + ch for c in level for ch in alpha]
            contexts.extend(level)
        for ctx in contexts:
            for ch in alpha:
                self._step[(ctx, ch)] = self._model._logprob(ctx, ch)

    def logp_step(self, context: str, ch: str) -> float:
        """Natural-log probability of ``ch`` following ``context`` (last ``order-1`` chars)."""
        ctx = context[-self._ctxlen :] if self._ctxlen else ""
        val = self._step.get((ctx, ch))
        if val is None:
            val = self._model._logprob(ctx, ch)
            self._step[(ctx, ch)] = val
        return val

    def score(self, text: str) -> float:
        """Mean natural-log probability per character (length-independent)."""
        return self._model.score(text)


@functools.cache
def _default_lm() -> _CharLM:
    """Return the cached default English :class:`_CharLM`.

    Built from the bundled n-gram count tables when available, else trained on
    :data:`_FALLBACK_ENGLISH`. Cached so every :func:`split` / :func:`metric`
    call (and every shuffle null) shares one build.
    """
    try:
        model = _build_gm_from_tables(order=DEFAULT_ORDER)
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        model = GenreModel.train(_FALLBACK_ENGLISH, order=3)
    return _CharLM(model)


def _beam_decode(cvals: list[int], lm: _CharLM, alphabet: str, beam: int) -> tuple[str, str]:
    """Left-to-right beam search for the two maximally-English additive streams.

    ``cvals`` are the ciphertext letters as indices into ``alphabet`` (the space
    in which addition is defined). At each position every surviving hypothesis is
    extended by all 26 choices of the stream-a index ``xa`` (which fixes the
    stream-b index ``(c - xa) mod 26``); the joint next-char LM score on both
    streams ranks the extensions and the top ``beam`` are kept. Returns the two
    recovered A-Z streams (as real letters, ``alphabet[index]``).
    """
    n = len(cvals)
    ctxlen = lm._ctxlen
    # states[k] = (score, a_ctx, b_ctx); back[i][k] = (parent_index, a_char, b_char)
    states: list[tuple[float, str, str]] = [(0.0, "", "")]
    back: list[list[tuple[int, str, str]]] = []
    for i in range(n):
        cpos = cvals[i]
        cand: list[tuple[float, int, str, str]] = []
        for pidx, (sc, actx, bctx) in enumerate(states):
            for xa in range(26):
                bpos = (cpos - xa) % 26
                ach = alphabet[xa]
                bch = alphabet[bpos]
                s = sc + lm.logp_step(actx, ach) + lm.logp_step(bctx, bch)
                cand.append((s, pidx, ach, bch))
        if len(cand) > beam:
            cand.sort(key=lambda t: t[0], reverse=True)
            cand = cand[:beam]
        new_states: list[tuple[float, str, str]] = []
        layer: list[tuple[int, str, str]] = []
        for s, pidx, ach, bch in cand:
            _, pactx, pbctx = states[pidx]
            nactx = (pactx + ach)[-ctxlen:] if ctxlen else ""
            nbctx = (pbctx + bch)[-ctxlen:] if ctxlen else ""
            new_states.append((s, nactx, nbctx))
            layer.append((pidx, ach, bch))
        states = new_states
        back.append(layer)

    if not states:
        return "", ""
    best = max(range(len(states)), key=lambda k: states[k][0])
    a_chars: list[str] = []
    b_chars: list[str] = []
    j = best
    for i in range(n - 1, -1, -1):
        pidx, ach, bch = back[i][j]
        a_chars.append(ach)
        b_chars.append(bch)
        j = pidx
    a_chars.reverse()
    b_chars.reverse()
    return "".join(a_chars), "".join(b_chars)


def split(
    ciphertext: str,
    *,
    alphabet: str = "STANDARD",
    beam: int = DEFAULT_BEAM,
    lm: _CharLM | None = None,
) -> dict:
    """Blindly split an additive EN+EN sum into its two English streams.

    ``ciphertext`` is treated as ``c_i = (a_i + b_i) mod 26`` in the index space
    of ``alphabet`` (``"STANDARD"``/``"KRYPTOS"`` or a 26-letter permutation, via
    :func:`buttcrack.keysources._alphabet`). Returns
    ``{"stream_a", "stream_b", "score"}`` where ``score`` is the mean of the two
    recovered streams' per-character LM scores (higher/less-negative = more
    jointly English). The split is swap-symmetric and approximate; see the module
    docstring. Reliable at roughly >= 150 characters.
    """
    alpha = _alphabet(alphabet)
    pos = {ch: i for i, ch in enumerate(alpha)}
    ct = only_letters(ciphertext)
    if lm is None:
        lm = _default_lm()
    cvals = [pos[ch] for ch in ct]
    stream_a, stream_b = _beam_decode(cvals, lm, alpha, beam)
    if stream_a:
        score = (lm.score(stream_a) + lm.score(stream_b)) / 2.0
    else:
        score = float("-inf")
    return {"stream_a": stream_a, "stream_b": stream_b, "score": score}


def metric(
    ciphertext: str,
    *,
    alphabet: str = "STANDARD",
    lm: _CharLM | None = None,
) -> float:
    """Best two-stream English score of ``ciphertext``, for a shuffle-null power test.

    A genuine EN+EN additive sum decomposes into a jointly-English pair whose
    score sits far above the same statistic computed on a shuffled copy of the
    ciphertext (which is a random additive combination). Comparing
    ``metric(ct)`` to ``metric(shuffle(ct))`` over several shuffles thus gives a
    z-test that this ciphertext is a real two-stream sum.
    """
    return split(ciphertext, alphabet=alphabet, lm=lm)["score"]


def encode(stream_a: str, stream_b: str, *, alphabet: str = "STANDARD") -> str:
    """Add two English passages character-wise: ``c_i = (a_i + b_i) mod 26``.

    Both inputs are reduced to A-Z letters; the sum is taken in the index space
    of ``alphabet`` and truncated to the shorter stream. Useful for building test
    "plants" (a known EN+EN sum) to validate :func:`split` and :func:`metric`.
    """
    alpha = _alphabet(alphabet)
    pos = {ch: i for i, ch in enumerate(alpha)}
    a = only_letters(stream_a)
    b = only_letters(stream_b)
    n = min(len(a), len(b))
    return "".join(alpha[(pos[a[i]] + pos[b[i]]) % 26] for i in range(n))
