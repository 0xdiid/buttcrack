"""Validate-on-synthetic harness — the discipline that makes negatives trustworthy.

A long cryptanalysis campaign repeatedly produced *confident-looking but
wrong* solver output, and just as often risked declaring a cipher unbreakable when the
real bug was in the attack.  The cure was a fixed ritual: before believing a negative on
the real ciphertext, **prove the attack recovers its own structure on a synthetic of the
same shape**.  An attack that cannot crack a same-grammar synthetic it was just handed has
a bug; only an attack that passes its positive control earns the right to report a
negative.

This module bottles that ritual:

* :func:`make_synthetic` builds a same-structure ciphertext from a family grammar
  (keyed alphabet, periodic Vigenere/Beaufort/Quagmire substitution, columnar
  transposition, and the two common layered compositions).
* :func:`positive_control` encodes a synthetic, runs an ``attack_fn`` against it, and
  reports whether the attack RECOVERED the structure (so a caller can gate trust).
* :func:`genuine_solve_signature` returns the calibrated readable-English bar
  (~``qscore_per_char`` and ``word_cov``) a real solve must clear at a given length, so a
  "solve" that sits below the bar is flagged as overfit.

The substitution is additive in *keyed-alphabet index space*, matching the shift
conventions in :mod:`buttcrack.layered` / :mod:`buttcrack.transsub`:

* Vigenere   enc ``c = p + k``   dec ``p = c - k``
* Beaufort   enc ``c = k - p``   dec ``p = k - c``   (reciprocal)
* Variant    enc ``c = p - k``   dec ``p = c + k``

with all indices taken in the keyed alphabet (``KRYPTOS...`` or standard A-Z).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .ciphers.columnar import _encode_letters, _read_order
from .ciphers.quagmire3 import keyed_alphabet
from .scoring import get_scorer
from .text import only_letters
from .words import long_word_coverage

#: Filler used to pad/trim a supplied plaintext to an exact synthetic length.  A repeated
#: English passage keeps the synthetic's monogram and n-gram statistics realistic, so an
#: attack's positive control exercises the same gradient it will face on the real text.
_FILLER = (
    "OVERTHEQUIETMORNINGTHELIBRARIANSORTEDEACHVOLUMEONTHESHELFANDNOTEDITSTITLEINHERLEDGER"
    "WHILESTUDENTSGATHEREDNEARTHEWINDOWSTOREADBENEATHTHEWARMLIGHTOFTHERISINGSUNOUTSIDEAS"
    "ABROADRIVERWOUNDPASTTHEOLDSTONEBRIDGEWHEREFARMERSCARRIEDBASKETSOFFRESHFRUITTOTOWNAND"
    "CHILDRENPLAYEDALONGTHEGRASSYBANKSLAUGHINGASTHEYCHASEDONEANOTHERTHROUGHTHEOPENFIELDS"
    "UNTILTHEDISTANTBELLOFTHECHURCHCALLEDTHEMHOMEFORTHEEVENINGMEALANDALONGNIGHTOFQUIETREST"
)

#: Supported substitution families (additive in keyed-alphabet index space).
SUBSTITUTIONS = ("vigenere", "beaufort", "variant", "quagmire3")

#: Supported structure kinds for :func:`make_synthetic`.
STRUCTURES = (
    "substitution",
    "columnar",
    "double-columnar",
    "substitution-over-columnar",
    "columnar-over-substitution",
)


def _alphabet(name: str) -> str:
    """Resolve an alphabet spec to its 26-letter keyed alphabet.

    ``"STD"``/``"STANDARD"`` -> plain A-Z; anything else is treated as a keyword and
    keyed (``"KRYPTOS"`` -> ``KRYPTOSABCDEFGHIJLMNQUVWXZ``).
    """
    upper = name.upper()
    if upper in ("STD", "STANDARD", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return keyed_alphabet(upper)


def _key_shifts(key: str, alphabet: str) -> list[int]:
    """Per-column additive shifts from a substitution key, in ``alphabet`` index space."""
    idx = {c: i for i, c in enumerate(alphabet)}
    letters = only_letters(key)
    if not letters:
        raise ValueError("substitution key must contain letters")
    return [idx[c] for c in letters]


def encode_substitution(
    plaintext: str, key: str, *, substitution: str = "vigenere", alphabet: str = "KRYPTOS"
) -> str:
    """Encode a periodic substitution over a keyed alphabet (family convention).

    ``substitution`` is one of :data:`SUBSTITUTIONS`; ``alphabet`` is ``"KRYPTOS"`` (or any
    keyword) or ``"STD"``.  The period is ``len(only_letters(key))``.
    """
    if substitution not in SUBSTITUTIONS:
        raise ValueError(f"unknown substitution {substitution!r}; expected one of {SUBSTITUTIONS}")
    alpha = _alphabet(alphabet)
    idx = {c: i for i, c in enumerate(alpha)}
    shifts = _key_shifts(key, alpha)
    period = len(shifts)
    pt = only_letters(plaintext)
    out = []
    for i, ch in enumerate(pt):
        p = idx[ch]
        k = shifts[i % period]
        if substitution == "beaufort":
            c = (k - p) % 26
        elif substitution == "variant":
            c = (p - k) % 26
        else:  # vigenere / quagmire3 (quagmire3 == vigenere over the keyed alphabet)
            c = (p + k) % 26
        out.append(alpha[c])
    return "".join(out)


def decode_substitution(
    ciphertext: str, key: str, *, substitution: str = "vigenere", alphabet: str = "KRYPTOS"
) -> str:
    """Inverse of :func:`encode_substitution`."""
    alpha = _alphabet(alphabet)
    idx = {c: i for i, c in enumerate(alpha)}
    shifts = _key_shifts(key, alpha)
    period = len(shifts)
    ct = only_letters(ciphertext)
    out = []
    for i, ch in enumerate(ct):
        c = idx[ch]
        k = shifts[i % period]
        if substitution == "beaufort":
            p = (k - c) % 26
        elif substitution == "variant":
            p = (c + k) % 26
        else:
            p = (c - k) % 26
        out.append(alpha[p])
    return "".join(out)


def encode_columnar(plaintext: str, keyword: str) -> str:
    """Complete/incomplete columnar with read-order = rankorder(keyword) in standard A-Z.

    Delegates to :mod:`buttcrack.ciphers.columnar` so the synthetic uses the exact same
    transposition the solvers invert (ties broken left-to-right).
    """
    return _encode_letters(only_letters(plaintext), _read_order(keyword))


def _fit_length(plaintext: str | None, length: int | None) -> str:
    """Return a clean letter string of exactly ``length`` (default: the filler length)."""
    base = only_letters(plaintext) if plaintext else _FILLER
    if not base:
        raise ValueError("plaintext reduced to no letters")
    if length is None:
        return base
    if length <= 0:
        raise ValueError("length must be positive")
    while len(base) < length:
        base += _FILLER
    return base[:length]


def make_synthetic(
    structure_spec: dict[str, Any],
    plaintext: str | None = None,
    *,
    key: dict[str, Any] | None = None,
    length: int | None = None,
) -> dict[str, Any]:
    """Build a same-structure ciphertext from a puzzle-family grammar.

    ``structure_spec`` selects the construction; ``key`` carries its parameters.  The two
    may be merged (any key field is read from ``structure_spec`` if absent in ``key``), so
    a single dict describing a confirmed recipe round-trips straight through.

    Supported ``structure`` values (:data:`STRUCTURES`):

    * ``"substitution"`` — periodic Vig/Beaufort/Variant/Quagmire over a keyed alphabet.
      keys: ``substitution``, ``alphabet``, ``sub_key``.
    * ``"columnar"`` — single columnar.  keys: ``columnar_keyword`` (or ``columnar_order``).
    * ``"double-columnar"`` — two columnars.  keys:
      ``columnar_keywords`` (list of two).
    * ``"substitution-over-columnar"`` — substitution is the OUTER layer
      (``CT = Sub(Columnar(PT))``).
    * ``"columnar-over-substitution"`` — columnar is the OUTER layer
      (``CT = Columnar(Sub(PT))``); the transposition-over-substitution shape.

    Returns ``{ciphertext, plaintext, structure, key, length}`` — everything a
    :func:`positive_control` needs to encode and then verify a recovery.
    """
    spec = dict(structure_spec)
    params: dict[str, Any] = {**spec, **(key or {})}
    structure = spec.get("structure")
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure {structure!r}; expected one of {STRUCTURES}")

    pt = _fit_length(plaintext, length)
    alphabet = params.get("alphabet", "KRYPTOS")
    substitution = params.get("substitution", "vigenere")
    sub_key = params.get("sub_key")

    def _sub_enc(text: str) -> str:
        if not sub_key:
            raise ValueError(f"structure {structure!r} requires a 'sub_key'")
        return encode_substitution(text, sub_key, substitution=substitution, alphabet=alphabet)

    def _col_enc(text: str, keyword: str) -> str:
        return encode_columnar(text, keyword)

    if structure == "substitution":
        ciphertext = _sub_enc(pt)
    elif structure == "columnar":
        kw = params.get("columnar_keyword")
        if not kw:
            raise ValueError("structure 'columnar' requires a 'columnar_keyword'")
        ciphertext = _col_enc(pt, kw)
    elif structure == "double-columnar":
        kws = params.get("columnar_keywords")
        if not kws or len(kws) != 2:
            raise ValueError("structure 'double-columnar' requires two 'columnar_keywords'")
        ciphertext = _col_enc(_col_enc(pt, kws[0]), kws[1])
    elif structure == "substitution-over-columnar":
        kw = params.get("columnar_keyword")
        if not kw:
            raise ValueError("structure requires a 'columnar_keyword'")
        ciphertext = _sub_enc(_col_enc(pt, kw))
    else:  # columnar-over-substitution
        kw = params.get("columnar_keyword")
        if not kw:
            raise ValueError("structure requires a 'columnar_keyword'")
        ciphertext = _col_enc(_sub_enc(pt), kw)

    return {
        "ciphertext": ciphertext,
        "plaintext": pt,
        "structure": structure,
        "key": params,
        "length": len(pt),
    }


def genuine_solve_signature(length: int) -> dict[str, float]:
    """The calibrated readable-English bar a genuine solve must clear at ``length``.

    Returns ``{qscore_per_char, word_cov}``.  Empirically (campaign data at a canonical
    272-letter length) clean English plaintext scores ``qscore_per_char`` near
    ``-4.2`` and ``long_word_coverage`` near ``0.69``; quadgram "salad" that fools an
    n-gram model but is not language sits well below both (``qscore_per_char`` ~ ``-6+`` and
    ``word_cov`` < ``0.2``).  Thresholds relax slightly for very short text, where both
    statistics are noisier, but never demand more than the 272-letter bar.

    Use these as the floor for declaring a candidate a *real* solve (and conversely, the
    ceiling a negative must stay under to be trusted).
    """
    # The 272-char anchor, with a small noise allowance that shrinks as length grows.
    base_q = -4.2
    base_cov = 0.69
    if length >= 240:
        return {"qscore_per_char": base_q, "word_cov": base_cov}
    # Shorter text: loosen by a fixed margin (fewer windows => noisier per-char score,
    # fewer long words tileable) but stay anchored to the same shape.
    slack = min(0.8, (240 - length) / 240.0)
    return {
        "qscore_per_char": round(base_q - 1.4 * slack, 4),
        "word_cov": round(max(0.35, base_cov - 0.5 * slack), 4),
    }


def _qscore_per_char(text: str) -> float:
    """Mean quadgram log-probability per character (length-independent readability)."""
    letters = only_letters(text)
    if not letters:
        return float("-inf")
    scorer = get_scorer()
    return scorer.score(letters) / len(letters)


def solve_confidence(text: str, length: int | None = None) -> dict:
    """Calibrated readability verdict for a candidate plaintext.

    A blind solver's raw n-gram fitness does not say whether a decode is *English* —
    n-gram "salad" can score high. This compares the candidate against
    :func:`genuine_solve_signature` on two language-grounded statistics and returns::

        {"word_coverage", "qscore_per_char", "recovered"}

    where ``recovered`` is ``True`` only when the text clears *both* the long-word
    coverage and quadgram-per-char bars for its length. Callers should surface
    ``recovered`` so a 3%-correct decode is never presented like a real solve.
    """
    letters = only_letters(text)
    n = length if length is not None else len(letters)
    sig = genuine_solve_signature(n if n else 1)
    wc = long_word_coverage(letters) if letters else 0.0
    q = _qscore_per_char(letters) if letters else float("-inf")
    # "recovered" is a *clearly-English vs n-gram-salad* gate, deliberately looser than the
    # strict canonical-solve bar in genuine_solve_signature (which is tuned to the specific
    # 272-char Kryptos plaintext and would false-negative legitimate English with shorter
    # words). Salad sits near word_cov 0 / qscore -6+; readable English clears both floors.
    # The 0.40 long-word-coverage floor matches the established "recovered" bar used by
    # transsub and runkey, so the flag means the same thing across the toolkit; the qscore
    # floor is a loose secondary guard against high-coverage n-gram salad.
    wc_floor = 0.40
    q_floor = sig["qscore_per_char"] - 0.3
    return {
        "word_coverage": round(wc, 4),
        "qscore_per_char": round(q, 4) if q != float("-inf") else q,
        "recovered": bool(wc >= wc_floor and q >= q_floor),
    }


def _extract_plaintext(result: Any) -> str:
    """Pull a plaintext string out of whatever an ``attack_fn`` returned.

    Accepts a bare string, a dict with a ``"plaintext"`` key, an object with a
    ``.plaintext`` attribute, or a list of such (first element).  Anything else yields the
    empty string (a clean "did not recover").
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("plaintext", "") or "")
    if isinstance(result, (list, tuple)):
        return _extract_plaintext(result[0]) if result else ""
    plaintext = getattr(result, "plaintext", None)
    return str(plaintext) if plaintext else ""


def positive_control(
    attack_fn: Callable[[str], Any],
    structure_spec: dict[str, Any],
    key: dict[str, Any],
    *,
    plaintext: str | None = None,
    length: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Prove ``attack_fn`` recovers its own structure before any negative is trusted.

    Builds a synthetic from ``structure_spec``/``key`` via :func:`make_synthetic`, runs
    ``attack_fn(ciphertext)``, extracts the recovered plaintext (see
    :func:`_extract_plaintext`), and judges recovery.

    Recovery is declared when the recovered plaintext **contains a 32-letter span of the
    true plaintext** (an exact structural recovery) *or* clears the calibrated readable-
    English bar from :func:`genuine_solve_signature` (``word_cov`` and ``qscore_per_char``).
    ``threshold`` overrides the word-coverage floor if given.

    Returns ``{recovered, score, word_cov, decode_preview, signature, plaintext_head}`` so
    a caller can both branch on ``recovered`` and eyeball the preview.  An attack that
    fails its own positive control has a bug — do not trust any negative it later reports.
    """
    synth = make_synthetic(structure_spec, plaintext, key=key, length=length)
    truth = synth["plaintext"]
    ciphertext = synth["ciphertext"]

    result = attack_fn(ciphertext)
    recovered_text = only_letters(_extract_plaintext(result))

    word_cov = long_word_coverage(recovered_text) if recovered_text else 0.0
    score = _qscore_per_char(recovered_text) if recovered_text else float("-inf")
    sig = genuine_solve_signature(len(truth))

    # Exact-span recovery is the strongest signal and is alphabet/scoring independent.
    span = min(32, len(truth))
    exact = bool(recovered_text) and truth[:span] in recovered_text

    cov_floor = threshold if threshold is not None else sig["word_cov"]
    readable = word_cov >= cov_floor and score >= sig["qscore_per_char"]

    return {
        "recovered": bool(exact or readable),
        "score": round(score, 4) if score != float("-inf") else None,
        "word_cov": round(word_cov, 4),
        "decode_preview": recovered_text[:64],
        "signature": sig,
        "plaintext_head": truth[:64],
    }


def _extract_plaintexts(result: Any, limit: int) -> list[str]:
    """Up to ``limit`` candidate plaintexts from whatever an ``attack_fn`` returned."""
    if isinstance(result, (list, tuple)):
        return [only_letters(_extract_plaintext(r)) for r in result[:limit]]
    single = only_letters(_extract_plaintext(result))
    return [single] if single else []


def control_battery(
    attack_fn: Callable[[str], Any],
    *,
    sibling: dict[str, str] | None = None,
    plant: dict[str, Any] | None = None,
    top_n: int = 5,
    span: int = 32,
) -> dict[str, Any]:
    """Grade an attack TRUSTED or VOID before any of its negatives enter the record.

    Two tiers, either or both (at least one required):

    * **Tier A — solved sibling.** ``sibling={"ciphertext", "plaintext"}`` is a case
      whose answer is known (a previously solved puzzle, a published test vector).
      The attack must reproduce a ``span``-letter run of the known plaintext within
      its top ``top_n`` candidates. This catches harness rot end-to-end — wrong
      input plumbing, broken scoring, a solver reading a file *path* instead of the
      ciphertext and scoring heap garbage (all observed in practice).
    * **Tier B — own plant at target length.** ``plant`` is a :func:`make_synthetic`
      spec (``structure_spec`` + key fields, optional ``plaintext``/``length``).
      The synthetic is built by THIS module's encoder — never by the attack under
      test, because a self-consistently-wrong implementation will happily recover
      its own mis-encoded plant. Recovery uses the :func:`positive_control` bar.

    The verdict vocabulary is the point: an attack that fails either tier is
    **VOID**, which is a different thing from its target result being negative —
    a VOID attack's null carries no information and must not be cited as evidence.
    Returns ``{"verdict": "TRUSTED"|"VOID", "tiers": [...]}``; feed a VOID straight
    to :meth:`buttcrack.evidence.Finding.voided`.
    """
    if sibling is None and plant is None:
        raise ValueError("provide a sibling case, a plant spec, or both")
    tiers: list[dict[str, Any]] = []

    if sibling is not None:
        truth = only_letters(sibling["plaintext"])
        need = truth[: min(span, len(truth))]
        cands = _extract_plaintexts(attack_fn(sibling["ciphertext"]), top_n)
        hit = next((i for i, c in enumerate(cands) if need and need in c), None)
        tiers.append(
            {
                "tier": "A:solved-sibling",
                "passed": hit is not None,
                "detail": (
                    f"known plaintext found at rank {hit + 1}/{len(cands)}"
                    if hit is not None
                    else f"known plaintext NOT in top {len(cands)} candidates"
                ),
            }
        )

    if plant is not None:
        spec = dict(plant)
        structure_spec = spec.pop("structure_spec", None) or {
            "structure": spec.pop("structure")
        }
        pc = positive_control(
            attack_fn,
            structure_spec,
            spec,
            plaintext=spec.pop("plaintext", None),
            length=spec.pop("length", None),
        )
        tiers.append(
            {
                "tier": "B:own-plant",
                "passed": bool(pc["recovered"]),
                "detail": (
                    f"plant recovered (word_cov={pc['word_cov']})"
                    if pc["recovered"]
                    else f"plant NOT recovered (word_cov={pc['word_cov']}, "
                    f"preview={pc['decode_preview'][:32]!r})"
                ),
            }
        )

    verdict = "TRUSTED" if all(t["passed"] for t in tiers) else "VOID"
    return {"verdict": verdict, "tiers": tiers}


def random_key(length: int, *, alphabet: str = "KRYPTOS", seed: int | None = None) -> str:
    """A random substitution key of ``length`` letters drawn from ``alphabet``.

    Used to build long-key synthetics (the one-time-pad-grade regime documented in the
    campaign), so a harness can confirm an attack that *should* fail blindly indeed does.
    """
    alpha = _alphabet(alphabet)
    rng = random.Random(seed)
    return "".join(rng.choice(alpha) for _ in range(length))
