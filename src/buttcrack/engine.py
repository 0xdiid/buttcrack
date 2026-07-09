"""High-level operations: encode, decode, crack one cipher, or auto across all."""

from __future__ import annotations

import random
import time

from . import registry, transforms
from .identify import identify
from .result import Candidate, CrackResult
from .scoring import get_scorer, resolve_scorer
from .text import only_letters
from .words import long_word_coverage

# --- overfit guard --------------------------------------------------------
# A stochastic solver (square ciphers, substitution hill-climb) can manufacture
# text that scores like English on n-grams yet is gibberish — "aristocrat salad".
# The n-gram model is blind to it, so we deflate a candidate's confidence by how
# well its plaintext tiles into *long* real words: genuine prose ~0.5-0.8, salad
# ~0. This keeps a confident-looking overfit from being reported as a solve and
# from out-ranking a real (often simpler) decrypt in `auto`.
WORD_GATE_MIN_LEN = 40  # below this, long-word coverage is too noisy to trust
WORD_GATE_FULL = 0.35  # coverage at/above which confidence is left untouched
WORD_GATE_FLOOR = 0.30  # most confidence a zero-coverage candidate may keep


def _apply_crib(candidates: list[Candidate], crib: str | None) -> list[Candidate]:
    """Mark/boost candidates whose plaintext contains the crib (in place).

    A multi-letter crib match is strong, cipher-agnostic evidence the decrypt is
    real, so a hit gets a ranking boost that dominates calibrated confidence — it
    surfaces the right cipher/key in `auto` and flags it for the caller. Cribs are
    *guesses* (a likely word), so this never assumes knowledge of the answer.
    """
    target = only_letters(crib).upper() if crib else ""
    if not target:
        return candidates
    for cand in candidates:
        if target in only_letters(cand.plaintext).upper():
            cand.meta = {**(cand.meta or {}), "crib_confirmed": True}
            cand.rank_bias += 1.0  # a crib hit beats any confidence-only ranking
    return candidates


def _gate_candidates(candidates: list[Candidate], lang: str) -> list[Candidate]:
    """Record each candidate's long-word coverage and deflate confidence by it.

    Sets ``cand.word_coverage`` (None when not computed) and multiplies confidence
    by a [FLOOR, 1] gate, so a salad that scores like English on n-grams but tiles
    into no real long words is demoted — in place.
    """
    for cand in candidates:
        if lang != "english":
            continue  # coverage needs the English word list
        letters = only_letters(cand.plaintext)
        if len(letters) < WORD_GATE_MIN_LEN:
            continue  # too short to judge; leave the n-gram confidence alone
        cov = long_word_coverage(letters, minlen=5)
        cand.word_coverage = cov
        frac = min(1.0, cov / WORD_GATE_FULL)
        cand.confidence *= WORD_GATE_FLOOR + (1.0 - WORD_GATE_FLOOR) * frac
    return candidates


def encode(cipher_name: str, text: str, key: str) -> CrackResult:
    cipher = registry.get(cipher_name)
    start = time.perf_counter()
    out = cipher.encode(text, key)
    scorer = get_scorer()
    cand = Candidate(
        plaintext=out,
        cipher=cipher.name,
        key=key,
        score=scorer.score(out),
        confidence=scorer.confidence(out),
    )
    return CrackResult(
        cipher=cipher.name,
        ciphertext=text,
        operation="encode",
        candidates=[cand],
        runtime_ms=(time.perf_counter() - start) * 1000,
    )


def pipeline(
    text: str, steps: list[tuple[str, str]], *, op: str = "decode"
) -> tuple[str, list[dict]]:
    """Apply a sequence of ``(cipher, key)`` decode (or encode) steps in order.

    For a layered cipher (superencipherment), ``op="decode"`` runs the decryption
    recipe — outer layer first — peeling one method per step; ``op="encode"``
    applies them as given. Returns ``(final_letters, trace)`` where ``trace`` lists
    each step's cipher/key and its output, so the chain is inspectable.
    """
    cur = text
    trace: list[dict] = []
    for i, (name, key) in enumerate(steps, 1):
        cipher = registry.get(name)
        cur = only_letters((cipher.decode if op == "decode" else cipher.encode)(cur, key))
        trace.append({"step": i, "cipher": cipher.name, "key": key, "output": cur})
    return cur, trace


def decode(cipher_name: str, text: str, key: str) -> CrackResult:
    cipher = registry.get(cipher_name)
    start = time.perf_counter()
    out = cipher.decode(text, key)
    scorer = get_scorer()
    cand = Candidate(
        plaintext=out,
        cipher=cipher.name,
        key=key,
        score=scorer.score(out),
        confidence=scorer.confidence(out),
    )
    return CrackResult(
        cipher=cipher.name,
        ciphertext=text,
        operation="decode",
        candidates=[cand],
        runtime_ms=(time.perf_counter() - start) * 1000,
    )


def _load_wordlist(path: str) -> list[str]:
    """Read a candidate-keyword wordlist (one per line) for dictionary cracking."""
    from .text import only_letters

    with open(path, encoding="utf-8", errors="ignore") as fh:
        words = {only_letters(line) for line in fh}
    return sorted(w for w in words if w)


def crack(
    cipher_name: str,
    text: str,
    *,
    top: int = 5,
    seed: int | None = None,
    timeout: float | None = None,
    lang: str = "english",
    crib: str | None = None,
    ngrams: str = "quadgrams",
    **opts,
) -> CrackResult:
    cipher = registry.get(cipher_name)
    scorer = resolve_scorer(ngrams, lang)
    rng = random.Random(seed) if seed is not None else random.Random()
    wordlist = opts.pop("wordlist", None)
    if wordlist:
        opts["keywords"] = _load_wordlist(wordlist)
    start = time.perf_counter()
    if cipher.ciphertext_alphabet_ok(text):
        candidates = cipher.crack(text, scorer, top=top, rng=rng, timeout=timeout, **opts)
    else:
        candidates = []  # input can't be this cipher (wrong ciphertext alphabet)
    _gate_candidates(candidates, lang)
    _apply_crib(candidates, crib)
    return CrackResult(
        cipher=cipher.name,
        ciphertext=text,
        operation="crack",
        candidates=candidates,
        runtime_ms=(time.perf_counter() - start) * 1000,
    )


#: stop `auto` once a candidate reaches this raw confidence — later, more complex
#: ciphers can only overfit, never legitimately beat a genuine solve.
AUTO_EARLY_EXIT = 0.85


def _auto_sweep(
    text: str,
    *,
    top: int,
    seed: int | None,
    ciphers: list[str] | None,
    per_cipher_timeout: float | None,
    lang: str,
    ngrams: str = "quadgrams",
) -> tuple[list[Candidate], list[str]]:
    """One identify-ordered sweep across ciphers; returns (gated candidates, notes)."""
    scorer = resolve_scorer(ngrams, lang)
    explicit = ciphers is not None  # an explicit --ciphers list overrides ordering/skip
    targets = ciphers or sorted(registry.names(), key=lambda n: registry.get(n).complexity)
    all_candidates: list[Candidate] = []
    notes: list[str] = []
    best_conf = 0.0
    for name in targets:
        rng = random.Random(seed) if seed is not None else random.Random()
        try:
            cipher = registry.get(name)  # inside try: a bad --ciphers name must not sink auto
            if not explicit and not cipher.auto_crackable:
                continue  # keyless attack is ill-posed; skip in unguided auto
            if not cipher.ciphertext_alphabet_ok(text):
                continue  # input can't be this cipher (restricted ciphertext alphabet)
            found = cipher.crack(text, scorer, top=top, rng=rng, timeout=per_cipher_timeout)
        except Exception as exc:  # one cipher failing must not sink `auto`
            notes.append(f"{name}: crack failed ({exc})")
            continue
        # Deflate overfit (salad) confidence BEFORE the early-exit/ranking checks,
        # so a high-scoring-but-meaningless square/hill-climb candidate can neither
        # short-circuit the sweep nor out-rank a genuine (often simpler) decrypt.
        _gate_candidates(found, lang)
        # Occam prior: prefer the simpler cipher AND the shorter recovered key
        # (more key positions = more freedom to overfit) when confidences are close.
        for cand in found:
            key_len = cand.meta.get("key_length", 1) if isinstance(cand.meta, dict) else 1
            cand.rank_bias = -0.01 * cipher.complexity - 0.004 * key_len
            best_conf = max(best_conf, cand.confidence)
        all_candidates.extend(found)
        if not explicit and best_conf >= AUTO_EARLY_EXIT:
            notes.append(f"early-exit after confident solve ({name})")
            break
    return all_candidates, notes


# Inner transposition layers tried when peeling a substitution-over-transposition.
_INNER_TRANSPOSITIONS = [
    "columnar",
    "incomplete-columnar",
    "railfence",
    "redefence",
    "route",
    "myszkowski",
    "amsco",
]


def _layered_additive_crack(
    text: str,
    *,
    seed: int | None,
    per_cipher_timeout: float | None,
    lang: str,
    ngrams: str = "quadgrams",
) -> tuple[Candidate, str] | None:
    """Crack an *additive* substitution over a transposition (Nicodemus-like).

    De-substitute additively (per-column chi-square at the calibrated period); if the
    result reaches English letter frequencies but isn't readable, the substitution
    was additive and the disorder is a transposition — so crack that inner layer.
    Returns ``(candidate, note)`` or None. Only the tractable case: a *keyed* outer
    can't be peeled by chi-square, so it's left alone.
    """
    from .analysis import calibrated_periods
    from .scoring import chi_squared

    letters = only_letters(text)
    if len(letters) < 60:
        return None
    scorer = resolve_scorer(ngrams, lang)
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def best_shift(col: str) -> int:
        return min(
            range(26),
            key=lambda s: chi_squared("".join(alpha[(ord(c) - 65 - s) % 26] for c in col)),
        )

    for info in calibrated_periods(letters, top=3):
        if info["z"] < 3.0:
            continue
        period = info["period"]
        shifts = [best_shift(letters[j::period]) for j in range(period)]
        if not any(shifts):
            continue  # no actual substitution layer — a pure transposition the sweep handles
        inter = "".join(
            alpha[(ord(c) - 65 - shifts[i % period]) % 26] for i, c in enumerate(letters)
        )
        # Skip if the additive de-substitution is already readable (a plain Vigenere
        # the main sweep handles). Otherwise let the transposition crack be the
        # validator — it reads English only if the peel + inner attack are both right.
        if scorer.average(inter) >= scorer._english_ref - 1.2:
            continue
        cands, _ = _auto_sweep(
            inter,
            top=3,
            seed=seed,
            ciphers=_INNER_TRANSPOSITIONS,
            per_cipher_timeout=per_cipher_timeout,
            lang=lang,
        )  # _auto_sweep already gates by word coverage
        best = max(cands, key=lambda c: c.confidence, default=None)
        if best is not None and best.confidence >= 0.5:
            vig = "".join(alpha[s] for s in shifts)
            inner = best.cipher
            best.cipher = f"vigenere+{inner}"
            best.key = f"vigenere:{vig} | {inner}:{best.key}"
            best.meta = {
                **(best.meta or {}),
                "layered": True,
                "outer": "vigenere",
                "vigenere_key": vig,
                "inner_cipher": inner,
                "period": period,
            }
            return best, f"layered: peeled additive substitution (period {period}) over {inner}"
    return None


_A_Z = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _crib_pattern(word: str) -> tuple[int, ...]:
    """Letter-equality pattern of a word (UNIVERSE -> (0,1,2,3,4,5,6,4))."""
    seen: dict[str, int] = {}
    return tuple(seen.setdefault(c, len(seen)) for c in word)


def _layered_crib_crack(
    text: str,
    crib: str,
    *,
    seed: int | None,
    timeout: float | None,
    lang: str,
    ngrams: str = "quadgrams",
    max_width: int = 8,
) -> tuple[Candidate, str] | None:
    """Crib-anchored crack of a monoalphabetic substitution over a columnar transposition.

    The keyed layered class blind search can't touch (no gradient — see
    docs/cryptanalysis-tips.md §8): a keyed substitution combined with a columnar
    transposition. A *crib* is the lever. Because a monoalphabetic substitution is
    position-wise, it commutes with the transposition's reordering, so
    ``untranspose(ct) == sub(plaintext)`` for the right column order (whichever layer
    is outer). For that order the un-transposed stream therefore contains ``sub(crib)``
    as a contiguous window. We brute-enumerate column orders (widths up to the
    factorial ceiling), rank each by the **quadgram score of the partial decryption**
    implied by the best consistent crib placement — almost every wrong order either
    can't place the crib or yields gibberish, so the true order ranks first — then
    solve the residual simple substitution for the top candidates and keep the decrypt
    that actually contains the crib. Needs a crib (>= 4 letters) and a long message;
    width is limited to what we can enumerate in the budget.
    """
    from itertools import permutations

    from . import search
    from .ciphers import _fractionation as frac
    from .ciphers._quagmire_solver import _fast_table
    from .scoring import index_of_coincidence

    cribU = only_letters(crib).upper()
    letters = only_letters(text).upper()
    n = len(letters)
    if len(cribU) < 4 or n < 200:
        return None
    # Mono sub + transposition both keep English's ~0.066 IoC; skip polyalphabetic.
    if index_of_coincidence(letters) < 0.055:
        return None
    scorer = resolve_scorer(ngrams, lang)
    rng = random.Random(seed) if seed is not None else random.Random()
    deadline = (time.monotonic() + timeout) if timeout else None
    table, _ = _fast_table(scorer)
    cribpat = _crib_pattern(cribU)
    cribch = list(cribU)
    L = len(cribU)

    def best_crib_quad(u: str) -> float | None:
        """Quadgram of the best consistent crib placement's partial decrypt, or None."""
        ucodes = [ord(ch) - 65 for ch in u]
        best: float | None = None
        for p in range(len(u) - L + 1):
            win = u[p : p + L]
            if _crib_pattern(win) != cribpat:
                continue
            m: dict[str, str] = {}
            used: set[str] = set()
            ok = True
            for cu, cc in zip(win, cribch, strict=True):
                if (cu in m and m[cu] != cc) or (cc in used and m.get(cu) != cc):
                    ok = False
                    break
                m[cu] = cc
                used.add(cc)
            if not ok:
                continue
            mp = {ord(k) - 65: ord(v) - 65 for k, v in m.items()}
            dec = [mp.get(x, x) for x in ucodes]
            s = 0.0
            a, b, c = dec[0], dec[1], dec[2]
            for i in range(3, len(dec)):
                d = dec[i]
                s += table[((a * 26 + b) * 26 + c) * 26 + d]
                a, b, c = b, c, d
            if best is None or s > best:
                best = s
        return best

    # Phase 1: brute-enumerate column orders, keeping the BEST order per width by
    # crib-anchored partial quadgram. (Within the true width that best IS the true
    # order; pooling across widths is biased — a wide width has more spurious
    # placements — so we keep one champion per width and let the substitution solve
    # in phase 2 decide which width is real.)
    scored: list[tuple[float, list[int]]] = []
    for width in range(2, max_width + 1):
        if width > n // 2:
            break
        if deadline and time.monotonic() > deadline:
            break
        best_q: float | None = None
        best_o: list[int] | None = None
        for perm in permutations(range(width)):
            if deadline and time.monotonic() > deadline:
                break
            q = best_crib_quad(frac.untranspose(letters, list(perm)))
            if q is not None and (best_q is None or q > best_q):
                best_q, best_o = q, list(perm)
        if best_o is not None:
            scored.append((best_q, best_o))  # type: ignore[arg-type]
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)

    # Phase 2: solve the simple substitution for the top orders; keep the readable one.
    def solve_sub(u: str, dl: float | None) -> str:
        def fitness(state: list[str]) -> float:
            tbl = {c: p for p, c in zip(_A_Z, state, strict=True)}
            return scorer.score("".join(tbl[ch] for ch in u))

        best, _ = search.anneal(
            init=lambda: search.shuffled(list(_A_Z), rng),
            neighbour=search.swap_neighbour,
            score=fitness,
            rng=rng,
            restarts=10,
            iters_per_temp=250,
            temp0=8.0,
            cooling=0.92,
            min_temp=0.05,
            deadline=dl,
        )
        tbl = {c: p for p, c in zip(_A_Z, best, strict=True)}
        return "".join(tbl[ch] for ch in u)

    # Phase 2: solve the simple substitution for each width's champion (best first);
    # only the true order's decrypt contains the crib, so stop at the first that does.
    cand_orders = scored[:6]
    best_cand: Candidate | None = None
    for i, (_, order) in enumerate(cand_orders):
        dl = deadline
        if deadline is not None:
            now = time.monotonic()
            if now > deadline and best_cand is not None:
                break
            dl = now + (deadline - now) / (len(cand_orders) - i)
        plain = solve_sub(frac.untranspose(letters, order), dl)
        cand = Candidate(
            plaintext=plain,
            cipher="substitution+columnar",
            key=None,  # recovered (sub alphabet, column order) isn't a single keyword
            score=scorer.score(plain),
            confidence=scorer.confidence(plain),
            meta={
                "layered": True,
                "outer": "substitution",
                "inner": "columnar",
                "width": len(order),
                "read_order": order,
                "crib": cribU,
            },
        )
        if best_cand is None or cand.score > best_cand.score:
            best_cand = cand
        if cribU in only_letters(cand.plaintext).upper():
            best_cand = cand
            break
    if best_cand is None or cribU not in only_letters(best_cand.plaintext).upper():
        return None  # crib didn't surface => not this structure (don't emit a guess)
    w = best_cand.meta["width"]
    return best_cand, f"layered: crib-anchored substitution over columnar (width {w})"


def auto(
    text: str,
    *,
    top: int = 5,
    seed: int | None = None,
    ciphers: list[str] | None = None,
    per_cipher_timeout: float | None = 2.0,
    lang: str = "english",
    crib: str | None = None,
    ngrams: str = "quadgrams",
) -> CrackResult:
    """Identify, then crack across ciphers (cheap first) and rank all candidates.

    Ciphers run in ascending complexity so a simple cipher's confident solve
    short-circuits the slow, overfit-prone stochastic solvers. A small pre-flight
    first peels a high-confidence nested encoding (base64/hex/A1Z26) when detected.
    Other wrappers (reversal, regular nulls) are offered by `butt transform` rather
    than auto-applied — deciding the forward sweep "failed" is unreliable when an
    overfit looks confident, so we don't guess; the diagnosis points there instead.
    """
    start = time.perf_counter()
    notes: list[str] = []
    work = text
    if ciphers is None:  # pre-flight only for an unguided sweep
        encs = transforms.detect_encoding(text)
        if encs:
            work = encs[0]["decoded"]
            notes.append(f"pre-flight: peeled {encs[0]['kind']} encoding")

    info = identify(work)
    candidates, sweep_notes = _auto_sweep(
        work,
        top=top,
        seed=seed,
        ciphers=ciphers,
        per_cipher_timeout=per_cipher_timeout,
        lang=lang,
        ngrams=ngrams,
    )
    notes += sweep_notes

    # Always try peeling an additive substitution off a transposition underneath
    # (the tractable layered case). It self-guards — returns None unless the
    # additive-over-transposition signature actually holds — so it's cheap on
    # ordinary inputs and isn't blocked by an overfit that dodged the word gate.
    if ciphers is None:
        hit = _layered_additive_crack(
            work, seed=seed, per_cipher_timeout=per_cipher_timeout, lang=lang, ngrams=ngrams
        )
        if hit is not None:
            cand, note = hit
            candidates.append(cand)
            notes.append(note)

    # With a crib, attempt the *keyed* substitution-over-columnar class that blind
    # search can't crack — but skip it once some confident candidate ALREADY contains
    # the crib (a genuine solve), so a plain Caesar/Vigenère with a crib doesn't pay
    # for the brute enumeration. A merely confident *overfit* that lacks the crib does
    # NOT block it (that's exactly the case the crib is meant to rescue).
    if ciphers is None and crib and only_letters(crib) and len(only_letters(crib)) >= 4:
        cribU = only_letters(crib).upper()
        solved = any(
            c.confidence >= 0.65 and cribU in only_letters(c.plaintext).upper() for c in candidates
        )
        if not solved:
            hit = _layered_crib_crack(
                work,
                crib,
                seed=seed,
                timeout=max(per_cipher_timeout or 0.0, 30.0),
                lang=lang,
                ngrams=ngrams,
            )
            if hit is not None:
                cand, note = hit
                candidates.append(cand)
                notes.append(note)

    _apply_crib(candidates, crib)
    candidates.sort(key=lambda c: (c.confidence + c.rank_bias, c.score), reverse=True)
    return CrackResult(
        cipher="auto",
        ciphertext=text,
        operation="auto",
        candidates=candidates[: max(top, 1)],
        runtime_ms=(time.perf_counter() - start) * 1000,
        notes=notes,
        identify=info,
    )
