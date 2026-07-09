"""Running-key (known-key-text) attack with an IoC-outlier screen.

Productizes a running-key attack: a periodic substitution whose
key is **not a word** but a *running key* equal to some other text — most usefully a
sibling puzzle's plaintext ("each puzzle builds on the last" taken literally), but in
general any guessed key document (a known crib passage, a quotation, a prior message).
Such a key is in no wordlist and never repeats, so every dictionary / blind / period
attack fails. But if you can *name a candidate key text*, this attack finds it for free.

The lever is the **IoC-outlier test** (mapping/transposition-independent). De-substitute
the ciphertext with a candidate text as a running key (Vigenère / Beaufort / variant, over
a keyed *or* standard alphabet) and measure the index of coincidence of the result:

  * a **wrong** key text  -> IoC ~ 0.038 (the de-sub stays flat/random),
  * the **right** key text -> IoC ~ 0.066 (the de-sub is now English — *possibly still
    transposed*, because a transposition preserves IoC).

So the true running key stands out as a lone high-IoC outlier even when an *outer*
columnar still scrambles the letter order (the shape
``CT = Vigenère(keyed alphabet, running_key) OVER columnar``). For an outlier whose de-sub is
English-but-scrambled, a columnar brute (exhaustive over all read-orders for width <= 8,
``anagram`` SA above) peels the transposition and returns clean English. For an outlier that
is already readable, there was no transposition and it is returned directly.

This is the running-key analogue of the keyword/dictionary sweeps in :mod:`buttcrack.layered`
and :mod:`buttcrack.transsub`; the difference is that the key is *supplied* (a text you
guess) rather than *searched*, which is the only tractable attack when the key is a long
non-repeating string. Note we deliberately do **not** reuse :func:`transsub.reveal_score`:
that detects a *repeating* inner period, which a running key does not have — the correct,
transposition-invariant discriminator here is whole-text :func:`scoring.index_of_coincidence`.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations
from typing import Any

from .ciphers.columnar import _decode_letters
from .ciphers.quagmire3 import keyed_alphabet
from .scoring import NgramScorer, get_scorer, index_of_coincidence
from .text import only_letters
from .validate import genuine_solve_signature
from .words import long_word_coverage

STD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Decode (de-substitution) conventions, in keyed-alphabet index space:
#:   vigenere p = c - k    beaufort p = k - c    variant p = c + k
#: (matching the encode conventions in :mod:`buttcrack.validate`).
CONVENTIONS = ("vigenere", "beaufort", "variant")
ALPHABETS = ("KRYPTOS", "STD")

#: De-sub IoC at/above which a candidate is treated as "English revealed" (random ~0.038,
#: English ~0.066); a transposition keeps IoC, so this fires even when an outer columnar
#: still scrambles order.
REVEAL_IOC = 0.058

#: Widest column count whose read-orders are enumerated exhaustively (w! decodes) in the
#: peel; wider widths are handed to anagram.solve's SA.
ENUM_MAX_WIDTH = 8


def resolve_alphabet(name: str) -> str:
    """``'KRYPTOS'`` / a keyword -> its keyed alphabet; ``'STD'``/``'STANDARD'`` -> A-Z."""
    upper = (name or "").upper()
    if upper in ("STD", "STANDARD", STD):
        return STD
    return keyed_alphabet(upper)


def running_desub(ciphertext: str, key_text: str, *, alphabet: str = "KRYPTOS",
                  convention: str = "vigenere") -> str:
    """De-substitute ``ciphertext`` using ``key_text`` as a running key (cycled to length).

    Both ciphertext and key are read in ``alphabet`` index space. ``convention`` is one of
    :data:`CONVENTIONS`. The key is cleaned to letters and cycled (``i % len(key)``) — a
    true running key is at least as long as the message, but cycling makes the screen robust
    to a short candidate too. Semantically identical to
    :func:`buttcrack.validate.decode_substitution` (kept self-contained so this attack module
    does not depend on the synthetic-test harness)."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; expected one of {CONVENTIONS}")
    alph = resolve_alphabet(alphabet)
    idx = {c: i for i, c in enumerate(alph)}
    ct = only_letters(ciphertext).upper()
    key = only_letters(key_text).upper()
    if not key:
        raise ValueError("key text reduced to no letters")
    klen = len(key)
    out = []
    for i, ch in enumerate(ct):
        c = idx[ch]
        k = idx[key[i % klen]]
        if convention == "vigenere":
            p = (c - k) % 26
        elif convention == "beaufort":
            p = (k - c) % 26
        else:  # variant
            p = (c + k) % 26
        out.append(alph[p])
    return "".join(out)


def desub_ioc(ciphertext: str, key_text: str, *, alphabet: str = "KRYPTOS",
              convention: str = "vigenere") -> tuple[float, str]:
    """De-substitute under one running key/alphabet/convention; return ``(ioc, stream)``."""
    stream = running_desub(ciphertext, key_text, alphabet=alphabet, convention=convention)
    return index_of_coincidence(stream), stream


def _normalize(key_texts, labels) -> list[tuple[int, str, str]]:
    """``(input_index, label, text)`` triples from a dict, a list of strings, or a list of
    ``(label, text)``; an explicit ``labels`` sequence overrides. ``input_index`` is the
    candidate's position in the *original* input and survives the drop of no-letter entries
    (so a reported ``key_index`` always maps back to what the user passed)."""
    if isinstance(key_texts, dict):
        raw = [(str(k), str(v)) for k, v in key_texts.items()]
    else:
        raw = []
        for j, entry in enumerate(key_texts):
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                raw.append((str(entry[0]), str(entry[1])))
            else:
                lab = labels[j] if labels is not None and j < len(labels) else f"key{j}"
                raw.append((str(lab), str(entry)))
    return [(i, name, text) for i, (name, text) in enumerate(raw) if only_letters(text)]


def _peel_columnar(stream: str, scorer: NgramScorer, *, max_width: int
                   ) -> tuple[float, int, list[int], str]:
    """Best ``(score_per_char, width, read_order, plaintext)`` over a columnar undo.

    Width 1 (identity / no transposition) is always a candidate, so a *pure* running-key
    cipher is recovered too. Widths 2..min(max_width, 8) are searched exhaustively over all
    read-orders; for ``max_width > 8`` the wider widths are handed to
    :func:`buttcrack.anagram.solve` (multiple-anagramming SA), whose order convention matches
    :func:`buttcrack.ciphers.columnar._decode_letters`."""
    n = max(1, len(stream))
    best = (scorer.score(stream) / n, 1, [0], stream)  # identity (no transposition)
    for w in range(2, min(max_width, ENUM_MAX_WIDTH) + 1):
        for perm in permutations(range(w)):
            order = list(perm)
            pt = _decode_letters(stream, order)
            s = scorer.score(pt) / n
            if s > best[0]:
                best = (s, w, order, pt)
    if max_width > ENUM_MAX_WIDTH:
        from .anagram import solve as _anagram_solve
        r = _anagram_solve(stream, widths=range(ENUM_MAX_WIDTH + 1, max_width + 1))
        pt = r.get("plaintext") or ""
        if pt:
            s = scorer.score(pt) / n
            if s > best[0]:
                best = (s, int(r["width"]), [int(x) for x in r["order"]], pt)
    return best


def screen_running_keys(
    ciphertext: str,
    key_texts,
    *,
    labels: Sequence[str] | None = None,
    scorer: NgramScorer | None = None,
    alphabets: Sequence[str] = ALPHABETS,
    conventions: Sequence[str] = CONVENTIONS,
    max_width: int = 8,
    peel: bool = True,
    reveal_ioc: float = REVEAL_IOC,
    lang: str = "english",
) -> dict[str, Any]:
    """Screen candidate running KEY-TEXTS by the IoC-outlier test, then peel the winner.

    For every ``(key_text x alphabet x convention)`` trial it de-substitutes and measures
    IoC. The true key snaps IoC to ~0.066 (transposed English) as a lone outlier vs ~0.038.
    If the winner's IoC is an outlier (``>= reveal_ioc``) and its stream is not already
    readable, a columnar brute (exact widths <= ``min(max_width, 8)``, SA above) peels the
    transposition and the English plaintext is reported.

    Returns a JSON-serializable dict::

        {ok, operation, trials, winner, ranked, structure, plaintext, score,
         qscore_per_char, word_coverage, signature, recovered}

    ``winner`` carries ``{label, key_index, alphabet, convention, ioc, z, z_outlier,
    transposed_english}``; ``z`` is the winner's IoC vs the whole trial population, and
    ``z_outlier`` is its IoC vs **one representative (best) IoC per other distinct key-text**
    (so the winner's own alphabet/atbash siblings don't dilute it) — or ``None`` when there
    are fewer than two other distinct candidates to compare against. ``recovered`` is the
    honest gate: ``word_coverage >= 0.40`` (a clear-English floor far above n-gram "salad",
    not the optimistic ``genuine_solve_signature`` word bar) AND a per-char quadgram score
    within 0.4 nat of the calibrated bar."""
    scorer = scorer or get_scorer("quadgrams", lang)
    ct = only_letters(ciphertext).upper()
    items = _normalize(key_texts, labels)
    if not items:
        return {"ok": False, "operation": "runkey-screen", "trials": 0, "winner": None,
                "ranked": [], "recovered": False, "plaintext": "",
                "note": "no usable key texts"}

    trials: list[dict[str, Any]] = []
    for orig_i, name, text in items:
        for alphabet in alphabets:
            for convention in conventions:
                ioc, stream = desub_ioc(ct, text, alphabet=alphabet, convention=convention)
                trials.append({
                    "label": name, "key_index": orig_i, "alphabet": alphabet,
                    "convention": convention, "ioc": round(ioc, 4), "_stream": stream,
                })
    if not trials:
        return {"ok": False, "operation": "runkey-screen", "trials": 0, "winner": None,
                "ranked": [], "recovered": False, "plaintext": "",
                "note": "no alphabet/convention to try"}
    iocs = [t["ioc"] for t in trials]
    mu = sum(iocs) / len(iocs)
    sd = (sum((v - mu) ** 2 for v in iocs) / len(iocs)) ** 0.5 or 1e-9
    for t in trials:
        t["z"] = round((t["ioc"] - mu) / sd, 2)
    trials.sort(key=lambda t: t["ioc"], reverse=True)
    top = trials[0]

    # Peel / read EVERY reveal-trial, not just the top IoC one: for a running key the
    # vigenère and beaufort de-subs are atbash-negations of each other (identical IoC), so
    # the top-IoC trial may be the negated garbage — only one of the tied conventions reads
    # as English. Pick the trial whose de-sub yields the best long-word coverage.
    sig = genuine_solve_signature(len(ct))
    reveal_trials = [t for t in trials if t["ioc"] >= reveal_ioc] or [top]

    def _read(stream: str) -> tuple[str, dict[str, Any] | None]:
        raw_cov = long_word_coverage(stream)
        if raw_cov >= 0.40:                       # already readable: no transposition
            return stream, {"transposition": None}
        if peel and len(stream) >= 4:
            _, w, order, pt = _peel_columnar(stream, scorer, max_width=max_width)
            if w > 1:
                return pt, {"transposition": "columnar", "columnar_width": w,
                            "columnar_order": order}
            return pt, {"transposition": None}
        return stream, None

    best = None  # (word_cov, qscore, trial, plaintext, structure)
    for t in reveal_trials:
        pt, struct = _read(t["_stream"])
        wc = long_word_coverage(pt)
        qs = scorer.score(pt) / max(1, len(pt))
        if best is None or (wc, qs) > (best[0], best[1]):
            best = (wc, qs, t, pt, struct)
    word_cov, qscore, win, plaintext, structure = best
    transposed_english = win["ioc"] >= reveal_ioc
    recovered = word_cov >= 0.40 and qscore >= sig["qscore_per_char"] - 0.4

    # Lone-outlier z for the *reported* winner: compare its IoC against one representative
    # (the best) IoC per OTHER distinct key-text. This excludes the winner's own
    # alphabet/atbash siblings (same key_index) and needs >= 2 distinct others to be
    # meaningful; otherwise it is None (e.g. a single candidate key — judge by `recovered`).
    best_per_key: dict[int, float] = {}
    for t in trials:
        if t["ioc"] > best_per_key.get(t["key_index"], -1.0):
            best_per_key[t["key_index"]] = t["ioc"]
    others = [v for k, v in best_per_key.items() if k != win["key_index"]]
    if len(others) >= 2:
        omu = sum(others) / len(others)
        osd = (sum((v - omu) ** 2 for v in others) / len(others)) ** 0.5
        z_outlier = round((win["ioc"] - omu) / osd, 2) if osd > 1e-6 else None
    else:
        z_outlier = None
    # If the winner reveals transposed-English but didn't read out, the columnar may be wider
    # than the peel ceiling — tell the caller rather than silently returning a near-miss.
    width_hint = (transposed_english and not recovered and peel
                  and (structure or {}).get("transposition") != "columnar")

    ranked = [{k: t[k] for k in ("label", "key_index", "alphabet", "convention", "ioc", "z")}
              for t in trials]
    for r in ranked:
        r["transposed_english"] = r["ioc"] >= reveal_ioc
    out = {
        "ok": True,
        "operation": "runkey-screen",
        "trials": len(trials),
        "winner": {
            "label": win["label"], "key_index": win["key_index"],
            "alphabet": win["alphabet"], "convention": win["convention"],
            "ioc": win["ioc"], "z": win["z"], "z_outlier": z_outlier,
            "transposed_english": transposed_english,
        },
        "ranked": ranked,
        "structure": structure,
        "plaintext": plaintext,
        "score": round(scorer.score(plaintext), 2),
        "qscore_per_char": round(qscore, 3),
        "word_coverage": round(word_cov, 3),
        "signature": sig,
        "recovered": bool(recovered),
    }
    if width_hint:
        out["note"] = (f"winner reveals transposed-English (IoC {win['ioc']}) but did not read "
                       f"out — the columnar may be wider than max_width={max_width}; raise it.")
    return out
