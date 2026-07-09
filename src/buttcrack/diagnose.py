"""One-shot structure triage for layered/composite ciphers.

The "what is this and how do I attack it" report distilled from a long layered-cipher
campaign — the question that cost the most time was always *what kind* of layered
cipher a flat-IoC ciphertext is, not running the eventual cracker.  This combines the
calibrated period spectrum, kappa autocorrelation, IoC non-stationarity (decay), the
evolving-keystream fingerprint, and the N/lcm crackability cliff into a single
structure-class verdict plus ranked, concrete ``butt`` commands to try next.

Pure / read-only — it never decrypts; run the recommended crackers for that.
"""

from __future__ import annotations

from typing import Any

from . import ngram_relation
from .analysis import (
    block_transposition_signal,
    calibrated_periods,
    crackability_cliff_auto,
    decay_fingerprint,
    ioc_decay,
    kappa_spectrum,
    linear_channel,
    period_inner_content,
)
from .cipher_id import SMALL_SAMPLE_COSET, period_significance
from .scoring import chi_squared, index_of_coincidence
from .text import only_letters

_RANDOM_IOC = 1.0 / 26.0


def diagnose(text: str) -> dict:
    """Triage ``text`` into a structure class with recommended attacks and the raw signals."""
    letters = only_letters(text)
    n = len(letters)
    if n < 24:
        return {
            "length": n,
            "reliable": False,
            "structure": "undetermined",
            "summary": "too few letters for statistical triage",
            "recommended": ["butt auto"],
            "signals": {},
        }

    ioc = index_of_coincidence(letters)
    chi2pl = chi_squared(letters) / n
    periods = calibrated_periods(letters, top=5)
    significant = [p for p in periods if p["z"] >= 3.0]
    decay = ioc_decay(letters) if n >= 96 else {}
    nonstat = bool(decay.get("non_stationary"))
    fingerprint = decay_fingerprint(letters) if n >= 96 else []
    kappa = kappa_spectrum(letters, max_lag=min(34, n // 6)) if n >= 60 else []
    cliff = crackability_cliff_auto(letters) if n >= 48 else {}
    block = block_transposition_signal(letters)
    best_block = block.get("best_block")

    # Reusable cryptanalytic deductions, gated on the measured stats below and emitted alongside
    # the structure verdict. ``flattened_reading`` marks the branches where >=1 flattening layer is
    # implied; ``headline_period`` is the period a branch chose to headline (checked for
    # small-sample).
    inferences: list[str] = []
    flattened_reading = False
    headline_period: int | None = None

    signals: dict[str, Any] = {
        "index_of_coincidence": round(ioc, 4),
        "chi_squared_per_letter": round(chi2pl, 4),
        "calibrated_periods": periods,
        "kappa_spectrum": kappa[:6],
        "ioc_decay": decay,
        "decay_fingerprint": fingerprint,
        "crackability": cliff,
        "block_transposition": {
            "best_block": best_block,
            "alignment": block.get("alignment"),
            "verdict": block.get("verdict"),
        },
    }

    # ---- structure-class triage (the routing that the campaign needed on day one) ----
    if ioc >= 0.059:
        if chi2pl <= 0.5:
            structure = "transposition (letters unchanged, order scrambled)"
            recommended = [
                "butt crack columnar",
                "butt crack route",
                "butt crack railfence",
                "butt auto",
            ]
            summary = f"IoC {ioc:.4f} ~ English, low chi2/letter: letters reordered, not remapped."
        else:
            structure = "monoalphabetic substitution (letters remapped)"
            recommended = ["butt crack substitution", "butt crack caesar", "butt auto"]
            summary = f"IoC {ioc:.4f} ~ English, high chi2/letter: single-alphabet substitution."
    elif significant and not nonstat:
        best = significant[0]
        lpc = n // best["period"]
        # Is the layer UNDER the period a language, or already flattened? (decides plain-
        # Quagmire vs. a two-layer / digraphic-inner / non-prose-payload attack.) Judge this on
        # the TRUE period: among candidates with enough letters/column (so a small-sample harmonic
        # like 35 = 5x7 on a short text can't inflate the coset IoC), take the one with the HIGHEST
        # per-column IoC. The true period maximises it — a sub-multiple mixes alphabets (flatter)
        # and a super-multiple has too few letters/column (excluded).
        reliable_periods = sorted(
            (p for p in significant if n / p["period"] >= 12), key=lambda p: -p["ioc"]
        )
        check_period = reliable_periods[0]["period"] if reliable_periods else best["period"]
        inner = period_inner_content(letters, check_period)
        signals["period_inner_content"] = inner
        flattened = inner.get("verdict", "").startswith("FLATTENED")
        headline_period = check_period if flattened else best["period"]
        flattened_reading = flattened
        if flattened:
            structure = (
                f"periodic polyalphabetic, period {check_period} (substitution OUTER) "
                "over a FLATTENED inner (NOT plain language)"
            )
            harmonic = (
                f" (top-z period {best['period']} is a small-sample harmonic;"
                f" the reliable fundamental is {check_period})"
                if best["period"] != check_period
                else ""
            )
            summary = (
                f"Period {check_period} is real (z={inner['z']}), but its coset IoC "
                f"{inner['coset_ioc']} is well below English {inner['english_ioc']}: the text "
                "UNDER "
                f"the period-{check_period} layer is not a simple language{harmonic}. A plain "
                "Vigenere/Quagmire peel will NOT read — this is a two-layer cipher (periodic sub "
                "over a polygraphic/digraphic inner: Playfair/Hill/fractionation) or a non-prose "
                "payload."
            )
            # Is that flattened inner LINEAR (Hill) or NONLINEAR (Playfair)? The linear-channel test
            # answers it directly and language-independently — it survives the outer periodic layer.
            lin = linear_channel(letters)
            signals["linear_channel"] = {"hit": lin.get("hit"), "verdict": lin.get("verdict")}
            if lin.get("hit"):
                summary += f" Linear-channel test: {lin['verdict']}"
                recommended = [
                    "butt crack hill   (a LINEAR/Hill channel is present under the period)",
                    "# peel the period-" + str(check_period) + " layer, then recover the Hill",
                    "butt crack slidefair",
                ]
            else:
                recommended = [
                    "butt crack slidefair   (periodic digraphic — Playfair slide)",
                    "butt crack seriated-playfair",
                    "# no linear channel: the inner is NONLINEAR digraphic (Playfair) "
                    "or fractionation",
                    "butt crib --keyed --crib <word>   (a crib is the lever when blind fails)",
                ]
        else:
            structure = f"periodic polyalphabetic, period {best['period']} (substitution OUTER)"
            summary = (
                f"Significant period {best['period']} (z={best['z']}), ~{lpc} letters/column; "
                "substitution is the outer layer over natural-language text (Vigenere/Quagmire "
                f"family). Coset IoC {inner.get('coset_ioc')} ~ English confirms a language inner."
            )
            recommended = ["butt crack quagmire3", "butt crack vigenere", "butt crack quagmire1"]
            if lpc < 12:
                summary += " Short columns (long key) — recovery uncertain; try a crib."
                recommended.append("butt crib --keyed --crib <word>")
            recommended.append("butt layered  (if a transposition sits UNDER the substitution)")
    elif nonstat:
        fam = fingerprint[0]["family"] if fingerprint else "unknown"
        structure = "non-stationary / evolving keystream"
        summary = (
            f"IoC drifts down the message (slope_z={decay.get('slope_z')}): an evolving keystream "
            f"(closest family: {fam}). CAVEAT: most evolving ciphers are IoC-stationary on "
            "average, so a real decay can also be a transposition artifact — verify first."
        )
        recommended = [
            "butt crib --autokey --crib <opening>",
            "butt crack progressive_key",
            "butt crack autokey",
            "butt transsub  (decay can be a transposition artifact)",
        ]
    else:
        structure = "polyalphabetic, no recoverable period"
        flattened_reading = True
        summary = (
            "Polyalphabetic (IoC < English) but no significant period: the substitution is "
            "likely INNER under a transposition (period scrambled), OR a long/aperiodic key."
        )
        recommended = [
            "butt transsub  (transposition over a periodic substitution)",
            "butt transsub --keyword-pairs --lengths 8,9,16,17 --wordlist <thematic>",
            "butt crib --product --p-range 5 20 --q-range 5 21 --crib <opening>",
            "butt crib --autokey --crib <opening>",
        ]
        # A flat inner with no period can be a HILL (bare, or under a transposition/route that
        # scrambled the block order) — the linear-channel test finds it language-independently.
        lin = linear_channel(letters)
        signals["linear_channel"] = {"hit": lin.get("hit"), "verdict": lin.get("verdict")}
        if lin.get("hit"):
            summary += f" Linear-channel test: {lin['verdict']}"
            recommended = ["butt crack hill   (a LINEAR/Hill channel is present)"] + recommended
        # A flat-IoC / no-period / no-repeats text divisible by a small n can be a homophonic
        # EXPANSION (tri-square family): each plaintext symbol -> an n-gram whose plaintext
        # channel is a fixed linear combination of the positions, the rest free homophones.
        # Nothing above sees it (every stream is flat); the linear-relation scan does.
        for gram in (3, 2):
            if n >= 4 * gram and n % gram == 0:
                rel = ngram_relation.scan(letters, n=gram, samples=400, seed=0, top=3)
                signals[f"linear_relation_n{gram}"] = {
                    "floor": rel.get("floor"),
                    "candidates": rel.get("candidates", [])[:3],
                }
                if "relation found" in rel.get("verdict", ""):
                    best = rel["candidates"][0]
                    summary += (
                        f" LINEAR RELATION at n={gram}: {best['alphabet']} coef={best['coef']} "
                        f"(IoC {best['ioc']} vs floor {rel['floor']}, p={best['p']}) — a "
                        f"homophonic-EXPANSION / tri-square-family cipher; extract the channel and "
                        "solve its residual layer."
                    )
                    recommended = [
                        f"butt relation --n {gram}  (confirm; then combine() the channel)",
                    ] + recommended
                    break

    if cliff:
        if cliff.get("recoverable"):
            summary += f" Crackability: {cliff.get('verdict')}."
        else:
            summary += (
                f" Crackability: {cliff.get('verdict')} — blind search may be hopeless;"
                " a crib is the lever."
            )

    strong_lags = [k["lag"] for k in kappa[:4] if k["z"] >= 3]
    if strong_lags:
        signals["structural_lags"] = strong_lags
        summary += f" Strong autocorrelation at lags {strong_lags} (possible period/grid widths)."

    # Block-of-b transposition fingerprint: repeated n-grams all share one residue mod b. This
    # is the day-one clue that the transposition (or a b-graph block cipher) moves whole blocks
    # — so the columnar/reveal search must run at that GRANULARITY (--unit b), not letter-wise.
    # A genuine block-of-b also aligns at every divisor, so surface b and its divisors as units.
    if best_block:
        units = [best_block] + [d for d in range(best_block - 1, 1, -1) if best_block % d == 0]
        summary += (
            f" BLOCK-ALIGNED: repeated {block['ngram']}-grams all sit at one residue mod "
            f"{best_block} (p={block['alignment'][best_block]['p']}) — a block-of-{best_block} "
            f"transposition or {best_block}-graph block cipher; run the transposition search at "
            f"this granularity (--unit)."
        )
        unit_cmds = []
        for u in units:
            unit_cmds.append(
                f"butt transsub --unit {u}  (reveal a periodic sub under a block-of-{u} "
                "transposition)"
            )
            unit_cmds.append(f"butt crack columnar --unit {u}")
        recommended = unit_cmds + recommended

    # ---- reusable structural inferences (generalizable cryptanalytic deductions) ----
    distinct = set(letters)
    lin_sig = signals.get("linear_channel") or {}
    lin_hit = bool(lin_sig.get("hit"))
    inner_sig = signals.get("period_inner_content") or {}
    coset_ioc = inner_sig.get("coset_ioc")
    min_period_ioc = min((p["ioc"] for p in periods), default=None)

    # (a) All 26 letters incl J, flat monograms, NO linear channel, fractionation-y: the emitted
    # alphabet itself constrains the inner grid. A 25-cell bifid square omits J; a 27-cell trifid
    # needs filler symbols. Seeing all 26 letters AND no fillers means the fractionation is wrapped
    # by an outer substitution that remapped its <=25-cell output back onto the full 26.
    fractionationy = ioc < 0.050 and not best_block
    if len(distinct) == 26 and fractionationy and not lin_hit:
        inferences.append(
            "all 26 letters incl J with no filler symbols implies an OUTER SUBSTITUTION over a "
            "<=25-cell fractionation (a 25-cell bifid cannot emit J; a 27-cell trifid emits filler "
            "symbols, which are absent)."
        )

    # (b) A below-random coset/column IoC is NOT proof that 'a monoalphabetic is impossible'
    # and does NOT exclude a polygraphic inner: flattening ciphers (Hill, fractionation) push
    # cosets below the 1/26 floor too, so a sub-floor coset is expected there, not diagnostic
    # against them.
    below_random = (coset_ioc is not None and coset_ioc < _RANDOM_IOC) or (
        min_period_ioc is not None and min_period_ioc < _RANDOM_IOC
    )
    if flattened_reading and below_random:
        inferences.append(
            "a below-random coset/column IoC is NOT diagnostic on its own: flattened ciphers "
            "(Hill, fractionation) also drive cosets below the 1/26 floor, so it neither proves "
            "'monoalphabetic impossible' nor excludes a polygraphic/fractionation inner."
        )

    # (c) Layer-count humility: the flat-IoC fingerprint proves flattening happened, not how many
    # layers stack — a periodic-sub-over-Playfair and a lone fractionation can print the same stats.
    if flattened_reading:
        inferences.append(
            ">=1 flattening layer present (the fingerprint cannot count its own layers): the "
            "statistics establish that at least one substitution/fractionation flattened the text, "
            "not the number of stacked layers."
        )

    # (d) Small-sample period caveat (wires in cipher_id.period_significance): don't headline a
    # period whose mod-p cosets are too thin to trust, even if its coset IoC looks elevated.
    if headline_period and headline_period >= 2:
        sig = period_significance(letters, periods=[headline_period], samples=60)
        _mean, lpc, pz, small = sig[headline_period]
        if small:
            inferences.append(
                f"CAUTION: period {headline_period} rests on only ~{lpc:.0f} letters/coset "
                f"(< {SMALL_SAMPLE_COSET}); its coset IoC is small-sample and may be an artifact "
                "that evaporates as the message lengthens — confirm on a longer sample or a "
                "smaller fundamental before committing to it."
            )

    if inferences:
        summary = summary + " || Inferences: " + " ".join(inferences)

    return {
        "length": n,
        "reliable": n >= 48,
        "structure": structure,
        "summary": summary,
        "recommended": recommended,
        "inferences": inferences,
        "signals": signals,
    }
