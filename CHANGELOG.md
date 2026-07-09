# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); this project is pre-1.0
and not yet published, so changes are grouped by milestone rather than release.

## Unreleased

### Added (period "inner content" diagnostic — language vs. flattened under a detected period)
Generalized from a two-layer-cipher triage where a weak periodic signal repeatedly looked like
a plain Vigenere but was not.

- **`analysis.period_inner_content(letters, period)`** classifies the layer *underneath* a
  detected period. The mean per-column IoC at that period is a mapping-invariant ceiling (every
  monoalphabetic-per-column cipher — Vigenere/Beaufort/Porta/Quagmire — preserves it), so it
  measures the index of coincidence of the text under the periodic layer. Comparing it to English
  splits two cases the old triage conflated: **natural-language inner** (coset IoC ~ English → a
  plain Vigenere/Quagmire peel reads it) vs. **flattened inner** (coset IoC real but well below
  English → the layer under the period is NOT a language; it is a two-layer cipher over a
  polygraphic/digraphic inner, or a non-prose payload, and a plain peel will never read). Includes
  a reliability guard: with few letters/column even random text inflates the coset IoC, so a
  small-sample harmonic (e.g. period 35 = 5x7 on a short message) is flagged, not misclassified.
- **Wired into `diagnose`**: the periodic-polyalphabetic branch now picks the *true* period (the
  reliable candidate with the highest per-column IoC), reports whether its inner is language or
  flattened, and routes the recommendation accordingly (plain Quagmire vs. a periodic-digraphic /
  two-layer / crib attack) instead of always suggesting Vigenere/Quagmire.
- **`hill_kpa.crib_drag(crib, cipher, n, ...)`** — the known-plaintext Hill attack when you know a
  probable word but not WHERE it sits: slides the crib across every block-aligned offset, recovers
  the key at each (via `recover_matrix`), decrypts the whole message, and ranks the results. The
  scorer is pluggable (`callable(str) -> float`), defaulting to English quadgrams but accepting a
  custom recogniser (gzip compressibility, a coordinate/keyword token detector, minimum entropy)
  so a **non-English payload** — a route, coordinates, a key — can be found by structure rather
  than being rejected by an English-only gate.

### Added (blind 3x3 Hill recovery, Hill known-plaintext attack, diagonal routes)
Three general capabilities generalized from a polygraphic-cipher solve campaign (no
puzzle-specific logic in the toolkit).

- **Blind 3x3 Hill recovery by row decomposition** (`ciphers/_hill_recover.py`, wired into
  `Hill.crack`). The 3x3 key space (`26**9`) is far too large for a matrix brute and a
  hill-climb has no gradient, so the old crack only did 2x2. The new attack decomposes the
  key by ROWS: each decrypt-matrix row is a 3-covector (only ~1471 up to invertible scale),
  scored by an additive-invariant per-class chi-square of its decimated stream; the
  strongest rows are assembled into an invertible matrix and resolved to readable plaintext
  by quadgram. Supports an arbitrary index alphabet (plain or keyed, e.g. `KRYPTOS`) and an
  optional period-`q` additive schedule (the affine/periodic-additive generalisation).
  `Hill.crack` now attempts both 2x2 and 3x3 and returns a plain, round-trippable key for a
  pure Hill over the standard alphabet. Reliable from ~60 trigraph blocks; `pair_brute`
  rescues a single monogram-outlier row on shorter text. Regression: recovers a
  3x3-Hill-over-keyed-alphabet + period-2-additive construction from 153 letters.
- **Hill known-plaintext / crib attack** (`hill_kpa.py`) and the **mod-26 linear solver** it
  rests on. `solve_mod26(A, b)` solves `A x = b (mod 26)` — over a *ring*, so Gaussian
  elimination fails on the zero-divisors 2 and 13 — via CRT over mod 2 and mod 13, and
  **enumerates the full solution set** of rank-deficient systems (a short crib can leave the
  key underdetermined). On top of it: `recover_matrix` (classic n×n KPA from aligned known
  plaintext, crib may start at any block offset) and `recover_affine` (matrix + unknown
  period-`q` additive schedule, in any index alphabet).
- **Diagonal route transpositions** (`ciphers/route.py`). Adds both diagonal families
  ("/" = `diag`, "\\" = `maindiag`) over all four corners, plain and serpentine (16 routes),
  alongside the existing row/column/spiral routes, each usable as a write-in or read-out
  route and verified to be a cell permutation (invertible).
- Tests: `test_hill_recover.py`, `test_hill_kpa.py`, and diagonal-route cases in
  `test_cc_route.py` (blind affine-Hill recovery, mod-26 solver edge cases incl. zero-divisor
  and underdetermined systems, KPA round-trips, route permutation/round-trip).

### Added (n-gram linear-relation detector — homophonic-expansion / tri-square family)
A new general detector for the cipher shape where each plaintext symbol expands to a fixed-length
n-gram of ciphertext letters and the plaintext channel is recovered as a **linear combination** of
the n-gram's positions in a keyed-alphabet index space, with the remaining positions free
homophones/nulls (the Delastelle three-square / "tri-square" family). This shape defeats every
standard test — each positional stream is flat (IoC ~ random), there is no period and no repeats —
yet a specific `sum_k coef[k]*index(C[k]) (mod 26)` reconstructs an above-floor channel.
- **`ngram_relation.scan(text, n=3, ...)`** enumerates every small-integer combination (default
  coefficients `{-1,0,1}`), in each candidate alphabet, and null-calibrates by shuffling each
  positional stream independently (preserves per-position histograms, destroys any real relation).
  Reports ranked candidates with a per-combination `z` and a **search-aware `p`** (so trying many
  combinations doesn't inflate significance).
- **`ngram_relation.combine(text, coef, alphabet, n=)`** extracts the recovered channel for a chosen
  relation, ready to feed to the substitution / quagmire / transsub solvers for its residual layer.
- **`butt relation`** CLI subcommand (`--n --alphabets --samples --seed --top`, JSON-aware) prints
  the ranked relations and, on a hit, the extracted channel.
- Tests: synthetic homophonic-expansion round-trip (scanner recovers the planted relation and
  `combine` reproduces the exact plaintext), plus clean/random negative controls.

### Added (gap-audit round 2 — block-granular reveal, higher-order fitness, key inversion)
Six general capabilities from a code-grounded, adversarially-verified gap audit (no PK-specific
logic in the toolkit); the through-line is making the *block/unit* (n-gram-granular) transposition
family — the layered "recognizable permutation over a periodic substitution" shape — a first-class,
CLI-reachable attack:
- **Higher-order fitness is reachable.** `--ngrams` now accepts `quintgrams` and `hexagrams`
  (English tables were already bundled but unselectable). Hexagrams are the sharpest
  real-English-vs-quadgram-salad separator on the hard keyless families; the stale "ships only up to
  quadgrams" notes in the docs/scorer are corrected.
- **`unit=` (block granularity) threaded through the reveal search.** `crack_columnar_reveal_enum`,
  `reveal_spectrum` (now a `(width, unit)` sweep), and their search-aware nulls take `unit`, and
  `analysis.search_aware_null(unit=g)` shuffles `g`-letter BLOCKS as tokens (a letter-shuffle null
  under a block construction is too easy to beat → false positives). A trigraph-block columnar over a
  Quagmire III is now recovered at `--unit 3` where the letter-only search is blind to it.
- **Exact-binomial, all-residue block-alignment detector.** `block_transposition_signal` now finds
  the max-count residue over ALL residues (a phase-offset grid at residue ≠ 0 was previously missed)
  and scores it with the exact binomial tail `p=(1/b)^k` (replacing a normal-approx z with a
  `count ≥ 6` gate that silently dropped legitimate small-k signals like a 5× trigram, p=0.004).
- **`diagnose`/`stats` route the block signal to `--unit`.** `diagnose` runs the fingerprint,
  reports `best_block`/residue/p, and *prepends* `butt transsub --unit b` (and its divisors) to the
  recommended attacks; `stats`/`diagnose` human output print the block-alignment line.
- **Reveal tools exposed on the CLI.** `butt transsub` gains `--enum` (brute every read-order,
  incl. non-dictionary, at any `--unit`), `--spectrum` (per-width/unit reveal + beats-null verdict),
  and `--orders` (a confirm-or-die decider that scores hypothesised orders vs a block-aware null) —
  previously library-only, so a CLI-driven agent could not reach them.
- **Transposition-key inversion.** `keyfinder.keyword_from_order` maps a recovered read-order back to
  the keyword(s) that induce it; `keyfinder.describe_permutation` labels an order with the human
  generator that produces it (reverse, rotate-k, riffle, odd/even interleave). Exposed as
  `butt keyword --order` and attached to `--enum`/`--orders` output, so a recovered permutation
  self-reports whether it is a word or a recognizable construction.

### Added (gap-audit follow-ups — confidence gating, block geometry, crib & known-alphabet deciders)
Six more general capabilities from a code-grounded gap audit (no PK-specific logic in the toolkit):
- **Calibrated solve-confidence gate.** `validate.solve_confidence(text)` returns
  `{word_coverage, qscore_per_char, recovered}` — a *clearly-English vs n-gram-salad* verdict
  (deliberately looser than the strict canonical-solve bar so it does not false-negative legitimate
  short-word English). Now attached to the result dicts of `quagmire_solver.solve` /
  `solve_fixed_alphabet`, `homophonic.solve`, and `joint.solve_config`, and surfaced by the CLI
  (`[NOT RECOVERED — likely not English …]`), so a 3%-correct blind decode is no longer printed like
  a real solve.
- **Beaufort fixed-alphabet solve + CLI crash fix.** `_decrypt`/`_encrypt`/`_best_shifts` take a
  `beaufort` flag and `solve_fixed_alphabet(kind="beaufort")` solves Beaufort (over the keyed
  alphabet by default). Fixes `butt quagmire --kind beaufort`, which previously hard-crashed with
  `unknown kind 'beaufort'`.
- **Block (unit=3) transposition threaded through `transsub`.** `crack_transposition_over_sub(…,
  unit=3)` and `crack_double_columnar_keywords(…, unit=3)` undo trigraph-granular columnars via
  `columnar._decode_units`, so the validated reveal-IoC discriminator + keyword sweep operate on
  block-transposition geometry. CLI: `butt transsub --unit 3`. `unit=1` behaviour is byte-identical; the
  blind double-columnar SA stays letter-only (a block-SA gains nothing — the objective is gradient-less).
- **Known-alphabet sweep decider.** `transsub.sweep_known_alphabet(ct, orders, alphabet=, unit=,
  periods=)` undoes each candidate transposition order, fast-solves the inner periodic substitution
  with a *known* alphabet (`solve_fixed_alphabet`), and ranks vs a shuffle null — an instant,
  calibrated accept/reject for any hypothesised low-entropy order (a recognition primitive otherwise
  re-implemented by hand in scratch scripts).
- **Crib-anchored inner-sub-under-columnar solver, CLI-exposed.** `butt crib --inner-columnar
  --crib PREFIX [--widths …] [--periods LO HI]` runs `cribbing.solve` (joint column read-order +
  period-p Quagmire-key recovery by consistency backtracking — sidesteps the flat blind objective),
  which was previously reachable only from the module `__main__`. Module now has `tests/test_cribbing.py`.
- **Block-of-b transposition fingerprint.** `analysis.block_transposition_signal` z-scores the mod-b
  alignment of repeated n-grams: a block transposition or b-graph block cipher (e.g. Hill) forces
  every repeat to start at `== 0 (mod b)`, while plaintext / single-letter ciphers scatter them; with
  a flat letter-IoC it flags the "periodic substitution hidden inside a block transposition" structure
  (the block-transposition fingerprint). Wired into `analysis.analyze` (`block_transposition` key).

Tests: `tests/test_toolkit_generalizations.py`, `tests/test_cribbing.py`, `tests/test_quagmire_fixed_alphabet.py`.

### Added (block-transposition generalizations — block transposition, diagnostics, key-sourcing, Latin)
Seven reusable capabilities lifted out of the block-transposition-over-periodic-substitution work
(kept fully general; no puzzle-specific logic in the toolkit):
- **Unit-k (block) transposition.** `ciphers.columnar` now transposes atoms of any size via
  `_encode_units`/`_decode_units` and a `--unit` crack option (default 1 = the old letter
  columnar, unchanged). `unit=3` moves 3-letter blocks as indivisible tokens, so trigrams survive
  intact — the "trigraph-granular" transposition no single-letter primitive could express. Handles
  a short final atom unambiguously.
- **Repeats-excised digraph diagnostic.** `analysis.repeat_adjusted_stats` re-measures digraph IoC
  after excising the redundant occurrences of every ≥k-times n-gram, with a shuffle null, to tell
  *real diffuse pair-structure* from *a few exact repeats* (a deterministic block cipher's identical
  blocks, or repeated plaintext under transposition). Wired into `analyze` (`repeat_adjusted` key).
- **Family/sibling baseline.** `analysis.family_baseline(target, corpus)` reports whether a
  ciphertext's IoC fingerprint is *normal for its series* or a genuine outlier, and the closest
  sibling — so a flat IoC isn't mistaken for an exotic construction when the whole family is flat.
- **Corpus-derived key candidates.** `keysources.keys_from_corpus` enumerates key strings from a
  corpus of prior solutions — full texts (running keys), acrostics (word/sentence/line), word
  tokens, fixed-width windows, and reverse/atbash transforms — each tagged with provenance, usable
  as a substitution key, running key, or (via `columnar._read_order`) a transposition order.
- **Composed-key build & decompose.** `keysources.compose_key`/`decompose_key` build and *invert*
  the word-pair composed key `QuagmireKEYED(wordA, wordB)` (lcm period). `decompose_key`
  recovers the word pair from a periodic key via the self-validating "decode with one word yields
  the other repeated" check. (Reproduces a real 40-char composed key from its word pair.)
- **Latin n-gram scoring.** Bundled `latin_{monograms,bigrams,trigrams,quadgrams}.txt` (≈606k-letter
  corpus) and `latin` registered in `scoring.LANGUAGES` with a reference passage — classical/CTF
  plaintext is often Latin, and every cracker already takes `lang=`.
- **Length-scaled reveal-period cap + double-columnar null.** `transsub.reliable_period_cap(n)`
  scales the trustworthy inner-period ceiling with message length (≈16 letters/column) instead of a
  hard 18, so long messages are no longer artificially blinded to a ~36 period (while short ones
  stay honestly capped — an information limit, not a bug). The blind double-columnar SA path now
  also reports a `reveal_null` (search-aware shuffle calibration), the overfit guard previously only
  on the keyword path.
- **Fast fixed-alphabet periodic solve.** `quagmire_solver.solve_fixed_alphabet(ct, alphabet,
  kind=, periods=)` recovers only the per-period shifts when the keyed alphabet is *known* — the
  common Kryptos-family case (the real sculpture uses the `KRYPTOS` keyed
  alphabet). Per-column chi-square pre-pick + a shift-only coordinate-ascent polish on the n-gram
  scorer; orders of magnitude faster than `solve`'s alphabet annealing, so a wide period band
  sweeps cheaply. Same return dict as `solve`; accepts a keyword or a full 26-letter alphabet (and
  `ct_alphabet` for Quagmire IV). Exposed as `butt quagmire --alphabet KRYPTOS [--ct-alphabet …]`.
  Complements blind `solve` (unknown alphabet) — use this whenever the alphabet is given, since the
  alphabet is an isolated optimum the annealer wastes most of its budget rediscovering.

Tests: `tests/test_added_capabilities.py` (25 cases, incl. positive/negative controls and an
exact composed-key reproduction); `tests/test_quagmire_fixed_alphabet.py` (fixed-alphabet
recovery, period sweep, keyword-vs-alphabet input).

### Added (running-key screen — `butt runkey`)
- **`butt runkey` — the running-key screen.** Generalizes the
  keyword/dictionary sweeps to candidate **KEY-TEXTS** (a sibling puzzle's plaintext, a crib,
  a quotation) used as a cycled running key — the only tractable attack when the key is a long,
  non-repeating, non-dictionary string that nothing can *search*. For every
  (key-text × alphabet × convention) — KRYPTOS/standard × Vigenère/Beaufort/variant — it
  de-substitutes and ranks by the **IoC-outlier test** (`scoring.index_of_coincidence`, which is
  transposition-invariant): the true key snaps IoC to ~0.066 as a lone outlier vs the ~0.038
  floor. A reveal-winner whose de-sub is transposed-English is peeled by an exhaustive width-≤8
  `columnar` brute and gated honestly on `long_word_coverage` + `genuine_solve_signature` (every
  reveal-trial is peeled, since a running key's Vigenère/Beaufort de-subs are atbash-negations
  that tie in IoC). New module `runkey.py` (`screen_running_keys`, `running_desub`, `desub_ioc`);
  CLI `--keytext`/`--keytext-file`/`--alphabets`/`--conventions`/`--max-width`/`--no-peel`/`--ioc-floor`.
  Tests: `tests/test_runkey.py` (synthetic substitution-over-columnar, atbash/alphabet coverage,
  negative control). **Concrete solve:** a real chained puzzle whose substitution was a
  keyed-alphabet Vigenère running-keyed on the *previous* puzzle's plaintext, over a width-8
  columnar — de-sub IoC snapped to ~0.069 as a lone outlier and the columnar brute finished it.
  The insight: in a *chained* puzzle family, the substitution key can be the previous puzzle's answer.

### Added (campaign consolidation — `diagnose`, crib solvers, family knowledge)
- **`butt diagnose` — one-shot layered/composite structure triage.** Combines the calibrated
  period spectrum, kappa autocorrelation, IoC decay, evolving-keystream fingerprint, and the
  N/lcm crackability cliff into a single **structure-class verdict + ranked, concrete `butt`
  commands** to try next (periodic-substitution-OUTER / periodic-INNER-under-a-transposition /
  non-stationary keystream / transposition / monoalphabetic). The "what is this and how do I
  attack it" report that the cryptanalysis campaign needed on day one.
- **New `analysis` diagnostics:** `kappa_spectrum` (lag autocorrelation — z per lag vs random),
  `crackability_cliff`/`crackability_cliff_auto` (the N/lcm OTP-grade go/no-go: a periodic/product
  keystream is blind-recoverable only while it repeats ≳2.5 cycles, i.e. effective period ≲ len/4),
  and `decay_fingerprint` (names the closest evolving-keystream family, or flags a likely
  transposition artifact). All surfaced in `butt stats` and fed to `diagnose`.
- **`crib` crib-anchored solvers** (beyond the additive-Vigenère drag): `--product` (two-coprime
  keystream **union-find** solver — the one lever past the crackability cliff; a crib of length
  ≥ p+q−1 connects the key graph and decodes everything), `--keyed` (keyed-alphabet / Quagmire
  crib-drag), and `--autokey` (plaintext-autokey crib-unzip).
- **`transsub --keyword-pairs`** — the directed **double-columnar keyword-PAIR sweep**
  (length-matched dictionary/thematic keyword pairs → reveal-IoC pre-filter → inner solve →
  search-aware-null gate), plus `transsub.reveal_spectrum` for `diagnose`.
- **`layered.solve_inner_periodic`** — the inner-substitution workhorse: multi-alphabet
  (KRYPTOS + standard) × Vigenère/Beaufort/variant × period-band, per-column chi-seed then
  quadgram coordinate-ascent. `transsub`/`layered` now use it, generalising the layered crackers
  beyond KRYPTOS-Quagmire-Vigenère.
- **`validate.py` — validate-on-synthetic harness.** `make_synthetic`, `positive_control`
  (confirm an attack recovers a *same-structure* synthetic before trusting a negative), and
  `genuine_solve_signature` (the calibrated readable-English bar, ~−4.2 qscore/char & ~0.69
  word-coverage at len ~272). The discipline that caught real solver bugs and makes negatives
  trustworthy.
- **Family-knowledge plumbing:** the layered/`transsub`/`validate` stack accepts a
  family-grammar spec (keyed alphabet, `rankorder` convention, thematic keyword +
  narrative-crib lists, confirmed recipes) as a JSON dict, so a campaign against a
  chained puzzle series can carry its accumulated knowledge between sessions.

### Added (keystream/layer-order triage diagnostics)
- **`analysis.ioc_decay` — non-stationarity diagnostic.** Splits the letters into equal
  segments, fits the slope of per-segment index-of-coincidence vs position, and z-scores
  it against shuffles of the same multiset. A strongly negative `slope_z` (≤ −2.5,
  surfaced as `non_stationary: true`) is the fingerprint of an **evolving / position-
  dependent keystream** (progressive-key, autokey, chain-addition/Gromark, dynamic-
  alphabet Chaocipher/Hutton, or OTP-grade) whose period is *not* recoverable — a crib is
  the lever. The key cross-check: a periodic/transposition cipher has **flat, stationary
  IoC**, so a real decay *rules out* the transposed-periodic family. Wired into `analyze()`,
  reported by **`butt stats`** as IoC drift, and routed in **`identify`**'s diagnosis to
  warn against chasing a period or transposition order when the keystream is non-stationary.
- **`analysis.search_aware_null` — selection-bias calibration helper.** A best-of-search
  statistic (best column order, best keyword, best reveal-IoC) is inflated simply because
  many candidates were tried, so the honest null is the **same search run on shuffles of
  the same letters**, not one random text. Returns `{observed, null_mean, null_max, z,
  beats_null_max}`; treat a maximum as signal only when it `beats_null_max`. Catches the
  concrete trap where a `transsub` double-columnar 'reveal' looked like z=7.8 against the
  random-text mean but collapsed to overfit under the proper best-of-set null.
- **`transsub` full incomplete-columnar enumeration + null gating.** The single-columnar
  reveal sweep now **exhaustively enumerates all column orders at small widths** (closing
  the non-dictionary-order gap a keyword sweep silently misses — numeric/random/off-wordlist
  keys), with the reveal-IoC maximum gated by `search_aware_null` so the larger search space
  can't manufacture a false positive.

### Added (layered solver — periodic substitution OVER a columnar transposition)
- **`butt layered` runs fully autonomously** (no flags needed): it auto-detects the
  substitution period from the raw-ciphertext column-IoC spectrum (the substitution is
  the outer layer, so its period shows there), then exhaustively brute-forces columnar
  read-orders up to width 8 — **parallelised across cores** (`--workers`) — for each
  candidate period, early-exiting once it reads clean English. `--period`/`--order`
  still force the guided path. End-to-end autonomous solve validated on a real puzzle.
- **`butt layered` cracks `CT = Quagmire(period p)( columnar_W( PT ) )`** to an EXACT,
  character-perfect plaintext (validated on a real puzzle: a long-period keyed-alphabet
  Quagmire III substitution over a narrow columnar — fully recovered). The
  columnar read-order is found by searching orders with a **deterministic full quadgram
  coordinate-ascent** of the shifts from the order-independent chi-square seed; running
  the recovery *to convergence* (not a fixed few passes) is essential — a truncated
  objective under-converges and ranks a near-miss order as best, plateauing ~80%, while
  a full climb separates the true order cleanly (≈−4.1/quadgram vs ≈−5.1 for wrong
  orders) with no restart noise. It is a periodic
  *keyed* (Quagmire/Vigenère) substitution applied over a columnar transposition, with
  an **unknown** column order. This is the sibling of the existing
  `engine._layered_additive_crack` (short-period *additive* sub over columnar) and
  `_layered_crib_crack` (mono-sub over columnar); it adds the keyed-alphabet, long-period,
  unknown-order case. Method: an **order-independent chi-square de-sub seed** (a
  transposition preserves monogram frequencies, so the per-column shift seed is computed
  once, independent of the column order) → **columnar order by hill-climb-with-restarts**
  (simulated annealing fails here; the order is a strong local max) → **shift recovery by
  quadgram coordinate-ascent + the 2-opt finisher**. Validated to exact recovery on known
  instances (`tests/test_layered.py`). New module `src/buttcrack/layered.py`.
- **Agent-native residual report.** When the period is long enough that columns hold only
  ~5 letters, ~20% of columns are *entangled* (a quadgram window straddles two columns)
  and their true shift is **not** the n-gram optimum — recovery plateaus at ~80% and no
  amount of quadgram/5-gram search resolves them. Rather than silently emit the
  n-gram-optimal (wrong) letters, `layered.column_alternatives` exposes, per ambiguous
  substitution column, the top candidate shifts and the exact plaintext slots that column
  controls **in context**, for the driving agent (an LLM, which can judge real English
  where an n-gram counter cannot) to make the final call. Surfaced by `butt layered`.
  (Surfaced on a real long-period substitution-over-columnar puzzle — structure fully
  recovered, ~80% of plaintext, the long-period residual left to the agent.)

### Fixed (long-key Quagmire — deterministic 2-opt finisher)
- **Long-key Quagmire (I/II/III) now solves cleanly and deterministically.** Recovering
  the per-column Vigenère shifts by single-column coordinate ascent (1-opt) traps on
  *coupled* local optima for long keys (e.g. a period-40 key over 280 letters ≈ 7
  letters/column): two nearby columns are jointly wrong and no single-column move
  improves either, so the result was RNG- and time-budget-dependent (a loose
  `butt crack` could get lucky while `butt auto` returned a few garbled letters).
  Added a deterministic **2-opt finisher** (`_quagmire_solver._two_opt_polish`): since
  a quadgram window spans only four consecutive positions, only column pairs within
  cyclic distance 3 can be *jointly* coupled, so refining just those near pairs (each
  over all 26×26 shift combinations, with incremental window rescoring) breaks the trap
  in ~1 s from a cold start. Wired in as the final polish on the winning keyword in both
  `dictionary_attack` and `blind_attack`, with a guaranteed time budget so it runs even
  when the caller's deadline has elapsed. Regression: `tests/test_quagmire_longkey.py`.
  (Surfaced on a real period-40 Quagmire puzzle.)

### Changed (machine-readable manifest — schema v3)
- **`butt schema` now documents the `stats` and `identify` output contracts** (a new
  `command_outputs` block) and the conditional envelope fields `crib_confirmed` and
  `ambiguous_with`, which were emitted but undocumented. `SCHEMA_VERSION` bumped to
  **3**. Closes the discoverability gap where an agent reading only the manifest
  couldn't learn the shape of `stats`/`identify` JSON or the crib-confirmation field.
- **README + parity matrix + cryptanalysis-tips brought in sync with the engine work**
  (a doc audit pass): README now lists `--ngrams`/`--blind`, the `--max-n 5` quintgram
  build, the ADFGX/ADFGVX two-phase and large-key columnar crackers, `crib_confirmed`,
  and `chi_squared_per_letter`; the parity matrix no longer marks **ADFGVX** as
  `decode-only` (it has a best-effort blind crack, same engine as ADFGX); the stale
  "layered auto-decode" roadmap bullet is corrected (it ships).

### Added (crib-anchored layered crack)
- **`auto --crib WORD` now cracks a keyed (monoalphabetic) substitution over a
  columnar transposition** — the layered class blind search can't touch (no gradient).
  Because a mono substitution commutes with the transposition's reordering,
  `untranspose(ct) == sub(plaintext)`, so for the right column order the un-transposed
  stream contains `sub(crib)` as a contiguous window. The cracker brute-enumerates
  column orders (widths up to the factorial ceiling ~8), keeps each width's champion
  by the **quadgram of the partial decryption** the crib placement implies, then
  solves the residual simple substitution and keeps the decrypt that actually contains
  the crib (no crib ⇒ no result — never a guess). Runs only when the cheap sweep found
  nothing confident that already contains the crib. Reported as `substitution+columnar`
  with `crib_confirmed`. Handles either layer order; needs a crib ≥4 letters, a long
  message, and an enumerable width. (`engine._layered_crib_crack`.)

### Added (cryptanalysis engine)
- **Generic simulated-annealing search** (`search.anneal`): a shared, restart-able
  SA engine over any discrete key space (substitution alphabets, keyed alphabets,
  column orders) with Metropolis acceptance and geometric cooling — the
  AZdecrypt-class search the crackers otherwise reimplement ad hoc. Climbs out of
  the local optima that trap plain greedy hill-climbing on long keyed alphabets and
  layered ciphers. Foundation for the blind keyed-alphabet and layered solvers.

### Changed (restricted-alphabet guard generalized)
- The ADFGX false-positive fix is now a declarative `ciphertext_alphabet` on the
  base `Cipher` + a shared `ciphertext_alphabet_ok` guard applied centrally in
  `crack`/`auto`. A cipher whose ciphertext must be a strict small set (ADFGX,
  ADFGVX) is skipped when the input isn't mostly over it, so it can never "solve"
  the few matching letters of an unrelated message. (Digit-ciphertext ciphers
  already self-reject letter input; bacon self-guards on a two-symbol stream.)

### Added (pre-flight transforms)
- **`butt transform`** un-wraps format tricks before cipher analysis: reversal,
  decimation/null-strip (`--decimate PERIOD[:OFFSET]`), and conservative
  nested-encoding peel (base64 / hex / A1Z26). **`auto` auto-peels** a
  high-confidence nested encoding (e.g. base64-wrapped ciphertext) before its
  sweep. Reversal/decimation are left as a manual tool, not auto-applied — deciding
  the forward sweep "failed" is unreliable when an overfit looks confident, so the
  diagnosis points at `transform` instead of guessing.

### Added (blind ADFGX / ADFGVX recovery)
- **Blind two-phase crack for ADFGX and ADFGVX** (`ciphers/_fractionation.py`,
  shared by both). These are fractionation *then* columnar transposition, and the
  layers peel in order without a crib: (1) recover the **transposition** by the
  **digraph index-of-coincidence** — when the columns are un-transposed correctly,
  consecutive symbol pairs reconstitute the original digraphs, whose frequency
  profile is a 1:1 substitution of English letters (IoC ≈ 0.066); a wrong order
  flattens it, and the statistic is *mapping-independent* so it works without the
  square. (2) Solve the recovered digraph stream as a **simple substitution** (the
  Polybius square) by quadgram annealing. A tetragraph-IoC tiebreak plus a
  top-candidate phase-2 selection resolve the reading-order ambiguity that plain
  digraph-IoC (being sequence-invariant) can't. Verified end-to-end recovering both
  ciphers from ciphertext alone. Replaces the previous ad-hoc, joint ADFGX search
  and the ADFGVX no-op. Best-effort: needs a long message (>= ~200 fractionation
  symbols) and isn't guaranteed; expensive, so it's not run by `auto`'s short budget.

### Added (5-gram fitness infrastructure)
- **Optional quintgram (5-gram) scoring with graceful fallback.** The scorer was
  already n-agnostic; this wires it up end to end: `scripts/build_ngrams.py --max-n 5`
  builds a quintgram table, and `crack`/`auto` take **`--ngrams {trigrams,quadgrams,
  quintgrams}`** to select the fitness model. buttcrack ships only up to quadgrams (a
  good quintgram table needs a large corpus), so `scoring.resolve_scorer` falls back
  to quadgrams when the requested table isn't present and the CLI prints a one-line
  note — asking for a sharper model never errors. `scoring.ngram_table_available`
  reports which tables are built.

### Added (large-key columnar transposition)
- **Columnar `crack` recovers wide keys by simulated annealing.** Exhaustive
  column-order search dies at the factorial wall (8! = 40320 is the ceiling), so a
  long transposition key was previously uncrackable. Widths past 8 now use
  `search.anneal` over the read-order permutation (swap-two-columns move, per-quadgram
  fitness) — a column order has a real n-gram gradient, so SA converges where brute
  force can't (verified recovering a width-13 key). The default (`max_width` 7) stays
  an exact brute force, so `auto` is unchanged; pass a larger `--width`/`--max-width`
  to reach long keys. When scanning multiple SA widths the time budget is split across
  them so an early width can't starve the rest.

### Changed (blind Quagmire recovery — shared engine + honest scoping)
- **Unified the Quagmire blind solver.** Quagmire II and III each carried their own
  ad-hoc, always-on simulated-annealing fallback that searched the keyed alphabet +
  rotations jointly. Both are replaced by one shared `_quagmire_solver.blind_attack`
  built on the generic `search.anneal`: it searches **only** the keyed alphabet and
  recovers the per-column cycleword **deterministically** (per-column chi-square),
  then polishes by quadgram — the stblake/AZdecrypt shape. Faster, deduplicated, and
  driven by the tested SA engine.
- **Blind recovery is now opt-in (`crack --blind`), not always-on.** Empirically, an
  arbitrary keyed alphabet is an *isolated* optimum with essentially no gradient — a
  single swap away from the true alphabet already scores like a random one — so the
  anneal rarely converges at puzzle lengths and previously just burned the whole
  timeout on every `crack quagmire2/3`. Default keyless cracking is now the reliable
  **keyword dictionary attack** (fast); `--blind` opts into the best-effort anneal
  (gated to >= 200 letters). The dependable levers for an unknown keyed alphabet
  remain the dictionary attack and a crib (`butt crib` / `--crib`). The deterministic
  cycleword recovery itself is exact given the right alphabet (regression-tested).

### Added (layer-detection stats)
- **Repeated-bigram periodicity detector** (`analysis.transposition_periods`, surfaced
  as `transposition_periods` in `butt stats`). Counts positions where a whole bigram
  *exactly recurs* at a given lag and z-scores it against shuffles of the same letters.
  This catches regular repeated structure at a fixed lag (repeated-key / route /
  periodic-fill patterns, or a repeated plaintext under a monoalphabetic sub). It is
  *not* a general columnar-transposition or homophonic-layer finder — an ordinary keyed
  columnar leaves no exact bigram repeats at its width and homophonic destroys repeats
  by design, so it stays silent there (verified). It fires only on a long, strong
  signal (absolute count >= max(6, 4% of length) AND z >= 4), so plain English does not
  false-spike; it stays silent rather than guess. `stats` also now reports
  **`chi_squared_per_letter`** (a length-normalized "are the letter frequencies still
  English?" reading that, when low alongside scrambled order, points at a transposition).

### Added (honest detection diagnostics)
- **Layered-cipher hint in the `diagnosis`.** When a periodic substitution
  *additively* de-substitutes to English letter frequencies but not readable text,
  identify now flags it as *"likely an (additive) substitution OVER a transposition;
  de-substitute, then attack the inner transposition."* (A *keyed* substitution over
  a transposition can't be told from a plain keyed one cheaply, so it deliberately
  doesn't guess there — no confident-but-wrong "layered" label.)
- **`identify` (and the `auto` identify block) now report a calibrated period
  spectrum (`periodic_ioc`) and a plain-language `diagnosis`.** For polyalphabetic
  text it names the significant period and letters-per-column, flagging long-key
  cases ("~7 letters/column; recovery uncertain, try a crib"); when no period is
  significant it explicitly warns against assuming running-key and points at
  crib-drag. Aimed at the failure mode that misdiagnosed a long-key Quagmire as
  running-key.

### Added (layered ciphers)
- **`butt pipeline --step cipher:key …`** chains decode (or `--encode`) steps for a
  layered cipher (superencipherment), peeling one method per step with an
  inspectable trace (`engine.pipeline`). Test vector: a KRYPTOS-keyed Vigenère over
  a double columnar transposition.
- **`auto` now cracks the tractable layered case automatically** — an *additive*
  substitution over a transposition (Nicodemus-like). When the single-layer sweep
  finds nothing, it de-substitutes by per-column chi-square and transposition-cracks
  the result; a hit is reported as e.g. `vigenere+columnar` with both keys and a
  `layered` note. (A *keyed* outer can't be peeled this way, so it's left honestly
  unsolved — no false layered claim, verified on an unsolved panel.)

### Added (crib support)
- **`--crib WORD` on `crack`/`auto`** — a guessed plaintext word (not the answer)
  that confirms and boosts any candidate whose plaintext contains it; sets
  `crib_confirmed` in the envelope. A cipher-agnostic verifier that surfaces the
  right cipher/key and guards against confident-but-wrong decrypts.
- **`butt crib --crib WORD`** — crib-drag locator for the Vigenere family: slides
  the crib and reports, per placement, the implied key fragment scored by n-gram
  fitness. For a running key the implied fragment is itself English, so this pins
  the crib's position and hands back a chunk of the key — the lever that breaks a
  long/running key when pure statistics stall. Needs no knowledge of the answer.
  `--top N` (default 6) caps the placements shown per cipher.

### Changed
- **Test suite runs in parallel** (`pytest-xdist`): fast tests via `-n auto`
  (~1 s); the stochastic `@pytest.mark.slow` solver tests stay **serial** (they're
  wall-clock-timeout bounded, so sharing CPU starves them into flaking). CI runs
  the two as separate steps.

### Added
- **Long-key Quagmire/Vigenère cracking.** Two gaps closed so a keyed-alphabet
  Vigenère with a *long, non-word* key (e.g. period 40 on 280 letters) now solves:
  (1) **calibrated period detection** — `analysis.calibrated_periods` z-scores
  per-column IoC against a random baseline and scans to ~length/5, surfacing a long
  period whose short (~7-letter) columns make the absolute IoC look like noise;
  exposed as **`periodic_ioc`** in `butt stats`. (2) **random-restart** quadgram
  shift recovery in the Quagmire solver (a single greedy pass gets trapped on many
  short columns) — a cheap 1-greedy filter picks the keyword+period, then a heavy
  restart polish converges the winner. `auto`/`crack quagmire3` now return
  `verdict: solved` on long-key Quagmire III ciphertexts that previously read as
  "no period / running key".
- **Quagmire keyword dictionary attack** (`quagmire1/2/3`): blind hill-climbing of
  a keyed alphabet doesn't converge at ACA message lengths, so `crack` now recovers
  the per-column shifts by quadgram for each candidate keyed-alphabet keyword. A
  built-in famous-keyword list (KRYPTOS first) is tried automatically; `--wordlist`
  supplies your own. Recovered keys round-trip through `decode`. With this, a
  192-letter KRYPTOS-keyed Vigenère (Quagmire III) now solves out of the box —
  `butt auto` and `butt crack quagmire3` return `verdict: solved`.
- **Overfit guard:** confidence is now deflated by *long-word coverage*. A
  stochastic solver can produce text that scores like English on n-grams but is
  gibberish; if it can't be tiled into real ≥5-letter words it's penalised, so it
  no longer reads as `solved` or out-ranks a genuine decrypt in `auto`
  (`words.long_word_coverage`). The signal is exposed as a new **`word_coverage`**
  field on every candidate and the result envelope (schema bumped to **v2**), so
  agents can distinguish "low confidence: too short" from "low confidence: scores
  like English but isn't language."
- **[Cryptanalysis tips & tricks](docs/cryptanalysis-tips.md)** — a playbook for
  driving `butt` against an unknown ciphertext (fingerprint → period → family →
  attack), including the failure modes above.

### Added (prior)
- **All 64 CryptoCrack cipher types**, each `encode`/`decode` validated against a
  published test vector (`docs/cryptocrack-parity.md` tracks per-cipher solver
  coverage: full / best-effort / decode-only).
- **Discoverability:** `butt schema` (machine-readable capability manifest),
  `butt help <cipher>` (key format + working example), `list --verbose`, and a
  `key_format`/`key_example` on every cipher (surfaced in `list --json`).
  Results carry a `schema_version`.
- **Tools:** `identify --types` (statistical per-type classifier), `keyword`
  (key/key-square finder), `words` (dictionary match/pattern/anagram/ngram),
  `convert` (A1Z26), `format`, `split`, `stats --contacts`, NDJSON `solve` batch.
- **Multi-language** scoring: `--lang {english,french,german,spanish,italian}`
  with bundled n-gram tables; `crack --wordlist` dictionary cracking.
- Quadgram scoring with calibrated, sample-size-aware confidence; `verdict`/
  `margin` trust signals; clean `{ok:false,error}` envelopes; CI (3.10–3.12).

### Fixed
- **ADFGX `auto` false positive:** on a 280-letter all-alphabet ciphertext, `auto`
  reported `verdict: likely, adfgx (0.74)` — impossible, since ADFGX ciphertext is
  written only in A/D/F/G/X. The cracker had been silently discarding the other
  ~83% of letters and "solving" the handful that matched (a short candidate that
  also slipped under the word-coverage gate's length floor). It now returns no
  candidate unless the input is essentially all A/D/F/G/X; `auto` on that input now
  reads `ambiguous` (honestly uncertain) instead.
- **`--wordlist` was silently dead:** the engine loaded it into the `keywords`
  opt but `interrupted-key` read `wordlist`, so dictionary cracking never received
  the words. Standardised on `keywords` (the Quagmire attack uses the same path).
- `encode` JSON no longer swaps `plaintext`/`ciphertext` — the fields name the
  plaintext-side / cipher-side text in both directions.
- `auto` runs cheap ciphers first and early-exits on a confident solve: a clear
  Caesar went from ~50 s (and a wrong, overfit `gronsfeld`) to ~0.04 s and the
  correct solve. Added a key-length Occam penalty against overfit longer keys.
