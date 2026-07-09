# Cryptanalysis tips & tricks

A field guide for driving `butt` (human or agent) against an unknown ciphertext.
Distilled from real solves — especially the things that wasted time before they
became obvious.

## 0. Validate a solver before you trust its *failure*

The single highest-leverage habit. Before concluding "cipher X is ruled out
because its cracker returned gibberish," confirm the cracker can solve a **known**
instance of X at the same length:

```bash
CT=$(butt encode quagmire3 "$(head -c 200 some_english.txt)" --key "AUTOMOBILE/HIGHWAY")
butt crack quagmire3 "$CT"      # does it come back?
```

If it can't crack its own output at that length, a blank result tells you nothing
about your real ciphertext — you've learned about the *solver*, not the cipher.
Keyless hill-climbing of keyed-alphabet ciphers, in particular, is **infeasible at
ACA lengths (~200 letters)**; don't read its silence as "not this cipher."

Corollary: a *known-instance* solve also tells you when a plateau is a bug, not a
wall. If a known instance of your structure recovers to 100% but the real one
sticks at ~80%, the difference is a *wrong parameter you're holding fixed* (wrong
key length, wrong transposition width/order), not fundamental ambiguity — go hunt
the parameter, don't blame the cipher. (This is exactly how a stuck real-world
layered puzzle got unstuck: the columnar order was a near-miss.)

## 0b. When a search over a permutation/order plateaus, fix the *objective's convergence*, not the restarts

If you're searching a discrete order (column read-order, route, key permutation)
and ranking each candidate by "recover the rest, then score," the per-candidate
recovery **must run to convergence**. A *truncated* inner optimizer (a fixed few
passes) under-converges, so a near-miss candidate scores almost as high as the
truth and gets ranked first — you converge on a plausible-but-wrong answer and
plateau. A **deterministic full climb to convergence (no random restarts)** both
reaches clean English for the true candidate and removes restart noise, so
candidate scores are directly comparable and the truth separates by a wide margin
(e.g. ≈−4.1/quadgram for the true column order vs ≈−5.1 for near-misses). Reach for
more restarts only after the inner optimizer is actually converging. (Field note: an
order search with a 3-pass recovery objective ranked a near-miss order #1 for a day;
a full climb found the true order immediately.)

## 1. Fingerprint before you swing

```bash
butt stats <ct> --json      # IoC, Kasiski, likely periods, digram/contact data
butt identify <ct> --json   # family routing from IoC + letter-fit
```

- **Index of coincidence** is the first fork:
  - **~0.066** — monoalphabetic *or* transposition (both preserve single-letter
    frequencies). Try `caesar/affine/substitution`; if frequencies are English but
    order is scrambled, it's transposition (`columnar/railfence/...`).
  - **~0.038–0.045 (flat)** — polyalphabetic *or* fractionation. The plaintext's
    letter frequencies have been smeared.
- A flat IoC **rules out plain transposition** outright (it would keep 0.066).
- **Don't eyeball the monoalphabetic-vs-transposition call** — `butt stats`/`identify`
  report `chi_squared_per_letter`: **low (~<0.05)** means the letter frequencies still
  match English even though the order is scrambled (→ a **transposition**), while a
  **high** value means the letters were remapped (→ a **substitution**). It's the
  objective version of "are the frequencies still English?" that splits the 0.066 fork.

## 1b. Un-wrap format tricks first

A "cipher" that resists everything is often just *wrapped*. Before deep analysis,
`butt transform` un-wraps the common ones:

```bash
butt transform <ct>                 # reversal + detected base64/hex/A1Z26 decode
butt transform <ct> --decimate 5:0  # drop every 5th letter (strip a regular null)
```

`auto` auto-peels a high-confidence nested encoding (e.g. base64-wrapped
ciphertext) on its own. Reversal and null-stripping are *not* auto-applied — a
confident overfit can make a forward sweep look "solved", so deciding it failed is
unreliable; try `transform` by hand when a solve won't come.

## 2. Find the period (polyalphabetic)

Split the text into `period` columns and take the **mean per-column IoC**. The
true period is where it jumps back toward English (~0.066); multiples of it light
up too.

**Use a calibrated baseline, not an absolute threshold, and scan far enough.**
This is the trap that makes a long-key periodic cipher look like "no period":

> A long key on a not-very-long message gives *few letters per column*. With
> N = 280 and period 40, each column holds only **7 letters**, so its IoC is
> noisy and the absolute mean (~0.069) barely clears random (~0.0385) — easy to
> dismiss. But z-scored against random text split the same way, period 40 pokes
> **above the p99** (with harmonics at 20 and 8). The spike is real; the naive
> reading isn't.

So: scan periods up to ~**length / 5** (not just ≤15), and rank by z vs a
random-text baseline — `butt stats` now reports this as **`periodic_ioc`**
(`{period, ioc, baseline, z}`, z > ~3 is strong). `likely_periods` (Kasiski) is a
second, independent read. A clean peak at *p* **and** *2p* with troughs between is
solid even on ~190 letters.

Corollary: **"no obvious period" is NOT evidence of a running key.** Flat overall
IoC with no short-period spike is *also* what a long-key Vigenère/Quagmire looks
like under naive analysis. Check the calibrated long periods before concluding
running-key (see §7).

## 3. Is it Vigenère, or a *keyed-alphabet* Vigenère (Quagmire)?

Once you know the period, solve each column as a Caesar shift (per-column
chi-squared vs English):

- **Columns resolve to English** → plain **Vigenère / Beaufort / Gronsfeld**.
  `butt crack vigenere` nails it.
- **Columns are clearly monoalphabetic (high per-column IoC) but the best Caesar
  shift is gibberish** → the alphabet is **keyed**. This is a **Quagmire**
  (or Porta). The columns are simple substitutions of a *mixed* alphabet, not
  shifts of the straight one.

> If the columns are English-distributed yet no Caesar shift reads, stop trying
> Vigenère variants — you need the keyed alphabet.

## 4. Cracking a Quagmire: dictionary, not hill-climb

Hill-climbing a 26-letter keyed alphabet from scratch does **not** converge at
~200 letters (the correct alphabet is an isolated optimum). The reliable route is
a **keyword dictionary attack**, which `butt` now does automatically:

```bash
butt crack quagmire3 <ct>                 # tries famous keywords first (KRYPTOS, ...)
butt crack quagmire3 <ct> --wordlist words.txt   # your own candidate keywords
```

How it works (and why it beats hill-climbing): for each candidate keyword it
builds the keyed alphabet and recovers the per-column shifts by **quadgram
coordinate-ascent over the full text**, with **random restarts**. Three gotchas:

- **Use quadgrams, not chi-squared, to recover the per-column shifts.** At
  ~20 letters/column chi-squared is too noisy and picks wrong shifts *even with
  the correct alphabet*; the full-text quadgram signal pins them exactly.
- **Random restarts are mandatory for long keys.** A single greedy pass ranks the
  *right keyword* first by a wide margin (cheap filter), but on a long key — say
  period 40, 40 columns — greedy alone gets trapped in a local optimum and reads
  as near-miss garbage. Restart from random shift vectors and keep the best to
  converge to clean English. (`butt` does a cheap 1-greedy filter to pick the
  keyword+period, then a heavy random-restart polish on the winner.)
- **The keyword is often a proper noun or theme word, not a common word — and the
  Vigenère key may not be a word at all.** Try thematic alphabet keywords first —
  `KRYPTOS` above all (the CIA sculpture's keyed alphabet, the most common keyed
  Vigenère in the wild) — but note the *Vigenère key itself* can be a long random
  string. That's fine: you only dictionary-search the **keyed alphabet** keyword;
  the per-column shifts (the long key) are recovered by hill-climb, not guessed.

`butt` ships a small famous-keyword list (KRYPTOS first); pass a richer
`--wordlist` for alphabet keywords off it. It scans calibrated periods up to
~length/5, so a long key (period 40 on 280 letters) is found automatically.

### Why blind keyed-alphabet recovery (`--blind`) almost never works

If the keyed alphabet is *not* a word (so the dictionary attack can't reach it),
the only keyless option is to search the 26-letter alphabet directly. `butt` exposes
this as an opt-in:

```bash
butt crack quagmire3 <ct> --blind        # best-effort; needs >= 200 letters
```

It uses the right algorithm (simulated annealing over the alphabet, with each
candidate's cycleword recovered deterministically by chi-square, scored by
quadgrams) — and it still rarely converges. The reason is a flat fitness landscape:
**measured on a 660-letter Quagmire III, the true alphabet scores far above random
(−3.9 vs −6.4 per quadgram), but a single swap away from it already scores −6.1 —
i.e. like a random alphabet.** The optimum is an isolated spike with no surrounding
gradient, so a hill-climb/anneal that's one move away looks no better than a random
start. (Mechanically: a wrong alphabet lets the per-column chi-square still flatten
each column to English *monogram* frequencies, so the only signal that the alphabet
is wrong is cross-column *quadgrams* — and those collapse the moment any pair of
letters is misplaced.) This is why `--blind` is **off by default** (it would just
burn the timeout) and why, for an unknown keyed alphabet, the real levers are the
**dictionary attack** and a **crib** — not a bigger search budget.

### Quagmire III alignment footgun

A "Vigenère performed *in* the keyed alphabet" (e.g. KRYPTOS) aligns the key on
the **first letter of the keyed alphabet**, not on `A`. The decoder defaults to
ACA alignment `A`, so:

```bash
butt decode quagmire3 <ct> --key "KRYPTOS/GREENHOUSE"     # gibberish (align A)
butt decode quagmire3 <ct> --key "KRYPTOS/GREENHOUSE/K"   # correct (align K)
```

The crack searches alignment for you (it recovers raw shifts), so this only bites
manual `decode`.

## 4a. Cracking a columnar transposition (and *why* it works where Quagmire doesn't)

For a complete columnar, the unknown is the **column read-order**. Up to width 8
`butt` brute-forces all `width!` orders exactly; wider keys hit the factorial wall
(12! ≈ 480M), so it switches to **simulated annealing** over the order:

```bash
butt crack columnar <ct> --width 13          # SA: exact search is infeasible here
butt crack columnar <ct> --max-width 15      # brute widths 2..8, SA widths 9..15
```

This is the *opposite* situation to blind keyed-alphabet recovery (§4 above) and the
contrast is the whole lesson: **a column order has a real n-gram gradient.** Measured
on a 224-letter width-12 columnar, the true order scores −4.1/quadgram, random −6.9,
and the degradation is *monotonic* with distance — one swap off true averages −5.5,
two swaps −6.2, three −6.5. So a hill-climb/anneal that's part-way there sees which
swaps help and climbs the rest of the way (SA recovers width 13–15 in seconds). A
keyed alphabet has no such gradient (one swap ≈ random), which is exactly why SA
cracks transpositions but not substitution alphabets. **Before you reach for a
search-based solver, perturb the known answer by one move and check the score
degrades *gradually* — if it collapses to random, search won't help.**

The default `max_width` is 7 (so `auto` stays exact and fast); raise `--width` /
`--max-width` only when you suspect a long key. Transposition period detection is
weak, so for very long keys supply the width if you can.

## 4c. Cracking ADFGX / ADFGVX: peel the transposition with a *mapping-independent* stat

ADFGX/ADFGVX are fractionation (each letter → a two-symbol digraph over the row/col
labels) *then* a columnar transposition. The transposition's whole job is to
*separate the two halves of each digraph*, so attack it first — and you can, without
knowing the square, using a statistic that doesn't depend on the digraph→letter
mapping:

- **Digraph index-of-coincidence.** Un-transpose with a candidate column order, then
  read the stream as consecutive non-overlapping digraphs. When the order is correct
  the digraphs reconstitute the originals — a 1:1 substitution of English letters —
  so their IoC ≈ 0.066; a wrong order mixes halves of different letters and flattens
  it toward uniform. IoC is invariant to relabelling, so this finds the transposition
  with the square still unknown. Anneal the column order to maximise it.
- **Reading-order tiebreak.** IoC is *sequence-invariant*, so a wrong column order
  with the same digraph multiset ties the true one. Break the tie with the
  **tetragraph** (4-symbol block = digraph-bigram) IoC, which is higher in the true
  reading order (consecutive digraphs carry English letter-bigram lumpiness) — and as
  a backstop, solve the square for the top few candidate orders and keep whichever
  reads as English. (Plain digraph-IoC alone will hand you a *transposed* plaintext.)
- **Then the square is just a simple substitution** over ≤36 digraph symbols — a
  quadgram hill-climb (`butt` searches the 26 letters, widening to digits only if
  more than 26 distinct digraphs appear).

```bash
butt crack adfgvx <ct> --width 9     # forced width (faster)
butt crack adfgvx <ct> --timeout 90  # blind width scan (slow)
```

Best-effort and length-hungry: it needs a long message (≥ ~200 label symbols, i.e.
~100 plaintext letters) and isn't run by `auto`'s short budget. The classic WWI break
used *many* messages of equal length; a single short message may not separate.

## 4b. Cribs: the general lever (confirm *and* break)

A *crib* is a guessed plaintext word — a likely or thematic term, **not** the
answer. Two cheap, high-leverage uses:

```bash
butt auto <ct> --crib ARCHIVE        # confirm/rank: boosts any candidate containing it
butt crib --crib ARCHIVE <ct>        # crib-drag the Vigenère family / running key
```

- **`--crib` on `crack`/`auto`** is a cipher-agnostic *verifier*: a multi-letter
  match sets `crib_confirmed` and surfaces the right cipher/key. The fastest guard
  against a confident-but-wrong decrypt.
- **`butt crib`** slides the crib and reports the *implied key* per placement; for
  a running key the implied fragment is itself English, so it pins the position and
  hands you a chunk of the key to extend. This is what breaks a long/running key
  when statistics stall — and it needs no knowledge of the solution.

## 5. Telling a real solve from a confident-looking overfit

Stochastic solvers (the square ciphers, substitution hill-climb) can manufacture
text that **scores like English on n-grams but is gibberish** — "aristocrat
salad." The n-gram score alone cannot see this; it can even score *better* than
real English.

The cheap, reliable discriminator is **long-word coverage**: what fraction of the
text tiles into real dictionary words of ≥5 letters.

- Genuine English: **~0.5–0.8**
- Quadgram salad: **~0.0** (it tiles only into 2–3 letter fragments)

`butt` folds this into reported `confidence`, so an overfit candidate is deflated
below a genuine (often simpler) decrypt and won't be reported as `solved`. When in
doubt, eyeball the plaintext for ≥5-letter words — salad has almost none.

**Sharper fitness:** `crack`/`auto` take `--ngrams {trigrams,quadgrams,quintgrams,hexagrams}`
(default quadgrams). A higher-order model better separates real English from salad and
helps with the §7 plaintext↔key 4-gram symmetry. English **bundles quintgrams and
hexagrams**, so `--ngrams hexagrams` works out of the box — reach for it whenever a
candidate scores like English but reads as gibberish. Other languages ship up to quadgrams;
build higher orders with `python scripts/build_ngrams.py --max-n 6 corpus/*.txt` (a missing
table falls back to quadgrams with a one-line stderr note).

## 6. Don't trust `verdict` blindly on hard ciphers

`verdict`/`confidence` are calibrated, but on the genuinely hard keyless families
(Quagmire, Gromark, the squares at short length) they can still be optimistic.
Corroborate a "solved" with: does it read as sentences? do the recovered key and
period look like real words/structure? does it round-trip through `decode`?

## 7. Running-key is a last resort, not a first guess

`butt crack running-key` will return a **confident-looking (~0.97) but blended**
plaintext/key for almost any flat-IoC text — because it's *designed* to split text
into two English streams, and two natural-language streams are inherently
ambiguous (swapping plaintext↔key is a score symmetry a 4-gram model can't break;
disambiguating needs ~6-gram stats or the known source text). That high
confidence is a **trap**: it does not mean the cipher is running-key.

Before believing "running key", rule out the cheaper explanations of "flat IoC,
no obvious period":

1. **A long-key periodic cipher** (Vigenère / Quagmire). Run the calibrated
   long-period scan (§2) up to ~length/5 — a period-40 key on 280 letters hides
   here. If a period is significant, it's periodic, not running-key.
2. **A keyed-alphabet periodic** with a non-word key — solvable by the dictionary
   + restart hill-climb (§4), no source text needed.

Only if no period is significant at any length *and* the keyed-alphabet attacks
fail should you treat it as running-key — and even then, report it as a blended
two-stream decomposition, not a clean solve.

## 8. What `butt` can and can't blind-crack (honest limits)

Layered (super-enciphered) ciphers split into a tractable and an intractable class,
and the line is exactly where a *mapping-independent* statistic survives:

- **Additive substitution over a transposition** (Nicodemus-like) — **automated**
  (`auto`): de-substitute by per-column chi-square, then crack the inner
  transposition (§2 calibrated period + the transposition crack).
- **ADFGX / ADFGVX** (fractionation then transposition) — **automated** (best-effort,
  long messages): the transposition peels by *digraph-IoC* because fractionation lets
  the transposition split a symbol, leaving a strong mapping-independent signal (§4c).
- **Keyed (monoalphabetic) substitution over a columnar transposition** — **not
  blind-automated, but cracked with a crib** (`auto --crib WORD`). Blind it's
  gradient-free: `untranspose(ct) = sub(plaintext)`, so in principle you'd recover the
  transposition by *bigram*-IoC then solve the substitution, but the bigram-IoC
  separation for single letters is weak (~0.069 vs ~0.063 random), the true column
  order is a narrow peak the anneal misses, and a nested sub↔transposition hill-climb
  converges to readable-looking-but-wrong fixed points — so blind search is left out
  rather than shipped as a confident-but-wrong solve. **A crib is the lever, and it's
  implemented:** since a mono sub commutes with the transposition, `untranspose(ct)`
  contains `sub(crib)` as a contiguous window for the *right* column order. `auto
  --crib WORD` brute-enumerates column orders (widths up to the factorial ceiling ~8),
  ranks each by the **quadgram of the partial decryption** the crib placement implies
  — almost every wrong order can't place the crib consistently or yields gibberish, so
  the true order's champion wins — then solves the residual substitution and keeps the
  decrypt that actually contains the crib (no crib ⇒ no result, never a guess). Needs a
  crib ≥4 letters, a long message, and a transposition narrow enough to enumerate.
- **Periodic (Quagmire/Vigenère) substitution over a *single* columnar** — **automated**
  (`butt layered --period P`). The substitution
  is the *outer* layer, so its period is visible in the *raw* ciphertext column-IoC
  spectrum (§2). Attack: (1) a chi-square de-sub seed is *order-independent* (a
  transposition preserves monogram frequencies), so compute it once; (2) recover the
  columnar read-order by **bruting orders** (small widths) scored by a **deterministic
  full quadgram coordinate-ascent run to *convergence* (no restarts)** of the shifts —
  the true order separates cleanly (≈−4.1/quadgram vs ≈−5.1 for wrong orders); (3)
  recover the per-column shifts. **Pitfall that cost a day:** a *truncated* recovery
  objective under-converges, so a near-miss order scores ~as high as the truth and gets
  ranked first (the solve plateaus ~80%). Run the per-candidate optimizer to
  convergence — see §0b. When columns are very short (~5 letters) and a few stay
  ambiguous, `butt layered` emits a per-column "alternatives in context" report for an
  agent/LLM to finalize.
- **Keyed/polyalphabetic substitution over a *double* columnar** — **not automated.**
  Double columnar is hard alone and a keyed/Quagmire outer can't be peeled by
  chi-square or a cheap mapping-independent statistic. `butt` *decodes* it given the
  recipe (`pipeline`) and *diagnoses* the structure, but won't blind-crack it.
- **Homophonic substitution over a transposition** (the Zodiac Z340 shape) — **not
  automated.** Homophonic substitution (many cipher symbols per plaintext letter)
  *deliberately destroys* the repetition that every cheap statistic relies on: there's
  no mapping-independent lever to peel the transposition, and solving the substitution
  itself is an AZdecrypt-class search that needs a 5/6-gram model (`butt` bundles
  quintgrams/hexagrams — `--ngrams hexagrams`) *and* ~10 000 symbols to converge; the
  symbol count, not the model, is the binding constraint at ACA lengths. `butt`
  cracks the *structured ACA Homophonic* type (a numeric cipher with only four unknown
  column shifts) and detects regular repeated-bigram periodicity (`stats`
  `transposition_periods`), but the general Z340-class blind solve is out of scope.

The unifying test (see §4b): a search-based solver only works when its fitness has a
gradient — perturb the known answer by one move and check the score degrades
*gradually*. Transposition orders and fractionation squares pass; keyed substitution
alphabets (and the keyed-over-transposition product) don't, which is why those stay
crib- or dictionary-driven.

## 9. The layer-order / keystream triage (the single biggest time-saver)

You have **flat IoC and no obvious short period.** Three very different ciphers look
identical here, and each wants a *different* attack. Spend two minutes splitting them
apart before you swing — guessing wrong burns hours on a search that *can't* work.

| Case | What it is | Tell | Attack |
|---|---|---|---|
| (a) | **Periodic substitution, OUTER** | calibrated **colIoC spike** at the period in the *raw* text | the periodic solver at that period |
| (b) | Periodic substitution, **INNER**, under a transposition | **no** raw spike — but undoing the right transposition makes the spike **reappear** | `butt transsub` (reveal discriminator) |
| (c) | **Non-stationary / evolving keystream** | IoC **drifts down** the message (`ioc_decay`, `slope_z <= -2.5`) | none blind — needs a crib |

**(a) Periodic substitution as the OUTER layer.** The substitution period is visible in
the *raw* ciphertext column-IoC spectrum (§2) — a calibrated colIoC spike at the period
(harmonics at multiples). `butt layered` handles this shape. The period
showing in the raw text is exactly what marks the substitution as outermost.

**(b) Periodic substitution hidden UNDER a transposition.** A transposition applied
*over* a periodic substitution **destroys the raw spike** — the columns are shuffled, so
the period is invisible in the raw spectrum. The lever is **mapping-independent**: undo a
candidate transposition and the per-column IoC spike **reappears** at the inner period,
because the inner substitution's per-period columns go monoalphabetic again *regardless of
which keyed alphabet was used* (the IoC doesn't depend on the alphabet → no need to solve
the substitution first to peel the transposition). This is the `reveal_score`
discriminator productized in `butt transsub` (single columnar by keyword sweep, double by
SA over the two read-orders). Crucially, `reveal_score` has a usable **gradient before the
substitution is solved**, where a quadgram objective does not — which is why search peels
this layer where it can't peel a keyed alphabet (§4, §8).

**(c) Non-stationary / evolving keystream.** Progressive-key, autokey, chain-addition
(Gromark), dynamic-alphabet (Chaocipher, Hutton), or OTP-grade keystreams are
*position-dependent*: they start more structured and grow toward random, so per-segment
IoC **decreases along the message.** `butt`'s new `ioc_decay` fits that slope and z-scores
it against shuffles of the same letters; `slope_z <= -2.5` (`non_stationary: true`) is the
fingerprint. There is no period to recover and no transposition to peel — a crib is the
only lever.

> **The decisive cross-check:** a transposed-periodic cipher (cases a/b) has **FLAT,
> STATIONARY IoC** — every segment reads the same in expectation, because transposition
> just reorders letters and a *periodic* keystream repeats. So a **real IoC decay RULES
> OUT** the transposed-periodic family outright. If `ioc_decay` fires, stop hunting for a
> period or a transposition order; you're in case (c) and only a crib will move it.

## 10. Search-aware nulls: calibrate against the *shuffled search*, not one random text

Any **best-of-search** statistic — best column order, best keyword, best reveal-IoC — is
inflated by **selection bias.** You tried thousands of candidates and kept the maximum;
structureless text will hand you a surprisingly high maximum *simply because so many
candidates were tried.* Comparing that maximum to **one** random text (or to the
random-text mean) is the wrong null and manufactures false positives.

The honest null is the **same search run on shuffles of the same letters** —
`analysis.search_aware_null(letters, search, samples=…)`, which returns
`{observed, null_mean, null_max, z, beats_null_max}`. Treat the result as signal only when
it **beats the shuffled-search max** (`beats_null_max`), not just the mean.

> **The concrete trap (it cost real time):** a `transsub` double-columnar 'reveal' looked
> like a z=7.8 spike against the random-text mean — apparently a slam-dunk transposition.
> Run against the proper best-of-set null (the *same* SA search on shuffled letters), it
> collapsed: the shuffles' best reveal-IoC sat right at the 'discovery', so it was overfit,
> not structure. The gate that catches this is *also* the §5/§8 rule restated for searched
> statistics: a high score is only signal once it clears the null appropriate to **how it
> was selected.**

## 11. Validate-on-synthetic before you trust a NEGATIVE

A negative ("this attack found nothing, so it isn't cipher X") is **worthless from an
un-validated attack** — see §0, here sharpened for layered/keystream work. Before
believing a blank result:

1. **Confirm the attack RECOVERS a same-structure synthetic** at the same length. Encode a
   known plaintext with the exact construction you suspect and check the attack returns it.
   If it can't crack its own output, its silence on the real text tells you nothing.
2. **Pin the genuine-solve signature** so you can recognize the real thing when it appears
   and detect a near-miss plateau. For a short (~270-char) panel a validated English solve
   typically sits near **qscore/char ≈ −4.2** and **word_coverage ≈ 0.69**; a candidate
   scoring like English on quadgrams but with low word-coverage is salad, and a plateau a
   hair below the signature is a *wrong fixed parameter* (period, width, order), not the
   cipher being unbreakable.

This discipline is what lets a negative stand as a *validated* "blind-unbreakable" rather
than a guess: every attack in scope is first shown to recover a synthetic of the same shape
at the same length, and only then run on the real text.

## 12. The N/lcm ≈ 2.5 crackability cliff (the OTP-grade test)

A periodic or product keystream is blind-recoverable **only while it repeats enough.** The
useful single number is the **effective period** vs message length: a key (or the lcm of
two combined keystreams) is recoverable while

> effective period **< ~length / 4** (equivalently, N / lcm **>~ 2.5** cycles in the message).

Past that cliff the keystream barely repeats, every statistic flattens, and — *even if you
are handed the correct period* — the columns hold too few letters to fix and the
construction collapses to noise. This is mapping-independent and unsentimental: a product
keystream with N/lcm ≈ 2.5 is OTP-grade and needs a **crib or a known opening**, while a
repeating period-45 substitution over a width-8 columnar at the same length cracks clean.
When you compute a candidate effective period, check it against this cliff *before* investing in
a search; below ~2.5 cycles, no search budget substitutes for a crib.

## 13. Close the dictionary-order gap with incomplete-columnar full enumeration

A keyword sweep over column orders only reaches orders that *spell a word* — it silently
misses every non-dictionary permutation, including the true one when the transposition key
is numeric, random, or not in your wordlist. At **small widths** the fix is just to
**enumerate all orders exhaustively** (incomplete-columnar / complete-columnar, all
`width!` permutations), gated by §10's search-aware null so the larger search space doesn't
manufacture a false positive. This closes the gap a keyword sweep leaves and is cheap while
the width keeps `width!` tractable (≤ ~8); past that, fall back to SA over the order (§4a).

## 14. The running-key screen — when the key IS another text (a sibling answer / a crib)

When every dictionary / keyword / blind / period attack fails **and the puzzles are a
chain**, the key may not be a word at all — it may be a *running key* equal to some other
text: a **previous puzzle's plaintext** ("each builds on the last" taken literally), a known
crib passage, a quotation. Such a key is in no wordlist and never repeats, so nothing
*searches* it — but if you can *name a candidate key text*, you can *test* it for free.

The lever is the **IoC-outlier test**, which is *transposition-invariant*. De-substitute the
ciphertext with a candidate text as a running key (Vigenère/Beaufort/variant, KRYPTOS *and*
standard alphabet) and read the index of coincidence of the result:

- **wrong** key text → IoC ~0.038 (stays flat/random),
- **right** key text → IoC ~0.066 — the de-sub is now English, *possibly still transposed*
  (a transposition preserves IoC), so the true key is a lone high-IoC outlier even when an
  outer columnar still scrambles the order.

```bash
butt runkey <ct> --keytext-file sibling_plaintext.txt --json
# winner snaps IoC to ~0.066 as a lone outlier; if its de-sub is transposed-English the
# winner is peeled by an exhaustive width<=8 columnar brute (§4a) and the English is read.
```

Crucially **do not screen with `transsub`'s `reveal_score`/per-column IoC** (§4a/§9/§13): a
running key does *not* repeat within the message, so the reveal spectrum stays flat — that's
the wrong tool. *Whole-text* IoC is the right discriminator. Two gotchas: (1) IoC ≠ readable —
the outlier's stream is *transposed* English, so you must peel the columnar before trusting it;
gate on `long_word_coverage`, never IoC alone (§5). (2) Vigenère and Beaufort de-subs of a
running key are atbash-negations of each other and **tie in IoC**, so peel *every* reveal-trial
and keep the one that actually reads English, not just the top-IoC one.

> **A running-key aha (field note).** One real puzzle in a series was
> `Vigenère(keyed alphabet, key = the previous puzzle's plaintext, cycled)` over a
> width-8 columnar. De-substituting the raw CT with the prior plaintext as a keyed-alphabet
> running key gave IoC **≈0.069** — a lone outlier vs the ~0.038 floor that every *other*
> sibling's plaintext, alphabet, and convention produced; an 8! columnar brute then
> returned the read-order at qscore/char ≈ −4.20. The whole automated arsenal (period scans,
> layered, transsub, fractionation) had failed because the key is non-periodic and
> non-dictionary; the screen finds it in one shot. In any "each puzzle builds on the last"
> series, try the *prior* puzzle's plaintext as the running key before anything fancy.

## 15. Block-GRANULAR transposition: when the shuffle moves *n-grams*, not letters

A transposition need not move single letters — it can relocate whole **`b`-letter blocks**
(e.g. trigraph blocks). Every reveal/columnar tool defaults to `unit=1`
(letters) and is **blind** to a block permutation: undoing a block-of-3 columnar *as letters*
never re-exposes the inner substitution. Two independent tells put you here, and one flag fixes it.

**The fingerprint (free, in `stats`/`diagnose`).** A block-of-`b` cipher relocates whole blocks,
so **every repeated ciphertext n-gram must start at one residue mod `b`** — `butt stats` and
`butt diagnose` report `block_transposition` with the largest such `b`, its residue, and the
**exact binomial p** (`p=(1/b)^k` when all `k` reliable repeats align; a phase-offset grid at
residue != 0 is caught too). `diagnose` then *prepends* `--unit b` (and its divisors) to the
recommended attacks. This is the day-one clue that says "search at block granularity."

**The lever (`--unit`).** Run the reveal search at that granularity:

```bash
butt diagnose <ct>                                   # -> "block-of-3 ... try --unit 3"
butt transsub --spectrum --unit 3 <ct>               # which (width, unit) re-exposes a sub? beats-null?
butt transsub --enum --unit 3 --min-width 6 --max-width 6 <ct>   # brute ALL width-6 block-orders
butt transsub --orders '3,0,5,1,4,2;...' --unit 3 <ct>  # CONFIRM-OR-DIE specific recognition hypotheses
```

`--enum` reaches **non-dictionary** block orders a keyword sweep can't; the decider (`--orders`)
scores hypothesised orders against a **block-aware shuffle null** (shuffling *blocks*, not letters —
a letter-shuffle null under a block construction is too easy to beat and manufactures false
positives, §10). A recovered order self-reports via `keyword --order` -> its keyword (if it spells
one) or a named generator (`reverse`, `rotate-k`, riffle, ...) — so a "recognizable" permutation
announces itself instead of reading as random.

> **Why this is the whole game for the block-transposition shape.** The reveal-IoC has a gradient *before* the
> substitution is solved (§9), so a block-order search converges where a keyed-alphabet search
> can't — but only if it moves the right-sized atoms. Get `--unit` from the block-alignment
> fingerprint, not by guessing.

## 16. Blind polygraphic (Hill) recovery: decompose the key by ROWS, not as a matrix

An `n x n` Hill key lives in a `26**(n*n)` space — `26**9 ~ 5.4e12` for 3x3 — so a matrix
brute is hopeless and a hill-climb has no gradient (one wrong entry scrambles every block).
The trick is that **decryption factors by row**: plaintext coordinate `i` is a *single* linear
form `p_i[j] = D[i] . c[j] (mod 26)` over ciphertext blocks `j`. So a candidate row is just an
`n`-covector — for 3x3 only `26**3 = 17576` points, **1471 after quotienting by the invertible
scalar** that leaves the recovered stream a mono-substitution of itself. Every row is enumerable,
and *this is where the gradient lives*: a partially-correct key shows up as one or two
high-scoring rows.

**Score a row by the shape of its decimated stream.** `D[i] . c` is one plaintext coordinate out
of `n` — it has English single-letter statistics but never reads contiguously. Score it against
English with a statistic invariant to the two nuisance freedoms:
- the **scalar** (the row is only defined up to an invertible multiple; `u.p` is a bin-permutation
  of `p`) — minimise over the 12 units. A chi-square against the *asymmetric* English profile is
  NOT scalar-invariant, so the true scale is exactly what wins;
- an **additive** offset per block-class — covers a plain Hill (offset 0) and the affine /
  periodic-additive generalisation (a keyword added before the Hill step; a per-class shift of
  period `q`) — minimise chi-square over the 26 shifts within each class.

**Then assemble and let quadgrams pin the rest.** Rank rows by that chi-square, take the strongest
few dozen, and for every invertible triple search the `n!` coordinate assignment and the per-row
scalars by quadgram of the *interleaved* whole text (monogram fixes them only weakly at short
length; quadgrams see across the scaling). The genuine matrix wins by a wide margin.

```python
from buttcrack.ciphers._hill_recover import recover
recs = recover(ct, scorer, alphabet="KRYPTOS", q_values=(1, 2), pair_brute=True)
# recs[0].decrypt_matrix / .offsets / .plaintext ; or just `butt crack --cipher hill`
```

- **`Hill.crack`** now runs this for 3x3 (and still brutes 2x2), returning a plain round-trippable
  key for a pure Hill over A-Z. Pass `alphabet=` / `q_values=` / `pair_brute=` via opts for keyed
  alphabets and affine schedules.
- **Power / length.** Ranking a row needs its decimated stream to look English, which takes
  samples: reliable from ~60 trigraph blocks (~180 letters). On shorter text one or two true rows
  rank as **monogram outliers** — `pair_brute` scans every row for a third given two strong seed
  rows, rescuing the *single*-outlier case (~50 blocks); a two-outlier case at very short length is
  below the blind floor (validate on a synthetic of the target length, §11).

### 16a. When you have a crib: Hill is linear, so a little known plaintext breaks it outright

An `n x n` key needs only `n` independent plaintext/ciphertext blocks. The one catch is that
`Z_26` is a **ring**: Gaussian elimination divides, and 2 and 13 are zero-divisors. Solve mod 2 and
mod 13 (both fields) and recombine by CRT; a rank-deficient mod-2 system means the crib is short
enough to leave the key underdetermined, so **enumerate the whole solution set** and pick the key
that decodes the rest as English.

```python
from buttcrack.hill_kpa import solve_mod26, recover_matrix, recover_affine
recover_matrix(known_pt, ct, n=3, alphabet="STD", offset=0)   # crib may start mid-message
recover_affine(known_pt, ct, n=3, q=2, alphabet="KRYPTOS")     # matrix + period-q additive
```

## 17. The coset-preserving transposition: an honest, provable blind wall

§9(b) says a transposition under a periodic substitution is peelable because undoing the right
order makes the per-column IoC spike **reappear** (`reveal_score` has a gradient before you solve
the sub). There is one shape where that lever fails *by construction*, and recognising it saves you
from an unwinnable search: a **coset-preserving** transposition — one that permutes letters only
*within* each residue class mod the substitution period `p` (each mod-`p` column is scrambled
internally, never across columns).

- It leaves **every coset's multiset unchanged**, so **coset-IoC is invariant under it** — the raw
  spectrum already shows the period, and no un-winding changes it. `reveal_score` is flat across all
  candidate orders: there is nothing to reappear.
- It kills **kappa(p)** (the within-column *adjacency* is destroyed) while preserving the coset
  distributions — the fingerprint of "peaked in multiset but order-scrambled".
- Consequence: at a single message length the winding is **not blind-detectable** and the
  per-coset key is **not blind-recoverable** — the space of within-coset permutations is `(n/p)!`
  per coset and no statistic ranks it. This is a genuine information wall, not a missing solver.

**Two things follow, both now tooled.**

1. **Calibrate coset statistics with the RIGHT null.** A coset-IoC "spike" must be compared to a
   null that *also* preserves the cosets — otherwise a plain letter-shuffle null (which destroys the
   cosets) is trivially beaten and manufactures a false positive. Use
   `windings.coset_preserving_shuffle(ct, p, rng=…)` (shuffles only within each residue class) as the
   null twin; `windings` also generates the coset-preserving permutations themselves (affine / fold /
   faro per class) and triangular (`T_n`) reads for the winding search.
2. **Past the wall, you need external information, not more search.** A crib (§4b/§16a), or — the
   high-value move in a chain — the **sibling**: if two unsolved messages share the construction, the
   pair carries roughly twice the data of either alone. `butt compare ct_a ct_b` tests exactly that
   (sorted-frequency-profile distance, a shared period/kappa signature — including the tell of *one
   wound, one flat* = same substitution, different winding — and a two-ciphertext additive
   superimposition), so you learn whether to attack them jointly before you invest.

## 18. Look-elsewhere: a period SCAN needs the family null too (`butt stats --family`)

§10 calibrates a *searched transposition* against the shuffled search. The identical bias hits a
**period scan**: `calibrated_periods` / `kappa_spectrum` report a *per-period* z, but you keep the
best of 15–50 periods, so the maximum is selection-inflated. A per-period `z ≈ +3` on a short
message is routinely multiplicity noise. `butt stats --family` (`analysis.period_family_significance`)
reports the honest number: the strongest period's calibrated z vs the distribution of the **max
calibrated z over the whole grid** on shuffles of the same letters. Believe a period only when it
clears that family null (`beats_null_max` / small `family_p`), not merely on a high per-period z —
random text will hand you a period at `z ≈ +2.5` that dies at `family_p ≈ 0.3`.

## 19. Non-prose payloads: the right decrypt can score as gibberish

A quadgram (or word-coverage) gate assumes the plaintext is **prose**. A message that is a **route,
a coordinate list, spelled numbers, or a table** is real language token-by-token but scores in the
English "ghost band" — so a correct decryption can be *rejected* by an English-only objective, and a
whole search silently discards the answer. Two levers: pass a **custom scorer** to the crib/Hill
paths (`crib_drag(..., scorer=…)`) that rewards the payload's structure, and screen surviving
candidates with `butt nonprose` (a route/instruction genre model vs a prose model, anchor-normalized)
— `leans_nonprose` flags a candidate that reads as directions/coordinates even though the prose score
looks weak. When the aligned-coset modal is well above prose's ~0.12, suspect a structured payload
and stop trusting the English gate.
