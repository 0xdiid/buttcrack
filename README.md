# buttcrack

![CI](https://github.com/0xdiid/buttcrack/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Ciphers](https://img.shields.io/badge/ciphers-78-brightgreen)

A command-line classical-cipher cracker that **AI agents can drive easily** — the
agent-friendly cousin of [CryptoCrack](https://sites.google.com/site/cryptocrackprogram/home).
The CLI is `butt`.

Classical-cipher solving has historically lived in GUI tools (CryptoCrack,
AZdecrypt) or libraries with no command-line surface. `butt` is built the other
way around: every command is non-interactive, deterministic, and emits
**machine-readable JSON**, so an agent can pipe ciphertext in and parse structured
results out — recovered key, plaintext, fitness score, calibrated confidence, and
ranked alternatives.

> Status: all 64 CryptoCrack cipher types are implemented and validated against
> published vectors, plus 14 more closing the gaps against dCode and CrypTool 2
> (78 total) — including Enigma M3, the M-94 Jefferson cylinder and Chaocipher.
> The engine, scoring, and agent contract are stable. Still pre-1.0.
>
> See [`docs/gap-analysis.md`](docs/gap-analysis.md) for the capability audit against
> dCode and CrypTool 2, including what is deliberately *not* implemented and why.

## Install

Requires Python 3.10+. No runtime dependencies.

```bash
uv venv && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
butt --version
```

## Quick start

```bash
# Crack a Caesar shift, human-readable
butt crack caesar "Wkh txlfn eurzq ira"

# Same thing, JSON for an agent
butt crack caesar "Wkh txlfn eurzq ira" --json

# Don't know the cipher? Identify + crack across everything:
echo "Gvbtm migf lsxretq..." | butt auto --json

# Encode / decode with a known key
butt encode vigenere "attack at dawn" --key LEMON
butt decode vigenere "lxfopv ef rnhr" --key LEMON
```

### JSON contract

Every `encode`/`decode`/`crack`/`auto` emits the same top-level shape:

```json
{
  "ok": true,
  "schema_version": 3,
  "operation": "crack",
  "verdict": "solved",
  "cipher": "vigenere",
  "ciphertext": "...",
  "plaintext": "the quick brown fox ...",
  "key": "LEMON",
  "score": -512.34,
  "confidence": 0.97,
  "word_coverage": 0.71,
  "margin": 0.41,
  "runtime_ms": 8.21,
  "candidate_count": 5,
  "candidates": [ { "plaintext": "...", "cipher": "...", "key": "...",
                    "score": -512.34, "confidence": 0.97, "word_coverage": 0.71, "meta": {} } ]
}
```

**Trust the `verdict`, not `ok`.** `ok` only means "a candidate was produced"
(brute-force ciphers always produce one). The real signal is:

- `verdict` — `solved` · `likely` · `ambiguous` · `unlikely` · `no-candidates` (and `n/a` for encode/decode, which don't crack)
- `schema_version` — bumped on any envelope change; `butt schema --json` returns the full manifest
- `confidence` — calibrated 0..1, **sample-size aware** (short/noisy input can't score high),
  and deflated by `word_coverage` so an overfit "salad" can't read as solved
- `word_coverage` — fraction of the plaintext tiled by real ≥5-letter words: a
  language-fit signal the n-gram score can't see (~0.5–0.8 for real English, ~0 for
  quadgram salad). `null` when not computed (non-English `--lang`, or text too short).
  Lets you tell *"low confidence: too short"* from *"low confidence: scores like
  English but isn't language"*
- `margin` — confidence gap to the runner-up; a small margin means two ciphers fit
  about equally. When the result is `ambiguous`/`unlikely` and the rivals are close,
  the rival cipher is named in `ambiguous_with`.
- `crib_confirmed` — top-level boolean, present and `true` only when `--crib` was
  given and the best candidate's plaintext contains it (a cipher-agnostic confirmation
  that dominates the confidence-based ranking). `crack`/`auto` only.

`auto` additionally includes an `identify` block (index of coincidence, per-letter
chi-squared (`chi_squared_per_letter` — low means letters still match English even
when order is scrambled, the objective monoalphabetic-vs-transposition test), ranked
likely families, a `reliable` flag, a **calibrated period spectrum** (`periodic_ioc`),
and a plain-language **`diagnosis`**) as a routing hint. The
diagnosis is deliberately honest about the hard cases — e.g. *"significant period
40, ~7 letters/column (long key); recovery uncertain, try a crib"*, or, when
nothing reads, *"no significant period — don't assume running-key, try a crib-drag"*.

`butt stats` also reports **IoC drift** (per-segment index-of-coincidence and its
z-scored slope): a monotonic decay along the message marks an *evolving / non-stationary
keystream* (progressive-key, autokey, chain-addition, dynamic alphabet, or OTP-grade)
whose period can't be recovered. `identify` routes on this, diagnosing a **non-stationary
keystream** and pointing at a crib — and a real decay *rules out* the transposed-periodic
family (which has flat, stationary IoC). See the [cryptanalysis tips](docs/cryptanalysis-tips.md)
§9 for the full periodic-OUTER / periodic-under-transposition / non-stationary triage.

**`butt diagnose [text]`** rolls the whole triage into one report: it runs the period
spectrum, lag autocorrelation, IoC-decay, evolving-keystream fingerprint and the N/lcm
crackability cliff, then prints a **structure-class verdict and the concrete `butt` commands
to try next** — the fastest way to answer "what kind of layered cipher is this and how do I
attack it". For the layered families themselves: `butt layered` (substitution OVER a
transposition), `butt transsub` (transposition OVER a substitution, incl. `--keyword-pairs`
for the double-columnar shape), and `butt crib --product/--keyed/--autokey` for
crib-anchored product / keyed-alphabet / autokey solvers. A field guide of hard-won
cryptanalysis lessons lives in [docs/cryptanalysis-tips.md](docs/cryptanalysis-tips.md).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | `crack`/`auto`: a trustworthy solve (`verdict` `solved` or `likely`); `encode`/`decode`/`list`/`identify`: success |
| `1` | Ran fine but produced nothing convincing (`ambiguous`/`unlikely`/`no-candidates`) |
| `2` | Bad input — unknown cipher, invalid key, missing/unreadable text |

Errors never dump a traceback. With `--json` they return `{"ok": false, "error": "...",
"error_type": "..."}`; otherwise a clean `error: ...` line on stderr.

> Note: transposition ciphers (`railfence`, `columnar`) reorder letters and cannot
> preserve spacing, so their output is a clean uppercase letter stream.

## Commands

| Command | Purpose |
|---|---|
| `butt encode <cipher> --key K [text]` | Encrypt with a known key |
| `butt decode <cipher> --key K [text]` | Decrypt with a known key |
| `butt crack <cipher> [text]` | Crack a specific cipher, keyless (`--crib WORD` to confirm/rank) |
| `butt auto [text]` | Identify and crack across all ciphers, ranked (`--crib WORD` also unlocks the crib-anchored keyed-substitution-over-columnar crack) |
| `butt crib --crib WORD [text]` | Crib-drag a guessed word across the Vigenère family / running key |
| `butt keysource [ct] --corpus … / --compose A B / --decompose K` | Candidate keys from prior-solution texts (running key/acrostic/word), two-word composed keys, and a one-shot running-key screen against a target ciphertext |
| `butt hillkpa [ct] --crib WORD [-n N]` | Hill known-plaintext attack: recover the n×n key from a crib (keyed alphabet, CRT over Z26) |
| `butt validate --structure S [--self-check]` | Build a same-structure synthetic ciphertext (validate-on-synthetic discipline) and optionally prove `butt auto` recovers it |
| `butt transform [text] [--decimate P:O]` | Un-wrap format tricks: reverse, base64/hex/A1Z26, strip nulls |
| `butt pipeline [text] --step cipher:key …` | Chain decode (or `--encode`) steps for a layered cipher |
| `butt layered [text] --period P` | Crack a periodic (Quagmire) substitution **over** a columnar transposition; emits a per-column residual report for an agent when columns are too short to decide |
| `butt identify [text] [--types]` | Classify likely cipher family, or specific types (`--types`) |
| `butt diagnose [text]` | One-shot layered/composite structure triage → verdict + recommended attacks |
| `butt stats [text] [--contacts] [--significance] [--family]` | Frequency / IoC / per-letter chi-squared / digraph / Kasiski / period & repeated-bigram detectors (+ vowel-finder); `--family` adds the look-elsewhere-corrected significance of the strongest period |
| `butt compare [ct_a] --with ct_b` | Sibling-pair analysis: do two ciphertexts share a construction? (frequency profile, period/kappa signature, additive superimposition) |
| `butt nonprose [text]` | Flag a candidate that scores like English but reads as a route/coordinates/list (structured non-prose payload) |
| `butt keyword [text]` | Recover the keyword from a keyed alphabet or square |
| `butt words {match,pattern,anagram,ngram} <q>` | Dictionary search |
| `butt convert [text] --to {numbers,pairs,letters}` | A1Z26 conversion |
| `butt format [text]` | Regroup / strip / recase |
| `butt split [text]` | Split a file of multiple ciphers into entries |
| `butt solve --batch jobs.ndjson` | Batch mode: one JSON request per line, one result per line |
| `butt list [-v]` | List ciphers (`-v` shows each cipher's key format) |
| `butt help <cipher>` | A cipher's key format, a working example, and usage |
| `butt schema [--compact]` | Machine-readable capability manifest (commands, flags, envelope, verdict enum) |

### Discoverability (for agents)

Everything is introspectable from the CLI — no need to read source or guess:

```bash
butt schema --json          # capability manifest: commands, flags, result envelope, verdict enum, languages
butt list --json            # every cipher: name, aliases, needs_key, key_format, key_example, complexity
butt help gromark           # one cipher's key format + a working --key example
```

Input comes from a positional argument, `--file PATH`, or stdin (`-`).
Add `-j/--json` for JSON; `--compact` for single-line JSON.

Useful flags on `crack`/`auto`: `--top N`, `--seed S` (reproducible
hill-climbing), `--timeout SECONDS`, `--lang {english,french,german,spanish,italian}`,
`--ngrams {trigrams,quadgrams,quintgrams}` (fitness model; quintgrams is sharper but
needs a built table and falls back to quadgrams with a stderr note when none is
bundled), `--wordlist FILE` (dictionary cracking), plus per-cipher tuning
(`--key-length`, `--restarts`, `--max-width`, `--max-rails`, `--blind`, …).

The **Quagmire family** (`quagmire1/2/3`) is cracked by a *keyword dictionary
attack* — far more reliable than blind hill-climbing at ACA message lengths. It
tries a built-in famous-keyword list (KRYPTOS first) automatically; pass
`--wordlist FILE` for your own candidate keyed-alphabet keywords:

```bash
butt crack quagmire3 <ct>                      # built-in keywords (KRYPTOS, …)
butt crack quagmire3 <ct> --wordlist names.txt # your candidate keywords
butt crack quagmire3 <ct> --blind              # opt-in blind keyed-alphabet anneal
```

`--blind` is an opt-in, best-effort fallback for a keyed alphabet that isn't a word:
it's off by default because an arbitrary keyed alphabet is an isolated optimum with
essentially no gradient (one swap from the answer scores like a random one), so the
anneal only has a chance on long texts (≥ ~200 letters) and usually the dictionary
attack or a crib is the real lever.

New to cracking an unknown ciphertext? See the
**[cryptanalysis tips & tricks](docs/cryptanalysis-tips.md)** playbook.

### Batch / NDJSON

```bash
printf '%s\n' \
  '{"id":1,"op":"auto","text":"Wkh txlfn eurzq ira"}' \
  '{"id":2,"op":"crack","cipher":"vigenere","text":"lxfopvefrnhr"}' \
  | butt solve
```

## Supported ciphers

**All 64 of [CryptoCrack](https://sites.google.com/site/cryptocrackprogram/home)'s cipher
types are implemented** (`butt list` shows the full set with per-cipher solver coverage).
Every cipher's encode/decode is validated against a published test vector.

- **Monoalphabetic**: caesar, rot13, atbash, affine, substitution (Aristocrat/Patristocrat),
  key-phrase, headline, numbered-key
- **Polyalphabetic**: vigenere, beaufort, variant-beaufort, gronsfeld, porta, portax, autokey,
  running-key, progressive-key, interrupted-key, quagmire 1–4, slidefair, gromark, periodic-gromark
- **Polygraphic / square**: playfair, seriated-playfair, four-square, two-square, tri-square,
  hill, bazeries
- **Fractionation**: bifid, cm-bifid, trifid, adfgx, adfgvx, digrafid
- **Morse / numeric**: fractionated-morse, morbit, pollux, tridigital, straddling-checkerboard,
  monome-dinome, nihilist-substitution, homophonic, grandpre, syllabary, compressocrat
- **Transposition**: railfence, redefence, columnar, incomplete-columnar, myszkowski, amsco,
  route, cadenus, swagman, grille, sequence-transposition, nihilist-transposition, nicodemus
- **Other**: bacon (biliteral), checkerboard, phillips, ragbaby, condi, null

Run `butt list` for names, aliases, and descriptions. Solver coverage varies by cipher
(full keyless crack / best-effort / decode-only) — see the parity matrix.

## How it works

- **Scoring** — candidate plaintexts are ranked by English n-gram log-probability
  (quadgrams by default). Confidence is self-calibrated against the loaded table,
  so it's comparable across ciphers, and then **deflated by long-word coverage**:
  a stochastic solver can manufacture text that scores like English on n-grams but
  is gibberish ("aristocrat salad"), so a candidate that can't be tiled into real
  ≥5-letter words is penalised — keeping an overfit from being reported as a solve
  or out-ranking a genuine decrypt in `auto`.
- **Crackers** — brute force where the keyspace is tiny (Caesar, affine, rail
  fence), index-of-coincidence + per-column chi-squared for Vigenère, quadgram
  hill-climbing with restarts for substitution, and a **keyword dictionary attack**
  for the Quagmire family (blind hill-climbing of a keyed alphabet doesn't converge
  at ACA lengths; recovering per-column shifts by quadgram against a candidate keyword
  does). For **columnar** transposition the column order is brute-forced up to width 8
  and recovered by **simulated annealing** on the n-gram score for wider keys
  (`--width`/`--max-width`). **ADFGX/ADFGVX** are cracked blind in two phases: peel
  the transposition by a mapping-independent **digraph index-of-coincidence**, then
  solve the recovered digraph stream as a simple substitution (the Polybius square).
  These layered/long-key solvers are best-effort and want long messages; see the
  [cryptanalysis tips](docs/cryptanalysis-tips.md) for the per-class scope and limits.
- **n-gram tables** — `src/buttcrack/data/english_*.txt` are generated from a
  public-domain corpus by `scripts/build_ngrams.py`. English ships **monogram→hexagram**
  tables; quadgram is the default fitness, but `--ngrams quintgrams`/`hexagrams` select the
  bundled higher orders — the sharpest real-English-vs-quadgram-salad separator on the hard
  keyless families (worth reaching for whenever a solve reads as gibberish that still scores
  well). Other languages ship up to quadgrams; build higher orders from your own corpus:

  ```bash
  python scripts/build_ngrams.py path/to/corpus/*.txt            # mono..quadgrams
  python scripts/build_ngrams.py path/to/corpus/*.txt --max-n 6  # + quint/hexagrams
  ```

## Development

```bash
uv pip install -e ".[dev]"
ruff check . && ruff format --check .   # lint + format
mypy                                    # type check
pytest                                  # tests
```

CI (GitHub Actions) runs all four across Python 3.10–3.12 plus a wheel
build/install smoke test on every push and PR.

## Roadmap

All 64 cipher types and the core analysis/utility tools are done. Genuinely
remaining work:

- **MCP server** exposing the same engine as typed tools for agents.
- **Blind layered auto-crack for the remaining super-enciphered classes** —
  homophonic/Z340-class, and keyed-substitution-over-transposition *without* a crib.
  (Shipped already: the base64→cipher pre-flight peel, the additive-substitution-over-
  transposition crack, and — with a crib — the keyed-substitution-over-columnar crack
  via `auto --crib`. See [cryptanalysis tips §8](docs/cryptanalysis-tips.md) for why
  the keyed/homophonic classes resist *blind* search.)
- **AI cipher-type identifier** to complement the statistical `identify --types`.
- **More languages** and foreign-character alphabets beyond the bundled five.
- Stronger keyless solvers for the `best-effort` ciphers (see the parity matrix).

Adding a cipher means implementing one `Cipher` subclass (`encode`/`decode`/
`crack`) with `key_format`/`key_example`, and registering it — the JSON, CLI,
batch, `auto`, `list`, and `help` plumbing come for free.

## License

MIT
