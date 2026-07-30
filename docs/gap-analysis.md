# Capability gap analysis: `butt` vs dCode vs CrypTool 2

> **Status: all five ranked recommendations are implemented.** See the "Unreleased"
> section of [`CHANGELOG.md`](../CHANGELOG.md) for what shipped, and
> [What shipped](#what-shipped) at the foot of this document for the outcome of each
> item, including the ones that turned out differently than this audit predicted.

Snapshot date: 2026-07-30. Inventories taken from:

* `butt` — `registry.names()` (72 cipher types) + `butt --help` (37 subcommands), this tree.
* dCode — https://www.dcode.fr/tools-list#cryptography (full scrape, all sections).
* CrypTool 2 — `CrypPlugins/` directory listing on `CrypToolProject/CrypTool-2@main`
  (~240 plugin projects).

## 0. The three tools are not the same kind of thing

| | `butt` | dCode | CrypTool 2 |
|---|---|---|---|
| Shape | CLI, JSON out, agent-driven | web forms, one page per cipher | GUI dataflow workspace |
| Classical cipher coverage | 72 types, all with **solvers** | ~120 types, mostly **codecs** | ~60 classical, ~12 with analyzers |
| Modern crypto | none (by charter) | hashes, RSA, RC4, XOR | full: AES/DES/RSA/ECC/hashes/protocols |
| Symbol & fandom alphabets | none | ~180 | ~2 (`SymbolCipher`, `Transcriptor`) |
| Statistical cryptanalysis | deepest of the three | 7 basic tools | ~8 tools |
| Calibrated nulls / evidence discipline | yes (`evidence`, `power`, `validate`, `nulls`) | no | no |

So "what butt lacks" splits into four very different buckets. Only buckets A and D
are real capability holes; B is out of charter and C is mostly noise.

---

## A. Real gaps — machine & dynamic-alphabet ciphers

These are genuine classical ciphers with genuine cryptanalysis literature that
`butt` has **zero** implementation of (verified by grep: 0 hits in `src/`).

| Cipher | dCode | CT2 | CT2 has analyzer |
|---|---|---|---|
| Enigma | yes | `Enigma` | `EnigmaAnalyzer` |
| M-209 (Hagelin) | — | `M209`, `HagelinMachine` | `M209Analyzer` |
| M-94 / M-138 strip | Jefferson Wheel | `CylinderCipher`, `M138` | `M138Analyzer` |
| Chaocipher | yes | `Chaocipher` | — |
| Solitaire (Schneier) | yes | `Solitaire` | `SolitaireAnalyser` |
| SIGABA | — | `SIGABA` | — |
| Typex | — | `Typex` | — |
| Purple | — | `Purple` | — |
| Fialka | — | `Fialka` | — |
| Lorenz SZ42 | — | `LorenzSZ42` | — |
| Spanish Strip | — | `SpanishStripCipher` | — |
| Mexican Army cipher disk | yes | `MexicanArmyCipherDisk` | `MexicanArmyCipherDiskAnalyzer` |
| LC4 / ElsieFour | — | `ElsieFourCipher` | — |
| T-310 | — | `T310` | — |

Why this matters here: `butt`'s own `identify.py:134` already names Chaocipher as a
dynamic-alphabet failure mode it can *detect* but not *attack*. The rotor family is
the single largest structural class the tool cannot even encode, let alone solve.

Ranked by likely value to this codebase:

1. **Chaocipher** — pure-Python, tiny, dynamic alphabet, already named in `identify`.
2. **M-94 / M-138 strip cipher** — direct cousin of the strip machinery in
   `windings.py`; the "strip" abstraction is already half-built here.
3. **Enigma** — big, but `EnigmaAnalyzer`-style IoC/trigram hillclimb is a known recipe.
4. Hagelin/M-209, Solitaire — self-contained keystream generators, easy encode side.
5. SIGABA/Typex/Purple/Fialka/Lorenz — low value for CTF-style puzzles, skip.

## B. Non-gaps — out of charter

CT2's bulk (~150 of ~240 plugins) is modern crypto and protocol work: AES, DES,
Camellia, Blowfish, Speck, PRESENT, ChaCha, Salsa20, Trivium, Grain, RC2/RC4, SHA
family, Keccak, BLAKE, Whirlpool, RSA, Paillier, Cramer-Shoup, ECC, lattices,
BB84 quantum key distribution, oblivious transfer, zero-knowledge, blockchain,
visual cryptography, WEP attacks, padding-oracle attacks, differential cryptanalysis
(DCA*), plus the whole workspace/visualization/IO layer (`FileInput`, `Webcam`,
`NetworkSender`, `WorkspaceManager`, ~30 plugins).

dCode's ~180 symbol/fandom alphabets (Aurebesh, Hylian, Wingdings, Gravity Falls,
Genshin, Zodiac, Dorabella, Voynich, …) are transcription tables, not cryptanalysis.
They are lookup dictionaries with no solver surface.

Neither is worth porting. Two narrow exceptions are called out in D below.

## C. Small classical ciphers `butt` lacks

All confirmed absent by grep. Each is <100 lines to add; none unlocks new
cryptanalysis.

* **Alberti** (dCode) — the ur-polyalphabetic disk.
* **Bellaso** (dCode) — pre-Vigenère reciprocal tableau.
* **Trithemius / Ave Maria** (dCode) — `progressive-key` already covers the numeric case.
* **Rozier**, **Wolseley** (dCode) — French/Victorian minor variants.
* **Collon** (dCode, polygrammic/grid) — the only genuinely missing *grid* cipher.
* **Triliteral** (dCode) — base-3 Bacon relative.
* **Ubchi** (dCode + CT2 `Ubchi`) — double columnar with a fixed key rule.
* **Nomenclator**, **Arnold cipher** (dCode) — codebook-shaped, not solvable blind.
* **Book cipher** (dCode + CT2 `BookCipher`) — `running_key.py` and `keysources.py`
  cover the adjacent territory but there is no page/line/word-index codec.
* **Tap code** and standalone **Polybius** (dCode) — Polybius exists *embedded*
  (22 files reference it) but is not a first-class registry entry, so it is not
  reachable from `butt encode/decode/crack`.

Highest-value three: **Collon**, **Ubchi**, standalone **Polybius**.

## D. Real gap — the outer wrapper / transport layer

This is the most actionable hole. `butt transform` handles only reverse, base64,
hex, A1Z26 and `--decimate`. dCode has ~40 tools for the *encoding shell* a puzzle
wraps its cipher in, and this tool has none of them:

1. **XOR cipher** (dCode + CT2 `XOR`) — repeating-key XOR with keylength search.
   Zero hits in `src/`; a classic first layer.
2. **Base26 / Base36 / Base37 / Base32 / ROT47 / ROT8000 / Unicode shift**.
3. **Keyboard-geometry codes** — keyboard shift, keyboard coordinates, LSPK90,
   T9 / multi-tap / phone keypad, DTMF.
4. **Wordlist encodings** — NATO phonetic, PGP word list, periodic table, prime
   multiplication, Navajo code.
5. **Symbol families worth having as tables** (the two exceptions from B):
   Braille, Pigpen/Masonic, Semaphore, Tap code, Morse variants (Wabun, Pollux ✓,
   Morbit ✓) — these show up constantly in real CTF chains.

The shape that fits this codebase is not 40 new ciphers: it is a **wrapper-detection
pass** in `transform` that tries the alphabet-preserving decodes and reports which
ones yield a letter distribution the downstream `identify`/`stats` can work on.
`pipeline` already exists to chain the result.

## E. Analysis tooling — `butt` is ahead, with two named exceptions

CT2's analysis plugins map cleanly onto existing `butt` surface:

| CT2 | `butt` equivalent |
|---|---|
| `FrequencyTest` | `stats` |
| `KasiskiTest` | `stats` (Kasiski) |
| `FriedmanTest` | `analysis.friedman_period_estimate`, `compare` kappa |
| `AutocorrelationFunction` | `diagnose` lag scan, `cipher_id` max-kappa |
| `WordPatterns` | `words --pattern` |
| `Dictionary` | `words` |
| `CostFunction` | `scoring` (n-gram to hexagram, entropy-normalized) |
| `SATSolver` | `crib_csp` (CP-SAT) |
| `KeySearcher` | `crack` / `auto` |
| `HomophonicSubstitutionAnalyzer` | `homophonic` |
| `TranspositionAnalyser` | `anagram`, `joint` |
| `AnalysisMonoalphabeticSubstitution` | `substitution` crack |

And `butt` has a large tier neither competitor has at all: `evidence`, `power`,
`nulls`, `validate` (plant gating), `contamination`, `census`, `landscape`,
`compare`, `nonprose`, `climbgate`, `relation`, `channel`, the `sub*` layered
solvers, `crib_algebra`/`crib_anchor`/`crib_csp`, `gpu_search`.

Two things CT2 exposes that `butt` does not surface directly:

* **`AutocorrelationFunction` / `FriedmanTest` as first-class commands.** Both
  computations exist inside `diagnose`/`compare`/`analysis` but neither is a
  documented `stats` flag. Cheap to expose.
* **`Transcriptor`** — a general "map arbitrary symbol set → letters" utility.
  This is the generic form of the whole dCode symbol-alphabet section, and would
  cover bucket D.5 in one component rather than 180.

---

## Ranked recommendations

1. **Wrapper layer in `transform`** (bucket D) — XOR with keylength search, the
   base-N family, and a symbol `Transcriptor`. Biggest coverage-per-line, and it
   feeds `identify`/`pipeline` which already exist.
2. **Chaocipher + M-94/M-138 strip** (bucket A) — the two rotor-family members
   closest to machinery already in this tree.
3. **Expose `autocorrelation` and `friedman` as `stats` flags** (bucket E) — the
   math is already written; this is a CLI-surface change only.
4. **Collon, Ubchi, standalone Polybius** (bucket C) — completes the grid and
   double-transposition families.
5. **Enigma + hillclimb analyzer** (bucket A) — real work (days, not hours); only
   worth it if a target actually smells like a rotor machine.

Explicitly not recommended: modern crypto, protocol plugins, fandom alphabets,
GUI/workspace concepts.

---

## What shipped

All five items are done (78 cipher types, up from 72). Three of the audit's own
predictions were wrong, and those are the entries worth reading.

| # | Item | Outcome |
|---|---|---|
| 1 | Wrapper layer | `buttcrack.wrappers` + five new `transform` flags |
| 2 | Chaocipher + M-94/M-138 | Both shipped; M-94 solves a full 25-disk order in <1s |
| 3 | `stats` autocorrelation/Friedman | Shipped — and both statistics were **wrong** as exposed |
| 4 | Collon, Ubchi, Polybius | All three shipped |
| 5 | Enigma + analyzer | Shipped; 5/6 on the plant gate |

### Where the audit was wrong

**"#3 is a CLI-surface change only, ~30 minutes."** It was not. Exposing the two
statistics revealed that neither was fit to report:

* The kappa spectrum scores each lag against the 1/26 random floor. Monoalphabetic and
  transposed text coincide at *plaintext* rate at every lag, so every lag cleared the
  floor and a harmonic search always found "a period". 4/4 false positives on ciphers
  with no period at all.
* Ranking a period by its own lag is backwards. English is anti-correlated at distance
  2-3, so a genuine period-2 key reads z < 0 at lag 2 while lags 4, 6, 8 spike. The
  first plant gate scored 3/12; ranking by harmonic family instead took it to 18/19.

The lesson generalises: a statistic buried inside a larger report is not thereby
validated. Both of these had been feeding `diagnose` and `cipher_id` all along.

**"Collon is a small cipher, <100 lines, unlocks no new cryptanalysis."** The obvious
implementation — anneal the 25-cell square, as Bifid does — scored 0/5 in 45 seconds
per attempt. Collon ciphertext can only contain ten distinct letters (five row labels,
five column labels), which makes each label pair a monoalphabetic symbol: 5/5 in 1.7s.
The same observation applies to Polybius. Two of the three "small" ciphers turned out
to be *solvable outright* rather than searchable.

**"Enigma: days, not hours; only worth it if a target smells like a rotor."** The full
60-rotor-order, 17576-position sweep runs in ~2.5 minutes in pure Python — roughly
16,000 trial decrypts per second. The real cost was not the machine but calibration:
the phase-1 IoC ranking is a whole-message statistic, and the initial 120-letter probe
put the true setting 1070th; at full length it ranks 1st.

### Still open

* **Enigma, unknown rotor order *and* non-trivial ring settings simultaneously** —
  partially closed. Each alone is solved by the default. Together, the default fixes the
  right-hand ring at A in phase 1, and because that ring decides *when* the middle rotor
  turns over, a measured plant put the true setting 128720th of 1054560 by IoC — past
  any shortlist. `ring_sweep=True` adds the ring to phase 1 (26x cost) and recovers the
  case exactly (`IV I V/B/AGH/QMT`) once the rotor order is narrowed. Across all 60
  orders at the default `keep=20` it still lands one turnover-step off, and the scorer
  prefers the true key (-1292.8 vs -1372.0) when offered it — so what remains is
  shortlist *recall*, not scoring, and `keep` is the lever.
* **The probe-length lever is narrower than it looks.** With rings AAA a 60-letter probe
  still solves, because every start position is equally affected. It is a non-trivial
  ring that breaks a short probe: the turnover error only shows up in the letters after
  the turnover, which a short probe never reaches.
* **XOR keys need ~25 ciphertext bytes per key byte.** A 20-byte key wants ~500 bytes.
  Below that it returns a key that is *mostly* right, quietly.
* Bucket C leftovers: Alberti, Bellaso, Rozier, Wolseley, triliteral, nomenclator, book
  cipher.
* Bucket A leftovers: M-209/Hagelin, Solitaire, SIGABA, Typex, Purple, Fialka, Lorenz,
  Spanish Strip, LC4, T-310 — all low value for puzzle work.
* Deliberately not implemented, with reasons, in the `buttcrack.wrappers` docstring:
  ROT8000, PGP word list, periodic table, DTMF, semaphore, pigpen.
