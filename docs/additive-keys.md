# Periodic additive keys: exact recovery and honest coverage

`butt additive` works with the declared model

```
C[i] = P[i] + K1[i % p1] + ... + Kr[i % pr] (mod 26)
```

All letters are indices in one shared alphabet, and each component starts at
phase zero. This covers repeating-key passes whose index arithmetic reduces to
this sum. It does not establish that a ciphertext uses this model. Different
alphabets between passes, transpositions, or nonlinear substitutions require a
different model. Unknown starting phases can be absorbed into arbitrary numeric
keys; recovering dictionary words assumes their unrotated, phase-zero spelling.

## Analyze periods

```sh
butt additive analyze --periods 2 3 6 8
```

This reports 19 raw key coordinates, 12 effective coordinates, seven redundant
coordinates, and an aggregate period bound of 24. The bound is the least common
multiple; particular keys can produce a shorter period. Shared factors and
repeated periods are supported. The effective dimension is the sum of Euler's
totient over the union of divisors of the component periods. Analysis does not
expand the aggregate cycle. The solver limits the sum of component periods to 512.

The Python helper `period_sets(rank, lo, hi, max_effective)` enumerates increasing,
distinct period tuples inclusively, allowing shared factors by default. Set
`coprime_only=True` only when that restriction is part of the hypothesis. The
helper caps enumeration at 100,000 tuples; it is not a repeated-period enumerator.
Repeated periods can be passed directly to the analyzer or solver.

## Solve cribs

```sh
butt additive crib LIPPSASVPH --periods 1 --fragment 0:HELLO
butt additive crib LIPPSASVPH --periods 1 --fragment 0:HE --fragment 8:LD
butt additive crib LIPPSASVPH --periods 1 --drag HELLO --max-placements 20
```

The first two examples determine `HELLOWORLD`. Offsets are zero-based positions
in the uppercase, letters-only ciphertext; spaces and punctuation do not occupy
positions. `--file path` or standard input can supply ciphertext. `--alphabet`
accepts a registered alphabet name or an explicit 26-letter permutation; the
default is `STANDARD`.

The solver uses independent linear systems modulo 2 and 13 and combines their
answers modulo 26. A plaintext letter is reported only if its pad coordinate is
determined in **both** systems. `?` in plaintext and `null` in the numeric keystream
mark uncertainty. `constraint_ranks` describes the supplied crib positions, which
may cover less than the full effective dimension. An inconsistent system reports
conflicts and does not expose a guessed plaintext. `representative_keys` gives
one numeric assignment when consistent; it is not a unique decomposition.

A floating crib tries each fitting offset up to `--max-placements`. Every compatible
placement is returned without language ranking. Compatibility alone is weak
support when the model has many free coordinates. `exhausted_placements` means
all offsets for this particular crib and model were considered; `capped` means
some were not. Crib results are conditional on the supplied text being correct.

## Recover dictionary components

```sh
printf 'E\nF\n' > words.txt
butt additive crib LIPPSASVPH --periods 1 --fragment 0:HELLO --wordlist words.txt
butt additive words --periods 1 --pad '4,4,?,4' --wordlist words.txt
```

The dictionary is a UTF-8 file with one word per line. Words are trimmed,
uppercased, deduplicated, and filtered to the requested component lengths and
alphabet. Non-ASCII words are rejected without transliteration. Short words are retained. The search indexes the final component by
its observed residue signature and enumerates the other components. Every
returned ordered tuple is checked against all known pad positions. Unknown pad
positions impose no constraint, so a partial pad can leave many candidates.

`--max-nodes` bounds dictionary ingestion, indexing, prefix enumeration, and
candidate checks; `--max-results` bounds output. The result includes filter and
work counts, bank completeness, the declared tuple count when known, and the
limit reason. `unique_in_bank` is true only after complete enumeration finds one
tuple. It never claims uniqueness outside the supplied bank. Repeated periods,
equivalent keys, and component ordering are preserved. A cap is conservatively
reported even if the result that reaches it happens to be the last possible one.

All additive commands emit JSON. Exit code 0 means the requested operation
completed (including a partial but consistent crib result); 1 reports a
contradiction or a cap, including a cap in attached dictionary recovery; 2 reports
invalid input. A floating-crib search with no compatible placements reports its
completed search in JSON with exit code 0.

## Audit the candidate universe

```sh
butt additive bank --wordlist words.txt --min-length 1 --max-length 12
```

This audit records all supplied entries, including exclusions and duplicates,
normalization, filter settings, counts, accepted candidates, and a SHA-256 digest.
It does not run a search. The digest covers rejected inputs too. Save the JSON
alongside results to make the chosen word universe reviewable. Audits contain the
supplied words: store sensitive inputs and their manifests outside public repos.

The library also provides `audit_period_tuples` and `SearchReceipt`:

```python
from buttcrack.search_coverage import audit_period_tuples, SearchReceipt

audit = audit_period_tuples([(2, 6), (3, 5)])
# Set evaluated only after your own search completes both accepted candidates.
receipt = SearchReceipt(
    audit=audit,
    status="exhausted_bank",
    evaluated=2,
    assumptions=("shared alphabet; phase-zero additive keys; specified crib",),
)
record = receipt.to_dict()
```

Receipts distinguish exhaustion, failed gates, timeouts, caps, interruptions,
candidates, statistical mismatches, and exact contradictions. Completion counts
must agree with the manifest. A proof reference for an exact contradiction is
required but is not verified by this bookkeeping API. Receipts do not execute
solvers or establish statistical power. Exhausting a finite bank cannot exclude
the whole cipher family. Use independent synthetic recovery and ambiguity tests
to establish what a particular search configuration can detect.
