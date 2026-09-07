# A reproducible cryptanalysis workflow

Use this workflow for a sustained investigation or any search whose failure will
influence the next experiment. The [tips](cryptanalysis-tips.md) describe attacks;
this guide describes how to decide what their results establish. Scale the records
to the work: a short experiment may need only a command and a few lines of notes.

This is a documentation convention, not an automated hypothesis manager. The
[candidate audit and receipt APIs](additive-keys.md#audit-the-candidate-universe)
provide some bookkeeping; maintaining dependencies and reopening conclusions
remains the investigator's responsibility.

## 1. State the hypothesis before choosing the search

Write the literal encryption equation or ordered operations. Specify the alphabet,
indexing convention, key phases, block alignment, padding, and normalization.
Separate observed facts, source statements, interpretations, and assumptions.
Preserve the exact source wording locally so interpretations cannot silently
become stronger constraints.

For example, a peak in a period scan can motivate a periodic-key hypothesis. It
does not establish that every layer has that period. An apparent thematic key can
motivate a word bank without requiring keys to be literal source-text substrings.

Keep a small set of plausible competing constructions. Include inexpensive simple
models and compositions of familiar operations; do not require complexity to
increase with each new message. Simplify the algebra where operations commute,
but retain factorized keys. Test noncommuting orders explicitly. Record numeric
key recovery separately from any proposed lexical derivation.

Before running, state what each possible outcome would change. Prefer an experiment
that distinguishes live hypotheses over another large search inside an assumption
that has never been tested.

## 2. Audit the domain before spending the budget

Record the candidate inputs and the actual generator output, including defaults,
normalization, filters, duplicate handling, and exclusion reasons. For a large
universe, preserve source inputs, generator code/version, hashes, counts, and a
way to replay membership without storing every Cartesian-product tuple.

Check representative cases through the same entry point used by the target run:

- Short words and minimum/maximum lengths, including values just outside bounds.
- Shared-factor and repeated periods, different alignments, and block boundaries.
- Equivalent keys and operation orders: distinguish mathematical equivalence from
  whether an ordered word sequence belongs to the bank.
- Any stopping, pruning, or top-candidate retention rule that can remove the truth.

Test membership and recovery separately. Deliberately inserting a known answer
into a final bank is a useful recognition test, but does not validate the generator
that normally supplies the bank. Confirm that seeded fixtures survive filtering
when claiming end-to-end pipeline coverage.

A bank audit proves what was supplied and retained, not that the vocabulary or
construction grammar is complete. Make restrictions visible in the eventual
conclusion. Avoid a cipher-family label for a much narrower parameterization.

## 3. Validate the instrument

Use fresh synthetic data with no private benchmark material. Separate four checks:

| Check | Procedure | What a pass supports |
|---|---|---|
| Arithmetic | Compare an independent implementation or small exhaustive oracle; re-encrypt recovered candidates | Correctness for the checked operations and conventions |
| Bank recognition | Intentionally include the truth and measure its rank | Recognition when that candidate is available |
| Blind recovery | Run from independently generated starts under the intended budget | Recovery on the stated fixtures and settings |
| Statistical calibration | Repeat the complete search and selection under a declared null | Comparison against that particular generator |

An encrypt/decrypt round-trip alone can hide a shared convention error. A high
score alone can hide an implausible plaintext. Use independent predictions or
constraints where available, and do not equate consistency with uniqueness.

Use distinct randomness for fixture keys, solver initialization, and null samples.
Record their seeds or generator states. Check actual start membership to catch
accidental inclusion of true keys. Intentional inclusion belongs in a bank
recognition test and must be labeled as such.

Vary texts, keys, lengths, period geometry, and relevant payload types. Report
per-case outcomes; do not let a pooled success rate conceal a failing subdomain.
Rotations or windows from one source text are not independent new plaintexts.
Reserve held-out fixtures when tuning budgets or scoring rules.

A failed gate leaves that search configuration unvalidated. Passing a few fixtures
does not establish uniform sensitivity across the family. Retain the same budget
and settings when interpreting a subsequent target run.

## 4. Run with recoverable state

Save the exact code revision (and any uncommitted patch), command, environment,
input hashes, manifest, budget, seeds, logs, and stopping reason. Track completed
candidate units separately from attempts, timeouts, and interrupted work. Record
checkpoint boundaries so resumed shards neither disappear nor count twice.

For nested searches, verify the implemented objective and compare rankings as the
inner budget increases. A deterministic local optimum is reproducible but need
not be correct. Give compared candidates comparable evaluation budgets.

Preserve useful partial results: residual ciphertext, transformation, key, score,
and rejection reason. Failure under one assumed inner model only rejects that
combination. Keep a bounded set of structurally diverse candidates and record the
retention rule, rather than silently deleting every result below a prose threshold.

If a process crashes, mark unfinished work interrupted. Inspect the failure before
resuming from a verified checkpoint. A crash or missing output is never an
exhausted search. Do not substitute guesses about the machine for evidence about
the cipher.

## 5. Record the outcome with its scope

Use separate statuses and keep the domain attached when summarizing:

| Outcome | Permitted interpretation |
|---|---|
| Exact contradiction | No solution under the stated equation, domain, and assumptions, supported by a checked proof |
| Exhausted bank | Every accepted member of the declared finite bank completed |
| Gate failed | This configuration did not meet the stated recovery criterion |
| Timed out / capped / interrupted | Some declared work remains unfinished |
| Statistical mismatch | The observed result is unusual under the specified generator and selection procedure |
| Candidate found | A compatible candidate was recovered; verification and ambiguity still need assessment |

Do not shorten these to an unqualified “ruled out” or “powered null.” A proof
reference in a receipt is not itself proof verification. An exhausted bank is not
family-wide exclusion. A successful recovery gate does not make a failed target
run an impossibility theorem.

For statistical results, preserve the null generator, sample count, search budget,
and selection procedure. Account for scanning multiple periods, keys, models, or
cribs. Zero observed exceedances is not zero population probability. Rarity under
one generator does not positively identify a different construction.

For crib recovery, label guessed text and positions as assumptions. Examine what
the fit predicts outside those positions, retain alternatives, and independently
re-encrypt. Do not count an imposed crib match as independent confirmation.

### Copyable experiment record

This is a suggested local record, not a schema consumed by `butt`. Use paths and
hashes pointing to your own artifacts; do not publish sensitive values.

```yaml
id: experiment-001
question: Can this declared search recover this class of additive keys?
hypothesis:
  equation: "C[i] = P[i] + A[i % a] + B[i % b] (mod 26)"
  alphabet: STANDARD
  phases: [0, 0]
  assumptions: [letters-only input, specified period bank]
  depends_on: [assumption-001]
domain:
  manifest: local/periods.json
  input_hash: "<hash>"
  filters: "<explicit settings, including defaults>"
validation:
  arithmetic: local/oracle-results.json
  bank_recognition: local/recognition-results.json
  blind_recovery: local/held-out-results.json
  start_membership_checked: false  # update after checking
run:
  code_revision: "<commit and patch reference>"
  command: "<exact command>"
  environment: local/environment.txt
  budget: "<limits>"
  random_streams: local/randomness.json
  checkpoint: local/checkpoint.json
  logs: local/run.log
outcome:
  status: "<one of the outcomes above>"
  completed: 0
  unfinished: "<count and reasons>"
  conclusion: "<claim restricted to this domain>"
  remaining_uncertainty: "<what was not tested or established>"
  next_discriminating_test: "<test that could change the decision>"
```

## 6. Review, correct, and hand off

Choose a review point before starting a batch: a time budget, a number of
experiments, or a failure that should trigger reconsideration. At that point ask:

- Which assumptions have never been directly tested?
- Which candidates were excluded before scoring, and why?
- Which negative conclusions rely on missing artifacts or failed gates?
- Does the next experiment distinguish models or repeat the same assumption?
- What would change the current interpretation, and have we tried to observe it?

Give claims stable identifiers and list their dependencies. If a premise or gate
is withdrawn, mark dependent conclusions **needs revalidation** and reopen the
research directions they closed. Preserve the old record with the correction;
do not silently overwrite it. Updating a caveat without updating research
priorities leaves the original error in control.

A handoff should contain the current hypotheses, evidence and limitations,
retracted claims, unexplored domains, reproducible commands, checkpoint state,
and the next discriminating experiment. Summaries must retain the restrictions
that made the original claims defensible. Count progress by uncertainty resolved,
not by trials accumulated across overlapping searches.

## 7. Keep private investigations out of reusable documentation

Public examples and tests should be independently generated synthetic cases.
Transfer general equations, algorithms, failure modes, and validation practices;
keep benchmark keys, plaintext, ciphertext, hints, chronology, and identifiable
parameter combinations in private artifacts. Audits can contain the full supplied
word bank, so their JSON is not automatically safe to publish.

Before publishing, review the added diff, fixtures, filenames, generated outputs,
and any packaged artifacts. Scan against known private identifiers and distinctive
text fragments, then inspect context manually. Automated matching can miss
paraphrases and combinations that identify a benchmark; a clean scan is evidence
from that check, not a guarantee of anonymity.
