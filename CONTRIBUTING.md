# Contributing

Thanks for your interest in buttcrack. It's an agent-first classical-cipher CLI;
the guiding principle is that **anything a user can do, an agent can discover and
do from the CLI** with stable, machine-readable output.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
```

## Checks (all must pass)

```bash
ruff check . && ruff format --check .   # lint + format
mypy                                    # type check
pytest -m "not slow" -n auto            # fast suite, parallel — ~1 s feedback loop
pytest -m slow                          # stochastic solver tests, serial (~10 min)
```

Run the slow tests **serially** (no `-n`): they're bounded by a wall-clock timeout,
so sharing CPU across xdist workers starves them and they flake. CI mirrors this —
fast tests parallel, slow tests serial — on Python 3.10–3.12 plus a wheel smoke test.

## Adding a cipher

1. Add `src/buttcrack/ciphers/<name>.py` with a `Cipher` subclass implementing
   `encode`, `decode`, and `crack`. Set `name`, `description`, `complexity`, and
   — importantly for discoverability — `key_format` and a working `key_example`.
   Reuse the shared modules where they fit: `_periodic` (Vigenère family),
   `squares` (Polybius), `morse`.
2. Register it in `src/buttcrack/ciphers/__init__.py` (`ALL_CIPHERS`). The JSON,
   CLI, batch, `auto`, `list`, `help`, and `schema` plumbing comes for free.
3. Add `tests/test_cc_<name>.py` with a **vector test validated against a cited,
   published source** (the bar for "validated"), a round-trip test, and — if the
   cracker reliably recovers — a `@pytest.mark.slow` crack test.

## Conventions

- `encode`/`decode` operate on a clean letter (or digit) stream; transposition
  ciphers must not preserve word spacing (it leaks plaintext word lengths).
- Keyless cracks that are ill-posed (yield confident garbage) should set
  `auto_crackable = False` so they don't pollute `auto`.
- Mark stochastic / brute-force crack tests `@pytest.mark.slow`.
- Keep the JSON envelope stable; bump `SCHEMA_VERSION` if it changes.
