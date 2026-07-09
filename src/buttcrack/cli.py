"""The ``butt`` command-line interface.

JSON-first: pass ``--json`` (or pipe stdout) to get a stable machine-readable
schema on every command. Human-readable tables are the default on a terminal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from . import (
    __version__,
    analysis,
    cipher_id,
    engine,
    keyfinder,
    ngram_relation,
    registry,
    splitter,
    textops,
    words,
)
from .identify import identify as identify_text
from .result import SCHEMA_VERSION, VERDICT_VALUES, CrackResult
from .scoring import LANGUAGES, ngram_table_available
from .text import only_letters

#: selectable n-gram fitness models for `crack`/`auto` (--ngrams). quintgrams and
#: hexagrams ARE bundled (english_*.txt) — the higher orders sharpen real-English vs
#: quadgram-salad separation, the single most useful knob on the hard keyless families.
NGRAM_MODELS = ("trigrams", "quadgrams", "quintgrams", "hexagrams")


class InputError(ValueError):
    """Raised for bad user input (missing text, unreadable file); becomes exit 2."""


# --------------------------------------------------------------------------- IO
def _resolve_text(args) -> str:
    """Get input text from the positional arg, --file, or stdin."""
    if getattr(args, "text", None) not in (None, "-"):
        return args.text
    if getattr(args, "file", None):
        with open(args.file, encoding="utf-8") as fh:
            return fh.read()
    if args.text == "-" or not sys.stdin.isatty():
        return sys.stdin.read()
    raise InputError("no input text (pass text, --file PATH, or pipe via stdin)")


def _recovered_flag(result: dict) -> str:
    """Trailing annotation for solver text output when the decode failed the English gate."""
    if result.get("recovered", True):
        return ""
    return f"  [NOT RECOVERED — likely not English, word_coverage={result.get('word_coverage')}]"


def _emit(result: CrackResult, args) -> None:
    top = getattr(args, "top", None)
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(result.to_dict(top=top), ensure_ascii=False, indent=indent))
    else:
        _render_human(result, top)


def _render_human(result: CrackResult, top: int | None) -> None:
    if result.operation in ("encode", "decode"):
        best = result.best()
        print(best.plaintext if best else "")
        return

    ranked = result.sorted_candidates()
    if top is not None:
        ranked = ranked[:top]
    if not ranked:
        print("no candidates found", file=sys.stderr)
        if result.notes:
            for n in result.notes:
                print(f"  note: {n}", file=sys.stderr)
        return

    best = ranked[0]
    print(
        f"verdict: {result.verdict()}  ([{best.cipher}] key={best.key!r}  "
        f"confidence={best.confidence:.2f}  margin={result.margin():.2f})"
    )
    print(f"  {best.plaintext}")
    if len(ranked) > 1:
        print("\nother candidates:")
        for c in ranked[1:]:
            preview = c.plaintext[:70].replace("\n", " ")
            print(f"  {c.confidence:.2f} [{c.cipher:<12}] key={str(c.key)[:24]:<24} {preview}")
    if result.identify:
        fams = ", ".join(f"{f['family']}={f['weight']}" for f in result.identify["likely_families"])
        print(f"\nidentify: IoC={result.identify['index_of_coincidence']}  families: {fams}")
        if result.identify.get("diagnosis"):
            print(f"  diagnosis: {result.identify['diagnosis']}")


# ----------------------------------------------------------------- subcommands
def _crack_opts(args) -> dict:
    """Collect cipher-specific tuning flags that were actually provided."""
    mapping = {
        "key_length": args.key_length,
        "max_key_length": args.max_key_length,
        "restarts": args.restarts,
        "width": args.width,
        "max_width": args.max_width,
        "max_rails": args.max_rails,
        "wordlist": getattr(args, "wordlist", None),
        "blind": getattr(args, "blind", False) or None,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _cmd_encode(args) -> int:
    text = _resolve_text(args)
    _emit(engine.encode(args.cipher, text, args.key), args)
    return 0


def _cmd_decode(args) -> int:
    text = _resolve_text(args)
    _emit(engine.decode(args.cipher, text, args.key), args)
    return 0


def _exit_code(result: CrackResult) -> int:
    """0 = a trustworthy solve (solved/likely), 1 = ran but nothing convincing."""
    return 0 if result.verdict() in ("solved", "likely") else 1


def _warn_ngram_fallback(args) -> None:
    """Note on stderr when a requested n-gram model isn't built (we use quadgrams)."""
    ngrams = getattr(args, "ngrams", "quadgrams")
    if ngrams != "quadgrams" and not ngram_table_available(ngrams, args.lang):
        print(
            f"note: no {args.lang} {ngrams} table bundled; scoring with quadgrams "
            f"(build one with scripts/build_ngrams.py --max-n 5)",
            file=sys.stderr,
        )


def _cmd_crack(args) -> int:
    text = _resolve_text(args)
    _warn_ngram_fallback(args)
    result = engine.crack(
        args.cipher,
        text,
        top=args.top,
        seed=args.seed,
        timeout=args.timeout,
        lang=args.lang,
        crib=getattr(args, "crib", None),
        ngrams=args.ngrams,
        **_crack_opts(args),
    )
    _emit(result, args)
    return _exit_code(result)


def _cmd_auto(args) -> int:
    text = _resolve_text(args)
    _warn_ngram_fallback(args)
    ciphers = [c.strip() for c in args.ciphers.split(",")] if args.ciphers else None
    result = engine.auto(
        text,
        top=args.top,
        seed=args.seed,
        ciphers=ciphers,
        per_cipher_timeout=args.timeout,
        lang=args.lang,
        crib=getattr(args, "crib", None),
        ngrams=args.ngrams,
    )
    _emit(result, args)
    return _exit_code(result)


def _cmd_crib(args) -> int:
    from . import crib as crib_mod

    text = _resolve_text(args)
    indent = None if getattr(args, "compact", False) else 2
    if getattr(args, "product", False):
        if args.pos is not None and args.p and args.q:
            res = crib_mod.product_crib_solve(
                text, args.crib, args.pos, args.p, args.q, alphabet=args.alphabet
            )
            out = {"ok": True, "operation": "crib-product", "result": res}
        else:
            pr = args.p_range or [5, 20]
            qr = args.q_range or [5, 21]
            cands = crib_mod.product_crib_sweep(
                text,
                [args.crib],
                p_range=range(pr[0], pr[1] + 1),
                q_range=range(qr[0], qr[1] + 1),
                alphabet=args.alphabet,
            )
            out = {"ok": True, "operation": "crib-product-sweep", "candidates": cands}
        print(json.dumps(out, ensure_ascii=False, indent=indent))
        return 0
    if getattr(args, "autokey", False):
        cands = crib_mod.autokey_crib_unzip(text, [args.crib])
        print(
            json.dumps(
                {"ok": True, "operation": "crib-autokey", "candidates": cands},
                ensure_ascii=False,
                indent=indent,
            )
        )
        return 0
    if getattr(args, "keyed", False):
        drag = crib_mod.keyed_alphabet_crib_drag(text, args.crib, alphabet=args.alphabet)
        print(
            json.dumps(
                {"ok": True, "operation": "crib-keyed", "placements": drag},
                ensure_ascii=False,
                indent=indent,
            )
        )
        return 0
    if getattr(args, "inner_columnar", False):
        from . import cribbing
        from .text import only_letters

        ct = only_letters(text)
        n = len(ct)
        widths = args.widths or [w for w in range(2, 33) if n % w == 0]
        pr = args.periods or [9, 16]
        sol = cribbing.solve(
            ct,
            only_letters(args.crib),
            widths=widths,
            periods=range(pr[0], pr[1] + 1),
        )
        if args.json or args.compact:
            print(
                json.dumps(
                    {"ok": True, "operation": "crib-inner-columnar", "result": sol},
                    ensure_ascii=False,
                    indent=indent,
                )
            )
        elif sol is None:
            print("crib-inner-columnar: no consistent solution (crib/widths/periods may be wrong)")
        else:
            print(
                f"[crib-inner-columnar w{sol['width']} p{sol['period']} "
                f"{sol['alphabet']}/{sol['variant']}] score={sol['score']:.3f}"
            )
            print(sol["plaintext"])
        return 0
    drag = crib_mod.crib_drag(text, args.crib, top=args.top)
    if args.json or args.compact:
        out = {"ok": True, "operation": "crib", "crib": args.crib.upper(), "placements": drag}
        print(json.dumps(out, ensure_ascii=False, indent=indent))
        return 0
    print(f"crib-drag for {args.crib.upper()!r} (high score = implied key reads as language):")
    for cipher, places in drag.items():
        print(f"\n{cipher}:")
        for p in places:
            print(f"  pos {p['position']:>3}  key={p['key_fragment']:<14} score={p['score']:.1f}")
    return 0


def _parse_period_range(spec: str, default=range(6, 46)) -> range:
    try:
        lo, _, hi = str(spec).partition("-")
        return range(int(lo), int(hi) + 1)
    except (ValueError, AttributeError):
        return default


def _parse_orders(spec: str) -> list[list[int]]:
    """Parse --orders SPEC: ';'-separated index lists, @file, or '-' for stdin."""
    if spec == "-":
        raw = sys.stdin.read()
    elif spec.startswith("@"):
        with open(spec[1:]) as fh:
            raw = fh.read()
    else:
        raw = spec
    raw = raw.replace("\n", ";")
    orders = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        orders.append([int(x) for x in chunk.replace(",", " ").split()])
    return orders


def _cmd_transsub(args) -> int:
    from .scoring import resolve_scorer
    from .transsub import (
        crack_columnar_reveal_enum,
        crack_double_columnar_keywords,
        crack_transposition_over_sub,
        reveal_spectrum,
        sweep_known_alphabet,
    )

    text = _resolve_text(args)
    scorer = resolve_scorer(getattr(args, "ngrams", "quadgrams"), args.lang)
    rng = random.Random(args.seed) if args.seed is not None else None
    unit = getattr(args, "unit", 1)
    indent = None if getattr(args, "compact", False) else 2

    # --- reveal-tool modes (expose the composable block-transposition reveal primitives on the
    # CLI) ---
    if getattr(args, "spectrum", False):
        units = (1, unit) if unit != 1 else (1, 3)
        spec = reveal_spectrum(
            text,
            widths=range(args.min_width, args.max_width + 1),
            periods=_parse_period_range(args.periods),
            units=tuple(dict.fromkeys(units)),
        )
        print(json.dumps(spec, ensure_ascii=False, indent=indent))
        return 0 if (spec.get("best") or {}).get("verdict") == "beats null" else 1
    if getattr(args, "orders", None):
        orders = _parse_orders(args.orders)
        if not orders:
            raise InputError("--orders given but no valid read-orders parsed")
        res = sweep_known_alphabet(
            text,
            orders,
            alphabet=args.alphabet,
            unit=unit,
            periods=_parse_period_range(args.periods),
        )
        for c in res["candidates"]:
            c["generator"] = keyfinder.describe_permutation(c["order"])
        print(json.dumps(res, ensure_ascii=False, indent=indent))
        return 0 if any(c["recovered"] for c in res["candidates"]) else 1
    if getattr(args, "enum", False):
        res = crack_columnar_reveal_enum(
            text,
            scorer,
            widths=range(args.min_width, args.max_width + 1),
            alphabet=args.alphabet,
            unit=unit,
            rng=rng,
        )
        if res.get("structure", {}).get("columnar_order") is not None:
            res["structure"]["generator"] = keyfinder.describe_permutation(
                res["structure"]["columnar_order"]
            )
        print(json.dumps(res, ensure_ascii=False, indent=indent))
        return 0 if res.get("recovered") else 1

    keywords = None
    if args.wordlist:
        with open(args.wordlist) as fh:
            keywords = [line.strip() for line in fh if line.strip()]
    if getattr(args, "keyword_pairs", False):
        if not keywords:
            raise InputError(
                "--keyword-pairs requires --wordlist (the true keyword pair must be present)"
            )
        lengths = [int(x) for x in str(args.lengths).split(",") if x.strip()]
        result = crack_double_columnar_keywords(
            text,
            scorer,
            lengths=lengths,
            wordlist=keywords,
            alphabet=args.alphabet,
            unit=getattr(args, "unit", 1),
        )
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(result, ensure_ascii=False, indent=indent))
        return 0
    result = crack_transposition_over_sub(
        text,
        scorer,
        alphabet=args.alphabet,
        layers=args.layers,
        widths=range(args.min_width, args.max_width + 1),
        keywords=keywords,
        sa_restarts=args.restarts,
        unit=getattr(args, "unit", 1),
        rng=rng,
    )
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(result, ensure_ascii=False, indent=indent))
    else:
        if result.get("structure") is None:
            print(f"no transposition found (reveal-IoC {result['reveal_ioc']})")
            return 0
        s = result["structure"]
        verdict = "RECOVERED" if result.get("recovered") else "unconfirmed (spurious reveal?)"
        print(
            f"{s.get('transposition')} OVER {s.get('substitution', 'substitution')} "
            f"period {s.get('period')} — reveal-IoC {result['reveal_ioc']}, "
            f"word-coverage {result.get('word_coverage')} [{verdict}]"
        )
        print(f"plaintext: {result['plaintext']}")
    return 0


def _cmd_runkey(args) -> int:
    from .runkey import screen_running_keys
    from .scoring import resolve_scorer

    text = _resolve_text(args)
    scorer = resolve_scorer("quadgrams", args.lang)

    key_texts: list[str] = []
    labels: list[str] = []
    for i, kt in enumerate(args.keytext or []):
        key_texts.append(kt)
        labels.append(f"arg{i}")
    for path in args.keytext_file or []:
        with open(path, encoding="utf-8") as fh:
            blob = fh.read().replace("\r\n", "\n").replace("\r", "\n")
        # A running key is usually ONE long passage (often line-wrapped), so a file with no
        # blank-line separation is treated as a single key-text. Blank lines split it into
        # one key-text per paragraph (a file of candidate keys).
        paras = [c for c in blob.split("\n\n") if c.strip()]
        chunks = paras if len(paras) > 1 else ([blob] if blob.strip() else [])
        for j, c in enumerate(chunks):
            key_texts.append(c)
            labels.append(f"{path}:{j}" if len(chunks) > 1 else path)
    if not key_texts:
        raise InputError("provide at least one --keytext or --keytext-file")

    result = screen_running_keys(
        text,
        key_texts,
        labels=labels,
        scorer=scorer,
        alphabets=tuple(a.strip() for a in args.alphabets.split(",") if a.strip()),
        conventions=tuple(c.strip() for c in args.conventions.split(",") if c.strip()),
        max_width=args.max_width,
        peel=not args.no_peel,
        reveal_ioc=args.ioc_floor,
    )
    shown = dict(result)
    shown["ranked"] = result.get("ranked", [])[: args.top]

    if args.json or getattr(args, "compact", False):
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(shown, ensure_ascii=False, indent=indent))
        return 0 if result.get("recovered") else 1

    w = result.get("winner")
    if w is None:
        print(result.get("note") or "no key-text produced an IoC outlier", file=sys.stderr)
        return 1
    tag = "transposed-English" if w["transposed_english"] else "no IoC outlier"
    lone = "n/a" if w["z_outlier"] is None else f"{w['z_outlier']:.1f}"
    print(f"runkey screen: {result['trials']} trials")
    print(
        f"winner: keytext={w['label']!r} {w['alphabet']}/{w['convention']}  "
        f"IoC={w['ioc']:.4f}  z={w['z']:.1f} (lone z={lone})  [{tag}]"
    )
    if result.get("note") and not result.get("recovered"):
        print(f"note: {result['note']}")
    s = result.get("structure")
    if s and s.get("transposition") == "columnar":
        verdict = "RECOVERED" if result["recovered"] else "unconfirmed"
        print(
            f"peeled: columnar width {s['columnar_width']} "
            f"order {','.join(map(str, s['columnar_order']))}  "
            f"qscore/char={result['qscore_per_char']:.2f}  "
            f"word-cov={result['word_coverage']:.2f} [{verdict}]"
        )
    print(f"plaintext: {result['plaintext']}")
    if len(shown["ranked"]) > 1:
        print("\nranked trials:")
        for r in shown["ranked"]:
            print(
                f"  IoC={r['ioc']:.4f} z={r['z']:>5.1f} "
                f"[{r['label']} / {r['alphabet']} / {r['convention']}]"
            )
    return 0 if result.get("recovered") else 1


def _word_vocab(spec: str | None) -> list[str]:
    """Parse a vocabulary spec: ``@file`` (whitespace-separated) or a comma/space list."""
    if not spec:
        return []
    if spec.startswith("@"):
        with open(spec[1:], encoding="utf-8") as fh:
            return [w for w in fh.read().split() if w]
    return [w for w in spec.replace(",", " ").split() if w]


def _cmd_keysource(args) -> int:
    from .keysources import compose_key, decompose_key, keys_from_corpus

    json_mode = args.json or getattr(args, "compact", False)

    def _out(obj: dict) -> None:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(obj, ensure_ascii=False, indent=indent))

    if args.compose:
        word_a, word_b = args.compose
        key = compose_key(word_a, word_b, alphabet=args.alphabet, convention=args.convention)
        obj = {
            "mode": "compose",
            "word_a": word_a.upper(),
            "word_b": word_b.upper(),
            "alphabet": args.alphabet.upper(),
            "convention": args.convention,
            "period": len(key),
            "key": key,
        }
        _out(obj) if json_mode else print(f"composed key (period {len(key)}): {key}")
        return 0

    if args.decompose:
        vocab = _word_vocab(args.words)
        if not vocab:
            raise InputError("--decompose needs a --words vocabulary (comma list or @file)")
        pairs = decompose_key(
            args.decompose, vocab, alphabet=args.alphabet, convention=args.convention
        )
        if json_mode:
            _out({"mode": "decompose", "key": only_letters(args.decompose).upper(), "pairs": pairs})
        elif not pairs:
            print("no word pair in the vocabulary composes that key", file=sys.stderr)
            return 1
        else:
            for pr in pairs:
                print(
                    f"{pr['word_a']} + {pr['word_b']}  (period {pr['period']}, {pr['convention']})"
                )
        return not pairs

    corpus: list[str] = list(args.corpus or [])
    for path in args.corpus_file or []:
        with open(path, encoding="utf-8") as fh:
            corpus.append(fh.read())
    if not corpus:
        raise InputError("provide --corpus/--corpus-file to derive keys, or --compose/--decompose")
    cands = keys_from_corpus(
        corpus, min_word=args.min_word, window_lengths=tuple(args.windows or ())
    )

    target = None
    if getattr(args, "text", None) not in (None, "-") or getattr(args, "file", None):
        target = _resolve_text(args)

    if target:
        from .runkey import screen_running_keys
        from .scoring import resolve_scorer

        run_kinds = ("full", "acrostic-sentence", "acrostic-line", "reverse:full")
        picks = [c for c in cands if c["kind"] in run_kinds]
        screen = screen_running_keys(
            target,
            [c["value"] for c in picks],
            labels=[f"{c['source']}/{c['kind']}" for c in picks],
            scorer=resolve_scorer("quadgrams", args.lang),
        )
        shown = dict(screen)
        shown["ranked"] = screen.get("ranked", [])[: args.top]
        if json_mode:
            _out(
                {
                    "mode": "screen",
                    "candidate_count": len(cands),
                    "screened": len(picks),
                    "screen": shown,
                }
            )
        else:
            w = screen.get("winner")
            if not w:
                print("no derived running key produced an IoC outlier", file=sys.stderr)
                return 1
            print(f"keysource screen: {len(picks)} running-key candidates from corpus")
            print(
                f"winner: {w['label']} {w['alphabet']}/{w['convention']} "
                f"IoC={w['ioc']:.4f} z={w['z']:.1f}"
            )
            if screen.get("recovered"):
                print(f"plaintext: {screen.get('plaintext')}")
        return 0 if screen.get("recovered") else 1

    if json_mode:
        _out({"mode": "derive", "count": len(cands), "candidates": cands[: args.top]})
    else:
        for c in cands[: args.top]:
            print(f"[{c['kind']:<18}] {c['value'][:60]}  (from {c['source']})")
        if len(cands) > args.top:
            print(f"... {len(cands) - args.top} more (raise --top or use --json)")
    return 0


def _cmd_validate(args) -> int:
    from .validate import genuine_solve_signature, make_synthetic, positive_control

    spec = {"structure": args.structure}
    key: dict = {"substitution": args.substitution, "alphabet": args.alphabet}
    if args.sub_key:
        key["sub_key"] = args.sub_key
    if args.columnar_keyword:
        key["columnar_keyword"] = args.columnar_keyword
    if args.columnar_keywords:
        key["columnar_keywords"] = _word_vocab(args.columnar_keywords)

    synth = make_synthetic(spec, args.plaintext, key=key, length=args.length)
    out = {
        "structure": synth["structure"],
        "length": synth["length"],
        "ciphertext": synth["ciphertext"],
        "plaintext": synth["plaintext"],
        "key": {k: v for k, v in synth["key"].items() if k != "structure"},
        "signature": genuine_solve_signature(synth["length"]),
    }
    if args.self_check:

        def _attack(ct: str) -> str:
            res = engine.auto(ct, per_cipher_timeout=args.timeout)
            best = res.best()
            return best.plaintext if best else ""

        out["self_check"] = positive_control(
            _attack, spec, key, plaintext=args.plaintext, length=args.length
        )

    if args.json or getattr(args, "compact", False):
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(out, ensure_ascii=False, indent=indent))
    else:
        print(f"synthetic [{out['structure']}] len={out['length']}")
        print(f"ciphertext: {out['ciphertext']}")
        print(f"plaintext:  {out['plaintext']}")
        sig = out["signature"]
        bar = f"qscore/char>={sig['qscore_per_char']}  word_cov>={sig['word_cov']}"
        print(f"genuine-solve bar: {bar}")
        if args.self_check:
            sc = out["self_check"]
            verdict = "RECOVERED" if sc["recovered"] else "did NOT recover"
            print(
                f"self-check (butt auto): {verdict}  "
                f"(word_cov={sc['word_cov']}, preview={sc['decode_preview']!r})"
            )
    return 0


def _cmd_hillkpa(args) -> int:
    from .ciphers.hill import matrix_to_key
    from .hill_kpa import crib_drag

    ct = _resolve_text(args)
    results = crib_drag(args.crib, ct, args.size, alphabet=args.alphabet, top=args.top)
    out = {
        "block_size": args.size,
        "alphabet": args.alphabet.upper(),
        "crib": only_letters(args.crib).upper(),
        "count": len(results),
        "results": [
            {
                "offset": r["offset"],
                "key": matrix_to_key(r["matrix"]),
                "score": round(r["score"], 4),
                "plaintext": r["plaintext"],
            }
            for r in results
        ],
    }
    if args.json or getattr(args, "compact", False):
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(out, ensure_ascii=False, indent=indent))
    elif not results:
        print(
            "no invertible key recovered (crib too short, wrong n, or wrong alphabet?)",
            file=sys.stderr,
        )
        return 1
    else:
        best = out["results"][0]
        print(f"hill KPA (n={args.size}, {args.alphabet.upper()}): {len(results)} key(s)")
        print(f"best: offset={best['offset']} key={best['key']} score={best['score']}")
        print(f"plaintext: {best['plaintext']}")
    return 0 if results else 1


def _cmd_compare(args) -> int:
    from .compare import compare

    text_a = _resolve_text(args)
    if args.with_text is not None:
        text_b = args.with_text
    elif args.with_file:
        with open(args.with_file, encoding="utf-8") as fh:
            text_b = fh.read()
    else:
        raise InputError("provide the second ciphertext via --with TEXT or --with-file PATH")

    res = compare(text_a, text_b, max_period=args.max_period)
    if args.json or getattr(args, "compact", False):
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(res, ensure_ascii=False, indent=indent))
    else:
        v = res["verdict"]
        print(
            f"compare: len {res['len_a']} vs {res['len_b']}  "
            f"IoC {res['ioc_a']:.3f}/{res['ioc_b']:.3f}"
        )
        print(
            f"freq-profile L1: a~b={res['freq_profile_l1']:.3f}  "
            f"a~EN={res['l1_a_english']:.3f}  b~EN={res['l1_b_english']:.3f}"
        )
        print(f"verdict: shared_construction={v['shared_construction']} ({v['confidence']})")
        for line in v["evidence"]:
            print(f"  - {line}")
    return 0


def _cmd_nonprose(args) -> int:
    from .nonprose import nonprose_flag

    flag = nonprose_flag(_resolve_text(args))
    if args.json or getattr(args, "compact", False):
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(flag, ensure_ascii=False, indent=indent))
    else:
        print(
            f"nonprose: route={flag['route_score']:.2f}  prose={flag['prose_score']:.2f}  "
            f"delta={flag['delta']:+.2f}  ->  {flag['verdict']}"
        )
    return 0


def _cmd_joint(args) -> int:
    from . import joint

    text = _resolve_text(args)
    alphabets = tuple(a.strip() for a in args.alphabets.split(",") if a.strip())
    variants = ("vig", "beaufort") if args.variant == "both" else (args.variant,)
    results = joint.solve(
        text,
        layer=args.layer,
        widths=range(args.min_width, args.max_width + 1),
        periods=range(args.min_period, args.max_period + 1),
        alphabets=alphabets,
        variants=variants,
        restarts=args.restarts,
        iters=args.iters,
        ngram=args.ngram,
        seed=args.seed or 0,
        top=args.top,
    )
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(results, ensure_ascii=False, indent=indent))
        return 0
    for r in results:
        print(
            f"[{r['layer']} w{r['width']} p{r['period']} {r['alphabet']}/{r['variant']} "
            f"{r['ngram']}] fitness={r['score']:.3f} quad={r['quad']:.0f}{_recovered_flag(r)}"
        )
        print(f"  {r['plaintext']}")
    return 0


def _cmd_anagram(args) -> int:
    from . import anagram

    text = _resolve_text(args)
    r = anagram.solve(text, widths=range(args.min_width, args.max_width + 1))
    if args.json:
        print(
            json.dumps(r, ensure_ascii=False, indent=None if getattr(args, "compact", False) else 2)
        )
        return 0
    print(f"[transposition w{r.get('width')}] score={r.get('score'):g} order={r.get('order')}")
    print(r.get("plaintext", ""))
    return 0


def _cmd_quagmire(args) -> int:
    from . import quagmire_solver

    text = _resolve_text(args)
    if getattr(args, "alphabet", None) and args.kind == "vigenere":
        print(
            "warning: --kind vigenere uses the standard A-Z alphabet, so --alphabet is "
            "ignored; use --kind quagmire3 for a keyed-alphabet (e.g. KRYPTOS) Vigenere",
            file=sys.stderr,
        )
    # --alphabet (known alphabet) and Beaufort both use the fast fixed-alphabet path
    # (Beaufort isn't an annealing kind; routing it here is also the crash fix).
    if getattr(args, "alphabet", None) or args.kind == "beaufort":
        r = quagmire_solver.solve_fixed_alphabet(
            text,
            args.alphabet or quagmire_solver.KRYPTOS_ALPHABET,
            kind=args.kind,
            periods=range(args.min_period, args.max_period + 1),
            ct_alphabet=getattr(args, "ct_alphabet", None),
        )
    else:
        r = quagmire_solver.solve(
            text,
            periods=range(args.min_period, args.max_period + 1),
            kind=args.kind,
            restarts=args.restarts,
            seed=args.seed,
        )
    if args.json:
        print(
            json.dumps(r, ensure_ascii=False, indent=None if getattr(args, "compact", False) else 2)
        )
        return 0
    print(f"[{args.kind} period {r.get('period')}] fitness={r.get('score'):g}{_recovered_flag(r)}")
    print(r.get("plaintext", ""))
    return 0


def _cmd_homophonic(args) -> int:
    from . import homophonic

    text = _resolve_text(args)
    r = homophonic.solve(text, restarts=args.restarts, iters=args.iters)
    if args.json:
        print(
            json.dumps(r, ensure_ascii=False, indent=None if getattr(args, "compact", False) else 2)
        )
        return 0
    print(f"[substitution] fitness={r.get('score'):g}{_recovered_flag(r)}")
    print(r.get("plaintext", ""))
    return 0


def _cmd_layered(args) -> int:
    from .layered import crack_layered, crack_quagmire_over_columnar
    from .scoring import resolve_scorer

    text = _resolve_text(args)
    scorer = resolve_scorer("quadgrams", args.lang)
    order = [int(x) for x in args.order.replace(",", " ").split()] if args.order else None
    rng = random.Random(args.seed) if args.seed is not None else None
    widths = range(args.min_width, args.max_width + 1)
    if args.period is None:
        # fully autonomous: detect the period and brute small columnar widths
        result = crack_layered(
            text, scorer, alphabet=args.alphabet, widths=widths, workers=args.workers, rng=rng
        )
    else:
        result = crack_quagmire_over_columnar(
            text,
            scorer,
            alphabet=args.alphabet,
            period=args.period,
            widths=widths,
            order=order,
            rng=rng,
        )
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(result, ensure_ascii=False, indent=indent))
    else:
        s = result["structure"]
        print(
            f"{s['substitution']} period {s['period']} OVER columnar "
            f"width {s['columnar_width']} order {s['columnar_order']}"
        )
        print(f"plaintext: {result['plaintext']}")
        if result["residual"]:
            print(
                f"\n{len(result['residual'])} ambiguous column(s) (short columns) — an agent "
                "should pick the shift that yields real words:"
            )
            for entry in result["residual"][:60]:
                opts = " | ".join(
                    f"sh{o['shift']}->{o['letters']}{'*' if o['current'] else ''}"
                    for o in entry["options"]
                )
                print(f"  col{entry['column']:2d}: {opts}")
    return 0


def _cmd_identify(args) -> int:
    text = _resolve_text(args)
    if getattr(args, "types", False):
        info = cipher_id.identify_types(text)
        if args.json:
            indent = None if getattr(args, "compact", False) else 2
            print(json.dumps(info, ensure_ascii=False, indent=indent))
        else:
            for r in info["ranked"][:6]:
                period = f" period={r['period']}" if r.get("period") else ""
                print(f"  {r['score']:.3f}  {r['type']}{period}  ({r.get('reasons', '')})")
        return 0
    info = identify_text(text)
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(info, ensure_ascii=False, indent=indent))
    else:
        print(
            f"length={info['length']}  IoC={info['index_of_coincidence']}  "
            f"chi2/letter={info['chi_squared_per_letter']}"
        )
        for f in info["likely_families"]:
            print(f"  {f['weight']:.3f}  {f['family']:<16} {', '.join(f['ciphers'])}")
        if info.get("periodic_ioc"):
            spectrum = "  ".join(f"{p['period']}(z{p['z']})" for p in info["periodic_ioc"])
            print(f"  periods: {spectrum}")
        if info.get("diagnosis"):
            print(f"  diagnosis: {info['diagnosis']}")
    return 0


def _cmd_diagnose(args) -> int:
    from .diagnose import diagnose

    text = _resolve_text(args)
    info = diagnose(text)
    if args.json or getattr(args, "compact", False):
        print(json.dumps(info, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    print(f"length={info['length']}  structure: {info['structure']}")
    print(f"\n{info['summary']}\n")
    print("recommended attacks:")
    for r in info["recommended"]:
        print(f"  - {r}")
    inferences = info.get("inferences") or []
    if inferences:
        print("\nstructural inferences:")
        for inf in inferences:
            print(f"  ! {inf}")
    sig = info.get("signals", {})
    if sig.get("calibrated_periods"):
        sp = "  ".join(f"{p['period']}(z{p['z']})" for p in sig["calibrated_periods"][:4])
        print(f"\nperiods: {sp}")
    if sig.get("structural_lags"):
        print(f"autocorr lags: {sig['structural_lags']}")
    blk = sig.get("block_transposition") or {}
    if blk.get("best_block"):
        bb = blk["best_block"]
        al = (blk.get("alignment") or {}).get(bb) or {}
        print(
            f"block alignment: best_block={bb} residue={al.get('residue')} p={al.get('p')} "
            f"— try --unit {bb}"
        )
    decay = sig.get("ioc_decay") or {}
    if decay:
        print(f"IoC decay: quarters {decay.get('quarter_ioc')} slope_z={decay.get('slope_z')}")
    if sig.get("crackability"):
        print(f"crackability: {sig['crackability'].get('verdict')}")
    return 0


def _cmd_pipeline(args) -> int:
    text = _resolve_text(args)
    steps: list[tuple[str, str]] = []
    for raw in args.step:
        name, sep, key = raw.partition(":")
        if not sep:
            raise InputError(f"--step must be 'cipher:key'; got {raw!r}")
        steps.append((name.strip(), key))
    op = "encode" if args.encode else "decode"
    final, trace = engine.pipeline(text, steps, op=op)
    if args.json or args.compact:
        out = {
            "ok": True,
            "operation": "pipeline",
            "direction": op,
            "trace": trace,
            "plaintext": final,
        }
        print(json.dumps(out, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    for t in trace:
        print(f"# step {t['step']}: {op} {t['cipher']} key={t['key']!r}")
    print(final)
    return 0


def _cmd_transform(args) -> int:
    from . import transforms

    text = _resolve_text(args)
    if getattr(args, "decimate", None):
        period, _, offset = args.decimate.partition(":")
        out_text = transforms.decimate(text, int(period), int(offset or 0))
        cands = [{"kind": f"decimate {args.decimate}", "text": out_text}]
    else:
        cands = transforms.candidates(text)
    if args.json or args.compact:
        out = {"ok": True, "operation": "transform", "candidates": cands}
        print(json.dumps(out, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    if not cands:
        print("no pre-flight transforms apply (input looks like a plain letter stream)")
    for c in cands:
        print(f"# {c['kind']}")
        print(c["text"])
        print()
    return 0


def _cmd_keyword(args) -> int:
    # --order: invert a recovered columnar READ-ORDER back to its keyword / named generator.
    if getattr(args, "order", None):
        order = _parse_orders(args.order)[0]
        gens = keyfinder.describe_permutation(order)
        words: list[str] = []
        if args.wordlist:
            with open(args.wordlist) as fh:
                wl = [ln.strip() for ln in fh if ln.strip()]
            words = keyfinder.keyword_from_order(order, wl)
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": bool(words or gens),
                        "operation": "keyword",
                        "order": order,
                        "keywords": words,
                        "generators": gens,
                    }
                )
            )
        else:
            print(f"keywords: {' '.join(words) or '(none in wordlist)'}")
            print(f"generators: {' '.join(gens) or '(no known generator)'}")
        return 0 if (words or gens) else 1
    text = _resolve_text(args)
    cands = keyfinder.find_keyword_in(text, size=args.size)
    if args.json:
        print(json.dumps({"ok": bool(cands), "operation": "keyword", "candidates": cands}))
    else:
        print(" ".join(cands) if cands else "(no keyword recovered — alphabet looks unkeyed)")
    return 0


def _cmd_split(args) -> int:
    text = _resolve_text(args)
    entries = splitter.split_ciphers(text)
    if args.json:
        print(
            json.dumps(
                {"ok": True, "operation": "split", "count": len(entries), "entries": entries},
                ensure_ascii=False,
            )
        )
    else:
        for e in entries:
            print(f"# {e['title'] or '(untitled)'}")
            print(e["body"])
            print()
    return 0


def _cmd_relation(args) -> int:
    text = _resolve_text(args)
    alphabets = tuple(a.strip() for a in args.alphabets.split(",") if a.strip())
    info = ngram_relation.scan(
        text,
        n=args.n,
        alphabets=alphabets,
        samples=args.samples,
        seed=args.seed,
        top=args.top,
    )
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(info, ensure_ascii=False, indent=indent))
    else:
        print(f"n={info['n']}  groups={info.get('groups')}  floor={info.get('floor')}")
        for c in info["candidates"]:
            print(f"  {c['alphabet']:>7} coef={c['coef']}  IoC={c['ioc']}  z={c['z']}  p={c['p']}")
        print(info["verdict"])
        best = info["candidates"][0] if info["candidates"] else None
        if best and "relation found" in info["verdict"]:
            chan = ngram_relation.combine(text, best["coef"], best["alphabet"])
            print(f"channel: {chan}")
    return 0


def _cmd_channel(args) -> int:
    text = _resolve_text(args)
    widths = tuple(args.widths) if args.widths else (2, 3)
    info = analysis.linear_channel_width(
        text,
        alphabet=args.alphabet,
        widths=widths,
        null_trials=args.samples,
        seed=args.seed,
    )
    if args.json or getattr(args, "compact", False):
        print(json.dumps(info, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    print(f"verdict: {info.get('verdict')}")
    for w in sorted(info.get("widths", {}), key=int):
        d = info["widths"][w]
        ioc = d.get("ioc")
        z = d.get("z")
        parts = [f"width {w}:"]
        parts.append(f"IoC={ioc:.4f}" if isinstance(ioc, (int, float)) else "IoC=-")
        parts.append(f"z={z:.2f}" if isinstance(z, (int, float)) else "z=-")
        parts.append(f"beats_null_max={d.get('beats_null_max')}")
        parts.append(f"blocks={d.get('blocks')}")
        parts.append(f"exhaustive={d.get('search_exhaustive')}")
        parts.append(f"hit={d.get('hit')}")
        line = "  " + "  ".join(parts)
        if d.get("note"):
            line += f"  — {d['note']}"
        print(line)
    return 0


def _cmd_subfrac(args) -> int:
    from . import sub_fractionation as sf

    text = _resolve_text(args)
    squares: str | list[str]
    if not args.squares or args.squares == "dictionary":
        squares = "dictionary"
    else:
        squares = [s.strip().upper() for s in args.squares.split(",") if s.strip()]
    langs = tuple(a.strip() for a in args.langs.split(",") if a.strip()) or ("english",)
    results = sf.crack_sub_over_bifid(
        text,
        outer_alphabet=args.alphabet,
        inner_period=args.inner_period,
        outer_period=args.outer_period,
        squares=squares,
        objective=args.objective,
        drop_letter=args.drop_letter,
        languages=langs,
        top=args.top,
        timeout=args.timeout,
    )
    if args.json or getattr(args, "compact", False):
        out = [
            {"square": s, "key": k, "plaintext": p, "score": round(sc, 3)}
            for (s, k, p, sc) in results
        ]
        print(json.dumps(out, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    if not results:
        print("no candidates")
        return 0
    for s, k, p, sc in results:
        print(f"score={sc:.2f}  key={k}  square={s}")
        print(f"  {p}")
    return 0


def _parse_squares(spec):
    if not spec or spec == "dictionary":
        return "dictionary"
    return [s.strip().upper() for s in spec.split(",") if s.strip()]


def _print_cracker_results(rows, args) -> int:
    """Shared reporter for the decoupled-cracker commands. ``rows`` are dicts."""
    if args.json or getattr(args, "compact", False):
        print(json.dumps(rows, ensure_ascii=False, indent=None if args.compact else 2))
        return 0
    if not rows:
        print("no candidates")
        return 0
    for r in rows:
        meta = "  ".join(f"{k}={v}" for k, v in r.items() if k not in ("plaintext", "score"))
        print(f"score={r['score']:.2f}  {meta}")
        print(f"  {r['plaintext']}")
    return 0


def _cmd_subplayfair(args) -> int:
    from . import sub_playfair as sp

    langs = tuple(a.strip() for a in args.langs.split(",") if a.strip()) or ("english",)
    shapes = [
        tuple(int(x) for x in s.lower().split("x")) for s in args.shape.split(",") if s.strip()
    ]
    results = sp.crack_sub_over_playfair(
        _resolve_text(args),
        outer_alphabet=args.alphabet,
        outer_period=args.outer_period,
        squares=_parse_squares(args.squares),
        shapes=shapes,
        objective=args.objective,
        drop_letter=args.drop_letter,
        languages=langs,
        top=args.top,
        timeout=args.timeout,
    )
    rows = [
        {"square": s, "key": k, "plaintext": p, "score": round(sc, 3)} for (s, k, p, sc) in results
    ]
    return _print_cracker_results(rows, args)


def _cmd_subserpf(args) -> int:
    from . import sub_seriated_playfair as ss

    langs = tuple(a.strip() for a in args.langs.split(",") if a.strip()) or ("english",)
    results = ss.crack_sub_over_seriated_playfair(
        _resolve_text(args),
        outer_alphabet=args.alphabet,
        inner_period=args.inner_period,
        outer_period=args.outer_period,
        squares=_parse_squares(args.squares),
        objective=args.objective,
        drop_letter=args.drop_letter,
        languages=langs,
        top=args.top,
        timeout=args.timeout,
    )
    rows = [
        {"square": s, "key": k, "plaintext": p, "score": round(sc, 3)} for (s, k, p, sc) in results
    ]
    return _print_cracker_results(rows, args)


def _cmd_subfoursq(args) -> int:
    from . import sub_four_square as f4

    langs = tuple(a.strip() for a in args.langs.split(",") if a.strip()) or ("english",)
    bl = _parse_squares(args.bl_squares) if args.bl_squares else None
    results = f4.crack_sub_over_four_square(
        _resolve_text(args),
        outer_alphabet=args.alphabet,
        outer_period=args.outer_period,
        tr_squares=_parse_squares(args.squares),
        bl_squares=bl,
        objective=args.objective,
        drop_letter=args.drop_letter,
        languages=langs,
        top=args.top,
        timeout=args.timeout,
    )
    rows = [
        {"tr_square": tg, "bl_square": bg, "key": k, "plaintext": p, "score": round(sc, 3)}
        for (tg, bg, k, p, sc) in results
    ]
    return _print_cracker_results(rows, args)


def _cmd_subtwosq(args) -> int:
    from . import sub_two_square as t2

    langs = tuple(a.strip() for a in args.langs.split(",") if a.strip()) or ("english",)
    bot = _parse_squares(args.bot_squares) if args.bot_squares else None
    layouts = {"vertical": (True,), "horizontal": (False,), "both": (True, False)}[args.layout]
    results = t2.crack_sub_over_two_square(
        _resolve_text(args),
        outer_alphabet=args.alphabet,
        outer_period=args.outer_period,
        top_squares=_parse_squares(args.squares),
        bot_squares=bot,
        layouts=layouts,
        objective=args.objective,
        drop_letter=args.drop_letter,
        languages=langs,
        top=args.top,
        timeout=args.timeout,
    )
    rows = [
        {
            "top_square": tg,
            "bot_square": bg,
            "layout": "V" if v else "H",
            "key": k,
            "plaintext": p,
            "score": round(sc, 3),
        }
        for (tg, bg, v, k, p, sc) in results
    ]
    return _print_cracker_results(rows, args)


def _cmd_stats(args) -> int:
    text = _resolve_text(args)
    info = analysis.analyze(text, with_contacts=getattr(args, "contacts", False))
    if getattr(args, "significance", False):
        info["period_significance"] = {
            str(p): {
                "mean_coset_ioc": round(m, 5),
                "letters_per_coset": n,
                "z": round(z, 2),
                "small_sample": bool(flag),
            }
            for p, (m, n, z, flag) in cipher_id.period_significance(text).items()
        }
    if getattr(args, "family", False):
        info["period_family"] = {
            stat: analysis.period_family_significance(
                text, statistic=stat, samples=getattr(args, "family_samples", 200)
            )
            for stat in ("coset_ioc", "kappa")
        }
    if args.json:
        indent = None if getattr(args, "compact", False) else 2
        print(json.dumps(info, ensure_ascii=False, indent=indent))
    else:
        print(
            f"length={info['length']}  IoC={info['index_of_coincidence']}  "
            f"chi2={info['chi_squared']}"
        )
        letters = "  ".join(
            f"{f['letter']}={f['percent']}%" for f in info["frequencies"][:8] if f["count"]
        )
        print(f"top letters: {letters}")
        bigrams = "  ".join(f"{g['gram']}({g['count']})" for g in info["bigrams"][:8])
        print(f"top bigrams: {bigrams}")
        if info["likely_periods"]:
            periods = "  ".join(f"{p['period']}(w{p['weight']})" for p in info["likely_periods"])
            print(f"likely periods: {periods}")
        for stat, r in (info.get("period_family") or {}).items():
            if r.get("best_period"):
                verdict = (
                    "SIGNIFICANT" if r["family_p"] < 0.05 else "not significant (multiplicity)"
                )
                print(
                    f"family[{stat}]: best period {r['best_period']} z={r['z']} "
                    f"family_p={r['family_p']} — {verdict}"
                )
        decay = info.get("ioc_decay") or {}
        if decay:
            flag = (
                " — NON-STATIONARY keystream (period unrecoverable)"
                if decay.get("non_stationary")
                else ""
            )
            print(f"IoC drift: quarters {decay['quarter_ioc']} slope_z={decay['slope_z']}{flag}")
        kappa = info.get("kappa_spectrum") or []
        strong = [f"{k['lag']}(z{k['z']})" for k in kappa[:5] if k["z"] >= 3]
        if strong:
            print(f"autocorr lags: {'  '.join(strong)}")
        blk = info.get("block_transposition") or {}
        if blk.get("best_block"):
            bb = blk["best_block"]
            al = (blk.get("alignment") or {}).get(bb) or {}
            print(
                f"block alignment: repeated {blk.get('ngram')}-grams all at residue "
                f"{al.get('residue')} (mod {bb}), p={al.get('p')} — block-of-{bb} transposition; "
                f"try --unit {bb}"
            )
        cliff = info.get("crackability_cliff") or {}
        if cliff.get("verdict"):
            print(f"crackability: {cliff['verdict']}")
        if info.get("contacts"):
            hi = "  ".join(f"{c['letter']}={c['variety']}" for c in info["contacts"][:8])
            print(f"contact variety (high=vowel-like): {hi}")
        if info.get("period_significance"):
            print("period significance (z vs matched null; * = small-sample/unreliable):")
            for p, d in sorted(info["period_significance"].items(), key=lambda kv: int(kv[0])):
                flag = " *" if d["small_sample"] else ""
                print(
                    f"  p={p}: coset-IoC={d['mean_coset_ioc']:.4f} z={d['z']:.2f} "
                    f"n/coset={d['letters_per_coset']}{flag}"
                )
    return 0


def _cmd_convert(args) -> int:
    text = _resolve_text(args)
    out = textops.convert(text, args.to, divider=args.divider, group_size=args.group)
    if args.json:
        print(json.dumps({"ok": True, "operation": "convert", "to": args.to, "output": out}))
    else:
        print(out)
    return 0


def _cmd_format(args) -> int:
    text = _resolve_text(args)
    out = textops.strip_whitespace(text) if args.strip else text
    if args.group:
        out = textops.group(out, args.group, letters_only=args.letters_only)
    if args.case == "upper":
        out = out.upper()
    elif args.case == "lower":
        out = out.lower()
    if args.json:
        print(json.dumps({"ok": True, "operation": "format", "output": out}))
    else:
        print(out)
    return 0


def _cmd_words(args) -> int:
    fn = {
        "match": words.match,
        "pattern": words.pattern,
        "anagram": words.anagram,
        "ngram": words.ngram,
    }[args.op]
    results = fn(args.query, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "operation": "words",
                    "op": args.op,
                    "query": args.query,
                    "count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(" ".join(results) if results else "(no matches)")
    return 0


def _cmd_list(args) -> int:
    rows = registry.describe()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            aliases = f" ({', '.join(r['aliases'])})" if r["aliases"] else ""
            print(f"{r['name']}{aliases}: {r['description']}")
            if args.verbose:
                kf = r["key_format"] or ("(none)" if not r["needs_key"] else "?")
                eg = f"  e.g. {r['key_example']}" if r["key_example"] else ""
                print(f"    key: {kf}{eg}")
    return 0


def _cmd_help(args) -> int:
    try:
        info = registry.describe_one(args.cipher)
    except KeyError as exc:
        return _emit_error(exc, args)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"{info['name']}  ({', '.join(info['aliases']) or 'no aliases'})")
        print(f"  {info['description']}")
        print(f"  key:      {info['key_format'] or ('(none)' if not info['needs_key'] else '?')}")
        if info["key_example"]:
            print(f"  example:  --key {info['key_example']}")
        print(f"  needs key: {info['needs_key']}   auto-cracked: {info['auto_crackable']}")
        print(f'  encode:   butt encode {info["name"]} "text" --key {info["key_example"] or "KEY"}')
        print(f'  crack:    butt crack {info["name"]} "ciphertext"')
    return 0


def _cmd_schema(args) -> int:
    """Machine-readable capability manifest for agents."""
    parser = build_parser()
    commands: dict[str, dict] = {}
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not isinstance(choices, dict):
            continue
        for cmd, sub in choices.items():
            flags = [
                {"flags": a.option_strings, "help": a.help}
                for a in sub._actions
                if a.option_strings
            ]
            commands[cmd] = {"flags": flags}
    manifest = {
        "tool": "buttcrack",
        "command": "butt",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "commands": commands,
        "result_envelope": {
            "ok": "candidate produced (NOT 'solved' — read verdict)",
            "verdict": "trust signal; one of verdict_values",
            "operation": "encode|decode|crack|auto",
            "cipher": "cipher name (or 'auto')",
            "plaintext": "the plaintext-side text",
            "ciphertext": "the cipher-side text",
            "key": "recovered/used key",
            "confidence": "calibrated 0..1 (sample-size aware; deflated by word_coverage)",
            "word_coverage": (
                "fraction tiled by real >=5-letter words (~0.5-0.8 English, ~0 'salad'); "
                "null if not computed (non-English/too short)"
            ),
            "margin": "confidence gap to runner-up",
            "crib_confirmed": (
                "crack/auto only: true when --crib was given and the top candidate's "
                "plaintext contains it (absent otherwise)"
            ),
            "ambiguous_with": "rival cipher named when verdict is ambiguous/unlikely and close",
            "candidates": (
                "ranked alternatives [{plaintext,cipher,key,score,confidence,word_coverage,meta}]"
            ),
        },
        "command_outputs": {
            "stats": {
                "length": "letter count",
                "index_of_coincidence": "IoC (~0.066 English, ~0.04 random/poly)",
                "chi_squared": "total chi-squared vs English letter frequencies",
                "chi_squared_per_letter": (
                    "chi-squared/letter; low (~<0.05) => still English freqs (transposition)"
                ),
                "frequencies": "per-letter counts",
                "bigrams": "top digraphs",
                "trigrams": "top trigrams",
                "kasiski_repeats": "repeated substrings + spacings",
                "likely_periods": "Kasiski-implied key periods",
                "periodic_ioc": "calibrated per-column IoC spectrum [{period,ioc,z,...}]",
                "transposition_periods": (
                    "lags with exact repeated-bigram spikes [{period,repeats,z}]"
                ),
                "ioc_decay": (
                    "positional IoC drift {quarter_ioc, slope_z, non_stationary} — "
                    "flags an evolving/non-stationary keystream"
                ),
                "contacts": "variety-of-contacts vowel finder (only with --contacts)",
            },
            "relation": {
                "n": "n-gram size scanned",
                "groups": "number of complete n-grams",
                "floor": "null-shuffle mean IoC (the random floor for this length)",
                "candidates": (
                    "ranked [{alphabet, coef, ioc, z, p}] — coef is the linear combination "
                    "of n-gram positions; z vs shuffle null; p is search-aware"
                ),
                "verdict": (
                    "'relation found' => combine() that coef/alphabet to get the plaintext "
                    "channel (a homophonic-expansion / tri-square-family cipher); else none"
                ),
            },
            "identify": {
                "length": "letter count",
                "index_of_coincidence": "IoC",
                "chi_squared_per_letter": "chi-squared/letter (English-fit / transposition test)",
                "reliable": "true if enough letters for the stats to be trustworthy",
                "likely_families": "ranked [{family,weight,ciphers}]",
                "periodic_ioc": "calibrated period spectrum [{period,ioc,z,...}]",
                "diagnosis": "plain-language routing hint, honest about hard cases",
            },
        },
        "verdict_values": list(VERDICT_VALUES),
        "exit_codes": {
            "0": "solved/likely (or encode/decode/info ok)",
            "1": "ran but unconvincing",
            "2": "bad input/usage",
        },
        "languages": list(LANGUAGES),
        "ciphers": registry.names(),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=None if args.compact else 2))
    return 0


def _cmd_solve(args) -> int:
    """NDJSON batch: one request object per line in, one result object per line out."""
    src = open(args.batch, encoding="utf-8") if args.batch not in (None, "-") else sys.stdin
    try:
        for lineno, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                print(json.dumps({"ok": False, "line": lineno, "error": f"bad json: {exc}"}))
                continue
            out = _run_batch_request(req)
            if "id" in req:
                out = {"id": req["id"], **out}
            print(json.dumps(out, ensure_ascii=False))
    finally:
        if src is not sys.stdin:
            src.close()
    return 0


def _run_batch_request(req: dict) -> dict:
    op = req.get("op", "auto")
    text = req.get("text", "")
    top = req.get("top", 3)
    seed = req.get("seed")
    try:
        if op == "auto":
            result = engine.auto(
                text, top=top, seed=seed, per_cipher_timeout=req.get("timeout", 5.0)
            )
        elif op == "crack":
            result = engine.crack(
                req["cipher"], text, top=top, seed=seed, timeout=req.get("timeout")
            )
        elif op == "encode":
            result = engine.encode(req["cipher"], text, req["key"])
        elif op == "decode":
            result = engine.decode(req["cipher"], text, req["key"])
        else:
            return {"ok": False, "error": f"unknown op {op!r}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return result.to_dict(top=top)


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="butt",
        description="buttcrack: crack classical ciphers from the command line.",
    )
    parser.add_argument("--version", action="version", version=f"buttcrack {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_io(p, with_text=True):
        if with_text:
            p.add_argument("text", nargs="?", help="input text (or '-'/stdin)")
            p.add_argument("--file", help="read input from a file")
        p.add_argument("-j", "--json", action="store_true", help="emit JSON")
        p.add_argument("--compact", action="store_true", help="single-line JSON")

    # encode / decode
    for name, fn in (("encode", _cmd_encode), ("decode", _cmd_decode)):
        p = sub.add_parser(name, help=f"{name} text with a known key")
        p.add_argument("cipher", help="cipher name (see `butt list`)")
        p.add_argument("--key", required=True, help="cipher key")
        add_io(p)
        p.set_defaults(func=fn)

    # crack
    p = sub.add_parser("crack", help="crack a specific cipher (keyless)")
    p.add_argument("cipher", help="cipher name (see `butt list`)")
    add_io(p)
    p.add_argument("--top", type=int, default=5, help="max candidates to return")
    p.add_argument("--seed", type=int, help="RNG seed for reproducible hill-climbing")
    p.add_argument("--timeout", type=float, help="time budget in seconds")
    p.add_argument("--key-length", type=int, help="force key length (vigenere)")
    p.add_argument("--max-key-length", type=int, help="max key length to try (vigenere)")
    p.add_argument("--restarts", type=int, help="hill-climb restarts (substitution)")
    p.add_argument(
        "--blind",
        action="store_true",
        help="attempt blind keyed-alphabet recovery (quagmire; best-effort, long texts)",
    )
    p.add_argument("--width", type=int, help="force column count (columnar)")
    p.add_argument("--max-width", type=int, help="max column count to try (columnar)")
    p.add_argument("--max-rails", type=int, help="max rails to try (railfence)")
    p.add_argument("--lang", choices=LANGUAGES, default="english", help="scoring language")
    p.add_argument(
        "--ngrams",
        choices=NGRAM_MODELS,
        default="quadgrams",
        help="fitness model (quintgrams sharper but needs a built table; falls back to quadgrams)",
    )
    p.add_argument("--wordlist", help="candidate-keyword file for dictionary cracking")
    p.add_argument(
        "--crib", help="known/guessed plaintext word; confirms & ranks candidates that contain it"
    )
    p.set_defaults(func=_cmd_crack)

    # layered (periodic substitution OVER a columnar transposition)
    p = sub.add_parser(
        "layered", help="crack a periodic (Quagmire) substitution over a columnar transposition"
    )
    add_io(p)
    p.add_argument(
        "--period",
        type=int,
        help="substitution period (key length); if omitted it is auto-detected from the "
        "raw-ciphertext column-IoC spectrum and small columnar widths are brute-forced",
    )
    p.add_argument(
        "--alphabet", default="KRYPTOS", help="keyed-alphabet keyword for the substitution"
    )
    p.add_argument(
        "--order", help="known columnar read-order e.g. 5,1,2,4,0,3,7,6 (skips the order search)"
    )
    p.add_argument("--min-width", type=int, default=4, help="min columnar width to search")
    p.add_argument("--max-width", type=int, default=8, help="max columnar width to search")
    p.add_argument(
        "--workers", type=int, help="parallel workers for the order brute (default: cores-2)"
    )
    p.add_argument("--seed", type=int, help="RNG seed for reproducible search")
    p.add_argument("--lang", choices=LANGUAGES, default="english", help="scoring language")
    p.set_defaults(func=_cmd_layered)

    # transsub (columnar transposition OVER a periodic substitution — mirror of `layered`)
    p = sub.add_parser(
        "transsub",
        help="crack a columnar transposition layered OVER a periodic (Quagmire) substitution",
    )
    add_io(p)
    p.add_argument(
        "--alphabet", default="KRYPTOS", help="keyed-alphabet keyword for the inner substitution"
    )
    p.add_argument(
        "--layers",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 = single columnar (keyword sweep); 2 = double columnar (blind SA, best-effort)",
    )
    p.add_argument("--min-width", type=int, default=7, help="min columnar width to search")
    p.add_argument("--max-width", type=int, default=9, help="max columnar width to search")
    p.add_argument(
        "--wordlist", help="keyword file (one per line) for the order sweep / keyword-pairs"
    )
    p.add_argument(
        "--keyword-pairs",
        action="store_true",
        help="directed double-columnar keyword-PAIR sweep (needs --wordlist)",
    )
    p.add_argument(
        "--lengths",
        default="9",
        help="comma-separated keyword widths for --keyword-pairs (e.g. 8,9,16,17)",
    )
    p.add_argument("--restarts", type=int, default=200, help="SA restarts for double columnar")
    p.add_argument("--seed", type=int, help="RNG seed for reproducible search")
    p.add_argument("--lang", choices=LANGUAGES, default="english", help="scoring language")
    p.add_argument(
        "--unit",
        type=int,
        default=1,
        help="transposition granularity: 1 = letters; 3 = trigraph (3-letter) blocks "
        "(keeps trigrams intact — block-transposition geometry). layers=1 only.",
    )
    p.add_argument(
        "--enum",
        action="store_true",
        help="fully enumerate every read-order (non-dictionary orders too) for widths "
        "min..max, ranked by the mapping-independent reveal-IoC + search-aware null "
        "(honours --unit). Finds numeric/random columnar keys a keyword sweep misses.",
    )
    p.add_argument(
        "--spectrum",
        action="store_true",
        help="diagnostic: per-(width,unit) best reveal-IoC + beats-null verdict — is a periodic "
        "substitution hidden UNDER a (block) transposition, and at what width/granularity?",
    )
    p.add_argument(
        "--orders",
        metavar="SPEC",
        help="CONFIRM-OR-DIE order decider: score specific candidate read-orders (recognition "
        "hypotheses) by solving the inner sub under each, vs a shuffle null. SPEC is "
        "';'-separated orders ('0,2,1,3;3,1,0,2'), @file (one per line), or '-' for stdin.",
    )
    p.add_argument(
        "--periods",
        default="6-45",
        help="inner-substitution period range for --orders/--spectrum, 'lo-hi' (default 6-45).",
    )
    p.set_defaults(func=_cmd_transsub)

    # runkey — running-key screen: rank candidate KEY-TEXTS by IoC-outlier, peel the winner
    p = sub.add_parser(
        "runkey",
        help="screen candidate running KEY-TEXTS by IoC-outlier, then peel a columnar",
    )
    add_io(p)
    p.add_argument(
        "--keytext",
        action="append",
        metavar="TEXT",
        help="a candidate running key-text, e.g. a sibling puzzle's plaintext (repeatable)",
    )
    p.add_argument(
        "--keytext-file",
        action="append",
        metavar="PATH",
        help="file of key-texts: blank-line-separated paragraphs, else one per line (repeatable)",
    )
    p.add_argument(
        "--alphabets",
        default="KRYPTOS,STD",
        help="comma-separated keyed-alphabet keywords to try (default KRYPTOS,STD)",
    )
    p.add_argument(
        "--conventions",
        default="vigenere,beaufort,variant",
        help="comma-separated substitution families to try",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=8,
        help="max columnar width for the peel (<=8 exhaustive; wider via anagram SA)",
    )
    p.add_argument(
        "--ioc-floor",
        type=float,
        default=0.058,
        help="de-sub IoC above which a winner is treated as transposed-English",
    )
    p.add_argument("--no-peel", action="store_true", help="screen only; don't peel the columnar")
    p.add_argument("--top", type=int, default=10, help="max ranked trials to show")
    p.add_argument("--lang", choices=LANGUAGES, default="english", help="scoring language")
    p.set_defaults(func=_cmd_runkey)

    # keysource — derive candidate keys from prior-solution texts; compose/decompose word-pair keys
    p = sub.add_parser(
        "keysource",
        help="derive candidate keys from prior-solution texts (running key / acrostic / word), "
        "compose/decompose two-word keys; give a target ciphertext to screen the running keys",
    )
    add_io(p)
    p.add_argument(
        "--corpus", action="append", metavar="TEXT", help="a prior-solution text (repeatable)"
    )
    p.add_argument(
        "--corpus-file",
        action="append",
        metavar="PATH",
        help="file of prior-solution text (repeatable)",
    )
    p.add_argument(
        "--compose",
        nargs=2,
        metavar=("WORD_A", "WORD_B"),
        help="build a composed word-pair key (period = lcm of the two lengths)",
    )
    p.add_argument(
        "--decompose", metavar="KEY", help="recover the word pair that composes KEY (needs --words)"
    )
    p.add_argument("--words", metavar="CSV|@FILE", help="candidate vocabulary for --decompose")
    p.add_argument("--alphabet", default="KRYPTOS", help="keyed alphabet for compose/decompose")
    p.add_argument("--convention", default="vigenere", choices=("vigenere", "beaufort", "variant"))
    p.add_argument("--min-word", type=int, default=3, help="min length for a word-key candidate")
    p.add_argument(
        "--windows",
        type=int,
        nargs="*",
        help="also emit every substring of these lengths (crib-drag territory)",
    )
    p.add_argument(
        "--top", type=int, default=40, help="max candidates to list / ranked screen trials to show"
    )
    p.add_argument(
        "--lang",
        choices=LANGUAGES,
        default="english",
        help="scoring language when screening against a target ciphertext",
    )
    p.set_defaults(func=_cmd_keysource)

    # validate — build a same-shape synthetic to prove an attack before trusting a negative
    p = sub.add_parser(
        "validate",
        help="build a same-structure synthetic ciphertext (validate-on-synthetic discipline); "
        "--self-check runs `butt auto` on it and reports whether it recovers",
    )
    p.add_argument(
        "--structure",
        required=True,
        choices=(
            "substitution",
            "columnar",
            "double-columnar",
            "substitution-over-columnar",
            "columnar-over-substitution",
        ),
        help="construction family to synthesise",
    )
    p.add_argument("--sub-key", help="substitution key (period = its length)")
    p.add_argument(
        "--substitution",
        default="vigenere",
        choices=("vigenere", "beaufort", "variant", "quagmire3"),
    )
    p.add_argument("--alphabet", default="KRYPTOS", help="keyed alphabet for the substitution")
    p.add_argument("--columnar-keyword", help="columnar read-order keyword")
    p.add_argument("--columnar-keywords", help="two keywords (comma list) for double-columnar")
    p.add_argument("--length", type=int, help="synthetic length (default: built-in filler length)")
    p.add_argument("--plaintext", help="use this plaintext instead of the built-in English filler")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="run `butt auto` on the synthetic and judge recovery",
    )
    p.add_argument("--timeout", type=float, default=5.0, help="per-cipher budget for --self-check")
    add_io(p, with_text=False)
    p.set_defaults(func=_cmd_validate)

    # hillkpa — Hill known-plaintext (crib) attack: recover the matrix from a little known plaintext
    p = sub.add_parser(
        "hillkpa",
        help="Hill known-plaintext attack: slide a crib, recover the n x n key over a keyed "
        "alphabet (CRT over Z26), decrypt and rank",
    )
    add_io(p)
    p.add_argument("--crib", required=True, help="known/probable plaintext (>= n full blocks)")
    p.add_argument("--size", "-n", type=int, default=3, help="Hill block size n (default 3)")
    p.add_argument(
        "--alphabet", default="STD", help="index alphabet: STD, KRYPTOS, or a 26-letter permutation"
    )
    p.add_argument("--top", type=int, default=10, help="max ranked keys to report")
    p.set_defaults(func=_cmd_hillkpa)

    # compare — sibling-pair analysis: do two ciphertexts plausibly share a construction?
    p = sub.add_parser(
        "compare",
        help="compare two ciphertexts for a shared construction (freq-profile distance, "
        "period/kappa signature, additive-translate superimposition) — for chained series",
    )
    add_io(p)
    p.add_argument(
        "--with", dest="with_text", metavar="TEXT", help="the second ciphertext (or --with-file)"
    )
    p.add_argument("--with-file", metavar="PATH", help="read the second ciphertext from a file")
    p.add_argument("--max-period", type=int, default=16, help="max period for the kappa signature")
    p.set_defaults(func=_cmd_compare)

    # nonprose — flag a candidate that scores like English but reads as route/structured text
    p = sub.add_parser(
        "nonprose",
        help="score text under a route/instruction model vs a prose model; flags a "
        "structured (directions/coordinates/list) payload an English scorer would miss",
    )
    add_io(p)
    p.set_defaults(func=_cmd_nonprose)

    # joint — dual nested simulated-annealing climber over transposition + substitution,
    # scored on the FINAL plaintext (entropy-normalized hexagrams). AZdecrypt-style.
    p = sub.add_parser(
        "joint",
        help=(
            "joint transposition+substitution dual-SA climber (entropy-normalized hexagram fitness)"
        ),
    )
    add_io(p)
    p.add_argument(
        "--layer",
        choices=("inner", "outer"),
        default="inner",
        help="inner = transposition OUTER / sub INNER; outer = sub OUTER",
    )
    p.add_argument("--min-width", type=int, default=8, help="min columnar width to search")
    p.add_argument("--max-width", type=int, default=17, help="max columnar width to search")
    p.add_argument("--min-period", type=int, default=9, help="min substitution period")
    p.add_argument("--max-period", type=int, default=16, help="max substitution period")
    p.add_argument(
        "--alphabets", default="KRYPTOS,STD", help="comma-separated keyed-alphabet keywords"
    )
    p.add_argument("--variant", choices=("vig", "beaufort", "both"), default="both")
    p.add_argument("--restarts", type=int, default=20, help="annealing restarts per config")
    p.add_argument("--iters", type=int, default=4000, help="annealing iterations per restart")
    p.add_argument(
        "--ngram", default="hexagrams", help="fitness n-gram model (hexagrams/quintgrams/quadgrams)"
    )
    p.add_argument("--seed", type=int, help="RNG seed")
    p.add_argument("--top", type=int, default=5, help="number of top results to report")
    p.set_defaults(func=_cmd_joint)

    # anagram — pure columnar transposition (multiple anagramming), no substitution
    p = sub.add_parser(
        "anagram",
        help="recover a pure columnar transposition (multiple anagramming, no substitution)",
    )
    add_io(p)
    p.add_argument("--min-width", type=int, default=4, help="min columnar width to search")
    p.add_argument("--max-width", type=int, default=17, help="max columnar width to search")
    p.set_defaults(func=_cmd_anagram)

    # quagmire — blind periodic polyalphabetic with keyed alphabets (Quagmire I-IV/Vigenere)
    p = sub.add_parser(
        "quagmire",
        help="blind Quagmire I-IV / Vigenere / Beaufort solver (SA over keyed alphabet + shifts)",
    )
    add_io(p)
    p.add_argument(
        "--kind",
        default="quagmire3",
        choices=("vigenere", "beaufort", "quagmire1", "quagmire2", "quagmire3", "quagmire4"),
        help="cipher variant to solve",
    )
    p.add_argument("--min-period", type=int, default=2, help="min key period to search")
    p.add_argument("--max-period", type=int, default=20, help="max key period to search")
    p.add_argument("--restarts", type=int, default=30, help="SA restarts per period")
    p.add_argument("--seed", type=int, help="RNG seed")
    p.add_argument(
        "--alphabet",
        help="known keyed alphabet or keyword (e.g. KRYPTOS): fast fixed-alphabet "
        "solve that recovers only the shifts, skipping the alphabet annealing",
    )
    p.add_argument("--ct-alphabet", help="second keyed alphabet for Quagmire IV (with --alphabet)")
    p.set_defaults(func=_cmd_quagmire)

    # homophonic — simple/homophonic substitution via simulated annealing on hexagram fitness
    p = sub.add_parser(
        "homophonic",
        help="simple/homophonic substitution solver (SA on entropy-normalized hexagrams)",
    )
    add_io(p)
    p.add_argument("--restarts", type=int, default=30, help="SA restarts")
    p.add_argument("--iters", type=int, default=20000, help="SA iterations per restart")
    p.set_defaults(func=_cmd_homophonic)

    # auto
    p = sub.add_parser("auto", help="identify and crack across all ciphers")
    add_io(p)
    p.add_argument("--top", type=int, default=5, help="max candidates to return")
    p.add_argument("--seed", type=int, help="RNG seed")
    p.add_argument("--timeout", type=float, default=5.0, help="per-cipher time budget (s)")
    p.add_argument("--ciphers", help="comma-separated subset of ciphers to try")
    p.add_argument("--lang", choices=LANGUAGES, default="english", help="scoring language")
    p.add_argument(
        "--ngrams",
        choices=NGRAM_MODELS,
        default="quadgrams",
        help="fitness model (quintgrams sharper but needs a built table; falls back to quadgrams)",
    )
    p.add_argument(
        "--crib", help="known/guessed plaintext word; confirms & ranks candidates that contain it"
    )
    p.set_defaults(func=_cmd_auto)

    # pipeline (chain decode/encode steps for a layered cipher)
    p = sub.add_parser("pipeline", help="chain decode (or encode) steps for a layered cipher")
    p.add_argument(
        "--step",
        action="append",
        required=True,
        metavar="CIPHER:KEY",
        help="a 'cipher:key' step, in order (repeat); decryption order for --decode",
    )
    p.add_argument("--encode", action="store_true", help="encode each step instead of decode")
    add_io(p)
    p.set_defaults(func=_cmd_pipeline)

    # transform (pre-flight un-wraps: reverse / nested-encoding peel / decimate)
    p = sub.add_parser("transform", help="undo format wrappers (reverse, base64/hex/A1Z26, nulls)")
    add_io(p)
    p.add_argument(
        "--decimate", metavar="PERIOD[:OFFSET]", help="drop every PERIOD-th letter at OFFSET"
    )
    p.set_defaults(func=_cmd_transform)

    # crib (crib-drag + crib-anchored solvers: Vigenere family, keyed, autokey, product)
    p = sub.add_parser(
        "crib", help="crib-drag / crib-anchored solvers (vigenere, keyed, autokey, product)"
    )
    p.add_argument("--crib", required=True, help="guessed plaintext fragment (>= 4 letters)")
    add_io(p)
    p.add_argument("--top", type=int, default=6, help="placements to show per cipher")
    p.add_argument(
        "--keyed", action="store_true", help="crib-drag in a keyed alphabet (Quagmire family)"
    )
    p.add_argument("--autokey", action="store_true", help="plaintext-autokey crib-unzip")
    p.add_argument(
        "--product",
        action="store_true",
        help="two-coprime-keystream product crib solver (union-find)",
    )
    p.add_argument("--pos", type=int, help="known crib position (product exact solve)")
    p.add_argument("--p", type=int, help="product period p (with --pos)")
    p.add_argument("--q", type=int, help="product period q (with --pos)")
    p.add_argument("--p-range", type=int, nargs=2, metavar=("LO", "HI"), help="product p range")
    p.add_argument("--q-range", type=int, nargs=2, metavar=("LO", "HI"), help="product q range")
    p.add_argument("--alphabet", default="KRYPTOS", help="keyed alphabet for keyed/product")
    p.add_argument(
        "--inner-columnar",
        action="store_true",
        help="crib-anchor a periodic substitution UNDER a columnar transposition: jointly "
        "recover the column read-order AND the period-p Quagmire key by consistency "
        "backtracking (the --crib is the plaintext PREFIX). Sidesteps the flat blind objective.",
    )
    p.add_argument(
        "--widths", type=int, nargs="+", help="columnar widths to try (must divide length)"
    )
    p.add_argument(
        "--periods", type=int, nargs=2, metavar=("LO", "HI"), help="inner-sub period range"
    )
    p.set_defaults(func=_cmd_crib)

    # identify
    p = sub.add_parser("identify", help="classify likely cipher family/type (no decryption)")
    p.add_argument("--types", action="store_true", help="rank specific cipher types (statistical)")
    add_io(p)
    p.set_defaults(func=_cmd_identify)

    # diagnose (one-shot layered/composite structure triage + recommended attacks)
    p = sub.add_parser(
        "diagnose",
        help="triage a layered/composite cipher's structure and recommend concrete attacks",
    )
    add_io(p)
    p.set_defaults(func=_cmd_diagnose)

    # keyword (key/key-square finder)
    p = sub.add_parser("keyword", help="recover the keyword from a keyed alphabet or square")
    p.add_argument("--size", type=int, help="square size (5 or 6) to force square mode")
    p.add_argument(
        "--order",
        metavar="SPEC",
        help="invert a columnar READ-ORDER ('0,2,1,3') back to its keyword (with --wordlist) "
        "and/or a named generator (reverse, rotate-k, riffle, ...).",
    )
    p.add_argument("--wordlist", help="candidate keywords for --order inversion")
    add_io(p)
    p.set_defaults(func=_cmd_keyword)

    # split (separate multiple ciphers)
    p = sub.add_parser("split", help="split a file of multiple ciphers into entries")
    add_io(p)
    p.set_defaults(func=_cmd_split)

    # stats
    p = sub.add_parser("stats", help="frequency / IoC / digraph / Kasiski analysis")
    p.add_argument("--contacts", action="store_true", help="add variety-of-contacts (vowel finder)")
    p.add_argument(
        "--significance",
        action="store_true",
        help=(
            "length-aware coset-period significance (z vs matched null; flags small-sample periods)"
        ),
    )
    p.add_argument(
        "--family",
        action="store_true",
        help="look-elsewhere-corrected significance of the strongest period (max-z over the whole "
        "period grid vs shuffles) — a per-period z that doesn't clear it is multiplicity noise",
    )
    p.add_argument(
        "--family-samples",
        type=int,
        default=200,
        help="shuffles for the --family null (default 200)",
    )
    add_io(p)
    p.set_defaults(func=_cmd_stats)

    # relation
    p = sub.add_parser(
        "relation",
        help="detect a linear relation among n-gram positions (homophonic-expansion / "
        "tri-square family)",
    )
    p.add_argument("--n", type=int, default=3, help="n-gram size (default 3)")
    p.add_argument(
        "--alphabets", default="KRYPTOS,STD", help="comma-separated keyed alphabets to test"
    )
    p.add_argument("--samples", type=int, default=2000, help="shuffle-null replicates")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--top", type=int, default=8, help="candidates to report")
    add_io(p)
    p.set_defaults(func=_cmd_relation)

    # channel (width-parameterized Hill / linear-channel detector, matched-null + surjective guard)
    p = sub.add_parser(
        "channel",
        help="detect a leaked linear (Hill) channel at a given block width, "
        "calibrated against a matched-count shuffle null (honest: reports INCONCLUSIVE "
        "where a wide-block search is not exhaustive, so a null is never mis-read as 'not Hill')",
    )
    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[2, 3],
        help="block widths to probe (default 2 3). A w-tap probe is BLIND to a wider Hill.",
    )
    p.add_argument("--alphabet", default="KRYPTOS", help="keyed alphabet for the index space")
    p.add_argument("--samples", type=int, default=200, help="matched-null shuffle replicates")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    add_io(p)
    p.set_defaults(func=_cmd_channel)

    # subfrac (decoupled substitution-over-fractionation crack: key is FREE given the square)
    p = sub.add_parser(
        "subfrac",
        help="crack a periodic substitution OVER a bifid fractionation by scanning the inner "
        "SQUARE and recovering the outer key for free (drop-letter-free residual + coordinate "
        "descent). Objectives cover prose AND non-prose (route/coordinate) payloads.",
    )
    p.add_argument("--alphabet", default="KRYPTOS", help="outer substitution keyed alphabet")
    p.add_argument("--inner-period", type=int, default=7, help="inner bifid seriation period")
    p.add_argument(
        "--outer-period",
        type=int,
        default=None,
        help="outer substitution key period (default=inner)",
    )
    p.add_argument(
        "--squares",
        default="dictionary",
        help="'dictionary' (full wordlist scan) or a comma-separated list of square keywords",
    )
    p.add_argument(
        "--objective",
        default="fitness",
        choices=("fitness", "ioc", "repeats"),
        help=(
            "key-recovery objective: fitness=quadgram (prose), ioc/repeats=payload-agnostic (route)"
        ),
    )
    p.add_argument(
        "--drop-letter",
        default=None,
        help="omitted square letter (default J); a letter, comma-list, or 'sweep' for all 26",
    )
    p.add_argument("--langs", default="english", help="comma-separated scoring languages")
    p.add_argument("--top", type=int, default=5, help="candidates to report")
    p.add_argument(
        "--timeout", type=float, default=None, help="seconds budget for a dictionary scan"
    )
    add_io(p)
    p.set_defaults(func=_cmd_subfrac)

    # subplayfair / subserpf / subfoursq / subtwosq — the digraphic siblings of subfrac
    def _add_common_sub(pp, *, drop_help, default_drop="J"):
        pp.add_argument("--alphabet", default="KRYPTOS", help="outer substitution keyed alphabet")
        pp.add_argument("--outer-period", type=int, default=7, help="outer substitution key period")
        pp.add_argument(
            "--squares",
            default="dictionary",
            help="'dictionary' or a comma-separated list of square keywords",
        )
        pp.add_argument(
            "--objective",
            default="fitness",
            choices=("fitness", "ioc", "repeats"),
            help="key-recovery objective (fitness=prose; ioc/repeats=payload-agnostic)",
        )
        pp.add_argument("--drop-letter", default=None, help=drop_help)
        pp.add_argument("--langs", default="english", help="comma-separated scoring languages")
        pp.add_argument("--top", type=int, default=5, help="candidates to report")
        pp.add_argument("--timeout", type=float, default=None, help="seconds budget for the scan")
        add_io(pp)

    p = sub.add_parser(
        "subplayfair",
        help="crack a periodic substitution OVER a Playfair inner by scanning the square and "
        "recovering the outer key for free (drop-letter-pruned descent).",
    )
    _add_common_sub(
        p, drop_help="omitted square letter (default J); a letter, comma-list, or 'sweep'"
    )
    p.add_argument(
        "--shape",
        default="5x5",
        help="inner grid shape(s), comma-separated RxC (default 5x5). 26-cell shapes like 2x13 "
        "keep all 26 letters (no drop). E.g. '2x13,13x2'.",
    )
    p.set_defaults(func=_cmd_subplayfair)

    p = sub.add_parser(
        "subserpf",
        help="crack a periodic substitution OVER a Seriated-Playfair inner (odd-length capable, "
        "natively period-N).",
    )
    p.add_argument("--inner-period", type=int, default=7, help="inner seriation period")
    _add_common_sub(
        p, drop_help="omitted square letter (default J); a letter, comma-list, or 'sweep'"
    )
    p.set_defaults(func=_cmd_subserpf)

    p = sub.add_parser(
        "subfoursq",
        help="crack a periodic substitution OVER a Four-square inner (a PAIR of keyed squares — "
        "the paired-thematic-word signature).",
    )
    p.add_argument(
        "--bl-squares", default=None, help="bottom-left square list (default: reuse --squares)"
    )
    _add_common_sub(
        p, drop_help="omitted square letter (default Q, four-square convention); or 'sweep'"
    )
    p.set_defaults(func=_cmd_subfoursq)

    p = sub.add_parser(
        "subtwosq",
        help="crack a periodic substitution OVER a Two-square (double Playfair) inner (a pair of "
        "keyed squares, vertical/horizontal).",
    )
    p.add_argument(
        "--bot-squares", default=None, help="bottom/right square list (default: reuse --squares)"
    )
    p.add_argument(
        "--layout",
        default="both",
        choices=("vertical", "horizontal", "both"),
        help="two-square layout to try",
    )
    _add_common_sub(
        p, drop_help="omitted square letter (default Q, two-square convention); or 'sweep'"
    )
    p.set_defaults(func=_cmd_subtwosq)

    # convert
    p = sub.add_parser("convert", help="convert letters <-> A1Z26 numbers")
    p.add_argument(
        "--to",
        choices=("numbers", "pairs", "letters"),
        default="numbers",
        help="numbers (1-26), pairs (01-26), or letters (numbers->letters)",
    )
    p.add_argument("--divider", default=" ", help="separator between numbers")
    p.add_argument("--group", type=int, help="group output into blocks of N")
    add_io(p)
    p.set_defaults(func=_cmd_convert)

    # words
    p = sub.add_parser("words", help="dictionary search: match/pattern/anagram/ngram")
    p.add_argument(
        "op",
        choices=("match", "pattern", "anagram", "ngram"),
        help="match (?/. wildcards) | pattern (isomorph) | anagram | ngram (substring)",
    )
    p.add_argument("query", help="the word/template/letters/sequence to search")
    p.add_argument("--limit", type=int, default=200, help="max results")
    p.add_argument("-j", "--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_words)

    # format
    p = sub.add_parser("format", help="regroup / strip / recase text")
    p.add_argument("--group", type=int, help="group into blocks of N")
    p.add_argument("--strip", action="store_true", help="remove all whitespace")
    p.add_argument("--letters-only", action="store_true", help="keep only A-Z when grouping")
    p.add_argument("--case", choices=("upper", "lower"), help="recase output")
    add_io(p)
    p.set_defaults(func=_cmd_format)

    # solve (batch)
    p = sub.add_parser("solve", help="NDJSON batch: one request per line")
    p.add_argument("--batch", help="NDJSON file ('-'/omit for stdin)")
    p.set_defaults(func=_cmd_solve, json=True)

    # list
    p = sub.add_parser("list", help="list available ciphers")
    p.add_argument("-v", "--verbose", action="store_true", help="show each cipher's key format")
    p.add_argument("-j", "--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_list)

    # help (per-cipher details incl. key format + example)
    p = sub.add_parser("help", help="show a cipher's key format and usage")
    p.add_argument("cipher", help="cipher name (see `butt list`)")
    p.add_argument("-j", "--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_help)

    # schema (machine-readable capability manifest)
    p = sub.add_parser("schema", help="machine-readable capability manifest (JSON)")
    p.add_argument("--compact", action="store_true", help="single-line JSON")
    p.set_defaults(func=_cmd_schema, json=True)

    return parser


def _emit_error(exc: Exception, args) -> int:
    """Turn an expected error into a clean envelope instead of a traceback."""
    message = str(exc).strip("'\"")
    if getattr(args, "json", False):
        print(
            json.dumps(
                {"ok": False, "error": message, "error_type": type(exc).__name__},
                ensure_ascii=False,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError, FileNotFoundError, IsADirectoryError, PermissionError) as exc:
        return _emit_error(exc, args)


if __name__ == "__main__":
    raise SystemExit(main())
