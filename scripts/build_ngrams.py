#!/usr/bin/env python3
"""Build English n-gram frequency tables from a text corpus.

Reads one or more plain-text files, strips everything down to the 26 uppercase
letters, counts n-grams for n in {1,2,3,4,5}, and writes ``english_<name>.txt``
files of ``NGRAM count`` lines (sorted by count, descending) into the data dir.

These tables power the fitness scoring used by the crackers. The quadgram table
is the workhorse for substitution hill-climbing; monograms drive chi-squared
scoring for Caesar/affine/Vigenere. The optional quintgram table sharpens the
fitness for the hardest searches (homophonic, long transpositions) but is large
and needs a big corpus to be well-populated. buttcrack bundles English mono..hexagrams;
other languages ship up to quadgrams, and the scorer falls back to quadgrams when a
requested higher-order table is not present. Build higher orders with --max-n 6.

Usage:
    python scripts/build_ngrams.py corpus/*.txt --out src/buttcrack/data
    # add a quintgram table for a richer fitness (large; needs a big corpus):
    python scripts/build_ngrams.py corpus/*.txt --max-n 5
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path

NGRAMS = {
    1: "monograms",
    2: "bigrams",
    3: "trigrams",
    4: "quadgrams",
    5: "quintgrams",
    6: "hexagrams",
}

# Drop n-grams seen fewer than this many times, to keep file sizes sane. Unseen
# n-grams are floored by the scorer anyway, so pruning the long rare tail is safe.
# Quint/hexagrams get a higher floor — there are far more of them and the rare tail is
# mostly noise — so the tables stay a manageable size.
MIN_COUNT = {1: 1, 2: 1, 3: 2, 4: 5, 5: 8, 6: 8}


def clean(text: str) -> str:
    """Fold accents to A-Z and uppercase (e.g. é->E, ñ->N), dropping the rest.

    Classical ciphers operate on the 26-letter Latin alphabet, so accented
    letters in non-English corpora are folded to their base letter.
    """
    folded = unicodedata.normalize("NFKD", text.upper())
    return "".join(ch for ch in folded if "A" <= ch <= "Z")


def count_ngrams(letters: str, n: int) -> Counter:
    counts: Counter = Counter()
    for i in range(len(letters) - n + 1):
        counts[letters[i : i + n]] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build English n-gram tables from a corpus.")
    parser.add_argument("files", nargs="+", type=Path, help="Corpus text files")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src" / "buttcrack" / "data",
        help="Output directory for <lang>_*.txt tables",
    )
    parser.add_argument("--lang", default="english", help="language prefix for output files")
    parser.add_argument(
        "--max-n",
        type=int,
        default=4,
        choices=sorted(NGRAMS),
        help="largest n-gram to build (5 = quintgrams; large, off by default)",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {len(args.files)} file(s)...", file=sys.stderr)
    chunks = []
    for path in args.files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        chunks.append(clean(raw))
    letters = "".join(chunks)
    print(f"Corpus: {len(letters):,} letters", file=sys.stderr)

    for n, name in NGRAMS.items():
        if n > args.max_n:
            continue
        counts = count_ngrams(letters, n)
        floor = MIN_COUNT[n]
        kept = [(g, c) for g, c in counts.items() if c >= floor]
        kept.sort(key=lambda gc: (-gc[1], gc[0]))
        out_path = args.out / f"{args.lang}_{name}.txt"
        with out_path.open("w", encoding="ascii") as fh:
            for gram, c in kept:
                fh.write(f"{gram} {c}\n")
        print(
            f"  {name:>10}: {len(kept):>8,} distinct (>= {floor})  -> {out_path.name}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
