"""Cipher-file splitter — buttcrack's answer to CryptoCrack's "Separate Cipher Files".

Real-world cipher collections (e.g. an ACA issue) pack many ciphers into one
file, separated by blank lines and/or a label/title line (``A-1.``, ``1.``,
``K1``, or a SHOUTING title in caps). This module breaks such text back into the
individual entries so each can be fed to a solver on its own.

Public API:
    split_ciphers(text) -> list[dict]   # {"title": str | None, "body": str}
"""

from __future__ import annotations

import re

# A leading label such as "A-1.", "1.", "12)", "K1", "C-13:" — a short token of
# letters/digits/hyphens optionally followed by a separator (``.``/``)``/``:``)
# and then (optionally) the body on the same line.
_LABEL_RE = re.compile(
    r"""
    ^\s*
    (?P<label>
        [A-Za-z]{0,3}        # optional letter prefix (A, K, BB, ...)
        -?
        \d{1,4}              # the number
        [A-Za-z]{0,3}        # optional letter suffix
      | [A-Za-z]{1,3}\d{1,4} # or letter-then-number with no hyphen (K1, C13)
    )
    \s*[.):]\s*              # a required separator after the label
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)


def _split_blocks(text: str) -> list[list[str]]:
    """Split into blocks on runs of blank lines, dropping leading/trailing blanks."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in text.splitlines():
        if raw.strip():
            current.append(raw)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _is_shouting_title(line: str) -> bool:
    """A heading line: all-caps prose, distinguishable from grouped ciphertext.

    ACA ciphertext is usually rendered in uniform all-caps groups (e.g. 5-letter
    blocks), so a line of equal-length tokens is treated as ciphertext, not a
    heading. A title reads like prose: multiple words of *varying* length (or at
    least one short word), none absurdly long.
    """
    stripped = line.strip()
    if not stripped or any(ch.islower() for ch in stripped):
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    words = stripped.split()
    if len(words) < 2 or any(len(w) > 12 for w in words):
        return False
    # Grouped ciphertext has uniform token lengths; prose titles vary. Require
    # either differing word lengths or a short connective word (THE, OF, A, ...).
    lengths = {len(w) for w in words}
    return len(lengths) > 1 or any(len(w) <= 3 for w in words)


def _normalize_body(lines: list[str]) -> str:
    """Join body lines with single spaces, collapsing internal runs of whitespace.

    Preserves the ciphertext characters and their order; only whitespace between
    tokens is normalized so downstream tools see a clean single-spaced block.
    """
    tokens: list[str] = []
    for line in lines:
        tokens.extend(line.split())
    return " ".join(tokens)


def _split_entry(block: list[str]) -> tuple[str | None, list[str]]:
    """Pull a title off the front of a block, returning (title, body_lines)."""
    first = block[0]

    match = _LABEL_RE.match(first)
    if match is not None:
        label = match.group("label").strip()
        rest = match.group("rest").strip()
        if rest:
            # Label sits on the same line as the start of the body.
            return label, [rest, *block[1:]]
        # Label is on its own line; body follows.
        return label, block[1:]

    if len(block) > 1 and _is_shouting_title(first):
        return first.strip(), block[1:]

    return None, block


def split_ciphers(text: str) -> list[dict[str, str | None]]:
    """Split a multi-cipher text into individual entries.

    Each returned dict has ``title`` (a label/heading if one was recognized,
    otherwise ``None``) and ``body`` (the normalized ciphertext block).

    Entries are separated by blank lines. Within an entry, a leading label
    (``A-1.``, ``1.``, ``K1``) or a SHOUTING title line is split off as the
    title. When no titles are found anywhere, a numbered fallback is applied so
    every entry still carries an identifier.

    Empty or whitespace-only input returns ``[]``.
    """
    blocks = _split_blocks(text)
    if not blocks:
        return []

    entries: list[dict[str, str | None]] = []
    for block in blocks:
        title, body_lines = _split_entry(block)
        body = _normalize_body(body_lines)
        if not body and title is not None:
            # The block was a lone label/heading with no following ciphertext;
            # treat that text as the body so nothing is silently dropped.
            body, title = title, None
        entries.append({"title": title, "body": body})

    # Numbered fallback: if nothing got a title, label them 1..N.
    if all(entry["title"] is None for entry in entries):
        for index, entry in enumerate(entries, start=1):
            entry["title"] = str(index)

    return entries
