"""Cipher registry — resolve names/aliases to cipher instances."""

from __future__ import annotations

from .ciphers import ALL_CIPHERS
from .ciphers.base import Cipher

_REGISTRY: dict[str, Cipher] = {}
_ORDER: list[Cipher] = []

for _cls in ALL_CIPHERS:
    _instance = _cls()
    _ORDER.append(_instance)
    for _key in (_instance.name, *_instance.aliases):
        _REGISTRY[_key.lower()] = _instance


def get(name: str) -> Cipher:
    """Resolve a cipher by name or alias."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"unknown cipher {name!r}; try `butt list`") from None


def names() -> list[str]:
    """Canonical cipher names in default order."""
    return [c.name for c in _ORDER]


def all_ciphers() -> list[Cipher]:
    """Every registered cipher instance, in default order."""
    return list(_ORDER)


def describe() -> list[dict]:
    """Metadata for every cipher (for `butt list` / `butt help`)."""
    out = []
    for c in _ORDER:
        out.append(
            {
                "name": c.name,
                "aliases": list(c.aliases),
                "needs_key": c.needs_key,
                "key_format": c.key_format,
                "key_example": c.key_example,
                "complexity": c.complexity,
                "description": c.description,
            }
        )
    return out


def describe_one(name: str) -> dict:
    """Full metadata for a single cipher (for `butt help <cipher>`)."""
    c = get(name)
    return {
        "name": c.name,
        "aliases": list(c.aliases),
        "needs_key": c.needs_key,
        "key_format": c.key_format,
        "key_example": c.key_example,
        "complexity": c.complexity,
        "auto_crackable": c.auto_crackable,
        "description": c.description,
    }
