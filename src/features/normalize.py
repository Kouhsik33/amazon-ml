"""Normalisation for business names, addresses and country labels.

Design rules this module follows:

* **Do not over-normalise.** Street numbers, unit numbers, PIN codes and rare
  name tokens carry the identity signal; they are preserved verbatim. Only
  tokens the data shows to be *unstable across true matches* are folded away.
* **Stay country-agnostic.** No ``if country == "India"`` branch exists. The
  alias/noise tables are mined from data (see ``mine_aliases.py``) and an
  unseen country simply contributes no aliases, falling back to generic
  mechanisms (romanisation, accent folding, prefix-abbreviation matching).
* **Emit several views, not one.** A single "canonical string" throws away
  signal. Callers get the full normalised form, a legal-suffix-free core, a
  compact (space-free) form that catches ``carneybryant.com``, a sorted-token
  key that catches word-order transposition, and the numeric tokens.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

from features.translit import has_indic, romanize, strip_accents

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# --------------------------------------------------------------------------
# Generic patterns
# --------------------------------------------------------------------------

_TLD = re.compile(r"\.(com|net|org|co|io|biz|info|in|us|fr|edu|gov)\b", re.I)
_NONALNUM = re.compile(r"[^0-9a-z]+")
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d")
_PURE_NUM = re.compile(r"^\d+$")

# literal placeholders the generator emits for a missing component
_NULLS = frozenset({"null", "none", "nan", "na", "n/a", "-", "--", "unknown"})


def _load(name: str, default):
    p = CONFIG_DIR / name
    if p.exists():
        return json.loads(p.read_text())
    return default


@dataclass
class Tables:
    """Mined, data-derived vocabulary. Empty tables => generic behaviour only."""

    name_noise: FrozenSet[str] = frozenset()      # legal suffixes & filler words
    addr_noise: FrozenSet[str] = frozenset()      # 'null', 'po', 'box', ...
    addr_alias: Dict[str, str] = field(default_factory=dict)   # 'ave' -> 'avenue'
    name_alias: Dict[str, str] = field(default_factory=dict)   # 'incorporated' -> 'inc'

    @classmethod
    def load(cls) -> "Tables":
        return cls(
            name_noise=frozenset(_load("name_noise.json", [])),
            addr_noise=frozenset(_load("addr_noise.json", [])),
            addr_alias=_load("addr_alias.json", {}),
            name_alias=_load("name_alias.json", {}),
        )


TABLES = Tables.load()


def reload_tables() -> None:
    global TABLES
    TABLES = Tables.load()


# --------------------------------------------------------------------------
# Country
# --------------------------------------------------------------------------

def normalize_country(value: str) -> str:
    """Casefold and squeeze a country label. Open set: any string is valid.

    No membership test, no mapping table, no filtering -- an unseen label such
    as 'France' passes through unchanged and compares equal only to itself.
    """
    if not value:
        return ""
    return _WS.sub(" ", strip_accents(value).strip().lower())


# --------------------------------------------------------------------------
# Shared text pipeline
# --------------------------------------------------------------------------

def _basic(text: str) -> str:
    """Romanise Indic, fold accents, lowercase, punctuation -> space."""
    if not text:
        return ""
    if has_indic(text):
        text = romanize(text)
    text = strip_accents(text).lower()
    text = text.replace("&", " and ")
    text = _NONALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _split_digit_letter(tokens: Sequence[str]) -> List[str]:
    """'23404b' -> ['23404','b'];  '2nd' stays '2nd' (handled by ordinals)."""
    out: List[str] = []
    for t in tokens:
        m = re.fullmatch(r"(\d+)([a-z]{1,2})", t)
        if m and m.group(2) not in _ORDINAL_SUFFIX:
            out.extend(m.groups())
        else:
            out.append(t)
    return out


_ORDINAL_SUFFIX = {"st", "nd", "rd", "th"}
_ORDINAL_WORD = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}


def _canon_ordinal(tok: str) -> str:
    """'2nd'/'second' -> '2'.  Street names alternate between the two forms."""
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok)
    if m:
        return m.group(1)
    return _ORDINAL_WORD.get(tok, tok)


# --------------------------------------------------------------------------
# Name
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NameView:
    full: str                 # normalised, all tokens
    core: str                 # noise tokens (legal suffixes etc.) removed
    compact: str              # core with spaces removed -> matches 'foobar.com'
    sorted_key: str           # sorted core tokens -> word-order invariant
    tokens: Tuple[str, ...]
    core_tokens: Tuple[str, ...]
    was_domain: bool


@lru_cache(maxsize=1 << 17)
def normalize_name(name: str) -> NameView:
    """Normalise a business name into several comparison views."""
    raw = name or ""

    # Domain-style names ('carneybryant.com', 'brightaut0body.com') are a whole
    # corruption class: strip the TLD and keep the run-together stem.
    was_domain = bool(_TLD.search(raw)) and " " not in raw.strip()
    if was_domain:
        raw = _TLD.sub(" ", raw)

    base = _basic(raw)
    tokens = [t for t in base.split() if t]
    tokens = [TABLES.name_alias.get(t, t) for t in tokens]

    core = [t for t in tokens if t not in TABLES.name_noise]
    if not core:                      # a name made only of legal words
        core = list(tokens)

    core_t = tuple(core)
    return NameView(
        full=" ".join(tokens),
        core=" ".join(core),
        compact="".join(core),
        sorted_key=" ".join(sorted(set(core))),
        tokens=tuple(tokens),
        core_tokens=core_t,
        was_domain=was_domain,
    )


# --------------------------------------------------------------------------
# Address
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AddressView:
    full: str
    core: str                      # noise/filler removed, aliases canonicalised
    tokens: Tuple[str, ...]
    core_tokens: Tuple[str, ...]
    numbers: Tuple[str, ...]       # every numeric token (street no., PIN, unit)
    alpha_tokens: Tuple[str, ...]  # non-numeric core tokens (street/city names)
    is_empty: bool


@lru_cache(maxsize=1 << 17)
def normalize_address(address: str) -> AddressView:
    """Normalise an address, preserving all numeric tokens."""
    base = _basic(address or "")
    tokens = [t for t in base.split() if t and t not in _NULLS]
    tokens = _split_digit_letter(tokens)
    tokens = [_canon_ordinal(t) for t in tokens]
    # canonicalise abbreviations both ways via the mined table ('ave'->'avenue')
    tokens = [TABLES.addr_alias.get(t, t) for t in tokens]

    core = [t for t in tokens if t not in TABLES.addr_noise]
    numbers = tuple(t for t in tokens if _PURE_NUM.fullmatch(t))
    alpha = tuple(t for t in core if not _DIGITS.search(t))

    return AddressView(
        full=" ".join(tokens),
        core=" ".join(core),
        tokens=tuple(tokens),
        core_tokens=tuple(core),
        numbers=numbers,
        alpha_tokens=alpha,
        is_empty=not tokens,
    )


def char_ngrams(text: str, n: int = 3) -> FrozenSet[str]:
    """Character n-grams of a space-free string, for cheap fuzzy overlap."""
    s = text.replace(" ", "")
    if len(s) < n:
        return frozenset({s}) if s else frozenset()
    return frozenset(s[i:i + n] for i in range(len(s) - n + 1))
