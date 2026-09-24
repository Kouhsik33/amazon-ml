"""Indic-script -> Latin romanisation, implemented as pure code (no external data).

The nine Brahmic blocks Unicode encodes for India share one ISCII-derived
layout: Devanagari starts at U+0900, Bengali U+0980, Gurmukhi U+0A00, Gujarati
U+0A80, Oriya U+0B00, Tamil U+0B80, Telugu U+0C00, Kannada U+0C80, Malayalam
U+0D00 -- and within a block the same offset means the same phoneme. So a
single Devanagari offset table romanises all nine by folding the codepoint
back to the U+0900 block.

The romanisation is deliberately phonetic-and-loose: it feeds fuzzy string
comparison, not a reader. Schwa handling is the ambiguous part (Devanagari
writes 'international' as i-n-ta-ra-ne-sha-na-la), so ``consonant_skeleton``
is also exposed for callers that want to compare on consonants alone.
"""
from __future__ import annotations

import re
import unicodedata

_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)

# offset within the block -> latin
_OFF = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # dependent vowel signs (matras)
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
    0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai",
    0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "",   # virama / halant: suppresses the inherent vowel
    0x3C: "",   # nukta
    0x5F: "",
    # block digits -> ascii digits
    0x66: "0", 0x67: "1", 0x68: "2", 0x69: "3", 0x6A: "4",
    0x6B: "5", 0x6C: "6", 0x6D: "7", 0x6E: "8", 0x6F: "9",
}

# consonants carry an inherent 'a' unless a virama or matra follows
_CONSONANTS = set(range(0x15, 0x3A))
_MATRAS = set(range(0x3E, 0x4D)) | {0x4D, 0x3C}

_INDIC_RE = re.compile(r"[ऀ-ൿ]")


def has_indic(text: str) -> bool:
    return bool(_INDIC_RE.search(text))


def romanize(text: str) -> str:
    """Romanise any Indic characters in ``text``; leave everything else alone."""
    if not has_indic(text):
        return text
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        cp = ord(ch)
        block = None
        for base in _BLOCKS:
            if base <= cp < base + 0x80:
                block = base
                break
        if block is None:
            out.append(ch)
            continue
        off = cp - block
        mapped = _OFF.get(off)
        if mapped is None:
            continue
        out.append(mapped)
        if off in _CONSONANTS:
            # inherent schwa, unless the next char kills or replaces it
            nxt = ord(text[i + 1]) - block if i + 1 < n else -1
            if not (0 <= nxt < 0x80 and nxt in _MATRAS):
                out.append("a")
    return "".join(out)


_VOWELS = re.compile(r"[aeiou]+")


def consonant_skeleton(text: str) -> str:
    """Drop vowels -- robust to the schwa ambiguity and to vowel typos."""
    return _VOWELS.sub("", text)


def strip_accents(text: str) -> str:
    """Remove combining marks: 'Bryánt' -> 'Bryant', 'Chúrch' -> 'Church'.

    The generator sprinkles single diacritics into otherwise ASCII names, and
    French records carry real ones; both fold to the same base letters.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    )
