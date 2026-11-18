"""Where the bytes are in a flat image: fill runs, programmed regions, strings, repeats.

Nothing here knows the target. A fill run is a long run of one byte value (0xFF on erased
flash, 0x00 or 0xFF as padding in other images); the rest are the regions worth reading.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class AsciiMagic:
    offset: int
    text: str


@dataclass(frozen=True)
class Copy:
    offset: int
    same_as: int
    length: int


@dataclass(frozen=True)
class StringHit:
    offset: int
    text: str


def entropy(buf: bytes) -> float:
    """Shannon entropy in bits per byte: near 8 is compressed or encrypted, low is code, text or tables."""
    if not buf:
        return 0.0
    total = len(buf)
    return -sum(n / total * math.log2(n / total) for n in Counter(buf).values())


def fill_spans(data: bytes, fill: tuple[int, ...] = (0xFF,), min_len: int = 4096) -> tuple[Span, ...]:
    """Runs of one fill byte at least `min_len` long, in offset order."""
    spans: list[Span] = []
    for byte in sorted(set(fill)):
        pattern = re.compile(re.escape(bytes([byte])) + b"{%d,}" % min_len)
        spans += [Span(m.start(), m.end() - m.start()) for m in pattern.finditer(data)]
    return tuple(sorted(spans, key=lambda span: span.offset))


def programmed_islands(size: int, gaps: tuple[Span, ...]) -> tuple[Span, ...]:
    """The complement of `gaps` (sorted, non-overlapping) in [0, size)."""
    islands: list[Span] = []
    cursor = 0
    for span in gaps:
        if span.offset > cursor:
            islands.append(Span(cursor, span.offset - cursor))
        cursor = max(cursor, span.end)
    if cursor < size:
        islands.append(Span(cursor, size - cursor))
    return tuple(islands)


def _letter(byte: int) -> bool:
    return 65 <= byte <= 90 or 97 <= byte <= 122


def find_ascii_magics(data: bytes, islands: tuple[Span, ...]) -> tuple[AsciiMagic, ...]:
    """Four ASCII letters at the start of an island, followed by a non-letter."""
    found: list[AsciiMagic] = []
    for island in islands:
        if island.length < 5:
            continue
        word = data[island.offset:island.offset + 4]
        if all(_letter(byte) for byte in word) and not _letter(data[island.offset + 4]):
            found.append(AsciiMagic(island.offset, word.decode("ascii")))
    return tuple(found)


def find_copies(data: bytes, islands: tuple[Span, ...]) -> tuple[Copy, ...]:
    """Islands of 256 bytes or more that repeat an earlier island byte for byte."""
    seen: dict[str, int] = {}
    copies: list[Copy] = []
    for island in islands:
        if island.length < 256:
            continue
        digest = hashlib.sha1(data[island.offset:island.end]).hexdigest()
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = island.offset
        else:
            copies.append(Copy(island.offset, previous, island.length))
    return tuple(copies)


def _keep_string(text: str) -> bool:
    if not any(ch.isalpha() for ch in text):
        return False
    if "=" in text or "/" in text or " " in text:
        return True
    if any(ch.isdigit() for ch in text) and len(text) >= 6:
        return True
    return len(text) >= 8 and all(ch.isalnum() or ch == "_" for ch in text)


def _string_rank(text: str) -> int:
    # Paths and assignments identify a region; bare identifiers are the long tail.
    if "=" in text or "/" in text:
        return 0
    if " " in text:
        return 1
    if any(ch.isdigit() for ch in text):
        return 2
    return 3


_PRINTABLE = re.compile(rb"[\x20-\x7e]{6,}")


def find_strings(data: bytes, islands: tuple[Span, ...], *, limit: int = 40) -> tuple[StringHit, ...]:
    """The most telling NUL- or newline-terminated strings of each island, at most `limit` per island."""
    hits: list[StringHit] = []
    for island in islands:
        blob = data[island.offset:island.end]
        # Compressed payloads are nearly 8 bits/byte and their "strings" are noise.
        if len(blob) > 4096 and entropy(blob) > 7.5:
            blob = blob[:512]
        found: list[StringHit] = []
        for m in _PRINTABLE.finditer(blob):
            if m.end() < len(blob) and blob[m.end()] not in (0, 10):
                continue
            text = m.group().decode("ascii")
            if _keep_string(text) and len(text) <= 160:
                found.append(StringHit(island.offset + m.start(), text))
        found.sort(key=lambda hit: (_string_rank(hit.text), hit.offset))
        hits.extend(sorted(found[:limit], key=lambda hit: hit.offset))
    return tuple(hits)
