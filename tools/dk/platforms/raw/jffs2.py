"""JFFS2 node chains: directory names, inode fragments, and the files they rebuild.

Magic 0x1985 in the image's byte order (little- or big-endian). A hit counts only as a
chain of nodes with known types and lengths that stay in the image.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

_MAGIC = 0x1985
_TYPES = {0x2003, 0x2004, 0x2006, 0xE001, 0xE002, 0xE008, 0xE009}
_DIRENT = 0xE001
_INODE = 0xE002
_KINDS = {4: "dir", 8: "file", 10: "link"}


def crc(buf: bytes) -> int:
    """JFFS2's crc32: the zlib polynomial seeded with 0 and not inverted at the end (not zlib.crc32)."""
    return zlib.crc32(buf, 0xFFFFFFFF) ^ 0xFFFFFFFF


@dataclass(frozen=True)
class Dirent:
    offset: int             # file offset of the name bytes
    ino: int
    pino: int
    version: int
    kind: str               # "file", "dir", "link", or the raw type number
    name: str


@dataclass(frozen=True)
class InodeFrag:
    ino: int
    version: int
    file_offset: int
    compr: int
    csize: int
    dsize: int
    data_at: int            # file offset of the csize payload bytes


@dataclass(frozen=True)
class Jffs2:
    offset: int
    end: int
    endian: str             # "le" or "be"
    nodes: int
    bad_crc: int            # node headers whose hdr_crc is not crc() of their first 8 bytes
    dirents: tuple[Dirent, ...]
    frags: tuple[InodeFrag, ...]


@dataclass(frozen=True)
class Jffs2File:
    path: str
    data: bytes


def _parse(data: bytes, offset: int, e: str) -> Jffs2 | None:
    nodes = bad = 0
    dirents: list[Dirent] = []
    frags: list[InodeFrag] = []
    cursor = offset
    while cursor + 12 <= len(data) and nodes < 100000:
        magic, nodetype, totlen, hdr_crc = struct.unpack_from(e + "HHII", data, cursor)
        if magic != _MAGIC or nodetype not in _TYPES or totlen < 12 or cursor + totlen > len(data):
            break
        bad += hdr_crc != crc(data[cursor:cursor + 8])
        if nodetype == _DIRENT:
            nsize = data[cursor + 28] if cursor + 41 <= len(data) else 0
            name = data[cursor + 40:cursor + 40 + nsize]
            if totlen < 41 or not 1 <= nsize <= 128 or 40 + nsize > totlen or not all(32 <= b < 127 for b in name):
                break
            pino, version, ino = struct.unpack_from(e + "III", data, cursor + 12)
            kind = _KINDS.get(data[cursor + 29], str(data[cursor + 29]))
            dirents.append(Dirent(cursor + 40, ino, pino, version, kind, name.decode("ascii")))
        elif nodetype == _INODE and totlen >= 68:
            ino, version = struct.unpack_from(e + "II", data, cursor + 12)
            file_offset, csize, dsize = struct.unpack_from(e + "III", data, cursor + 44)
            if 68 + csize <= totlen and csize < 0x1000000 and dsize < 0x1000000:
                frags.append(InodeFrag(ino, version, file_offset, data[cursor + 56], csize, dsize, cursor + 68))
        nodes += 1
        cursor += (totlen + 3) & ~3
    if nodes < 2 or not dirents:
        return None
    return Jffs2(offset, cursor, "le" if e == "<" else "be", nodes, bad, tuple(dirents), tuple(frags))


def find_jffs2(data: bytes) -> tuple[Jffs2, ...]:
    found: list[Jffs2] = []
    for needle, e in ((b"\x85\x19", "<"), (b"\x19\x85", ">")):
        covered = start = 0
        while (hit := data.find(needle, start)) >= 0:
            start = hit + 1
            if hit < covered or hit % 4:
                continue
            parsed = _parse(data, hit, e)
            if parsed is not None:
                found.append(parsed)
                covered = start = parsed.end
    return tuple(sorted(found, key=lambda fs: fs.offset))


def _inflate(data: bytes, frag: InodeFrag) -> bytes | None:
    blob = data[frag.data_at:frag.data_at + frag.csize]
    if frag.compr == 0:  # JFFS2_COMPR_NONE
        out = blob
    elif frag.compr == 6:  # JFFS2_COMPR_ZLIB
        try:
            out = zlib.decompress(blob)
        except zlib.error:
            return None
    else:
        return None
    return out if len(out) == frag.dsize else None


def _latest(dirents: tuple[Dirent, ...]) -> dict[int, Dirent]:
    by_ino: dict[int, Dirent] = {}
    for dent in dirents:
        previous = by_ino.get(dent.ino)
        if previous is None or dent.version >= previous.version:
            by_ino[dent.ino] = dent
    return by_ino


def _path(by_ino: dict[int, Dirent], ino: int) -> str | None:
    parts: list[str] = []
    seen: set[int] = set()
    while ino not in (0, 1):
        dent = by_ino.get(ino)
        if dent is None or ino in seen or "/" in dent.name or dent.name in (".", ".."):
            return None
        seen.add(ino)
        parts.append(dent.name)
        ino = dent.pino
    return "/".join(reversed(parts)) or None


def files(data: bytes, filesystems: tuple[Jffs2, ...]) -> tuple[Jffs2File, ...]:
    """Files whose inode fragments are raw or zlib, rebuilt in file-offset order."""
    found: list[Jffs2File] = []
    for fs in filesystems:
        by_ino = _latest(fs.dirents)
        grouped: dict[int, list[InodeFrag]] = {}
        for frag in fs.frags:
            grouped.setdefault(frag.ino, []).append(frag)
        for ino in sorted(grouped):
            dent = by_ino.get(ino)
            path = _path(by_ino, ino) if dent is not None and dent.kind == "file" else None
            if not path:
                continue
            parts: list[bytes] = []
            cursor = 0
            for frag in sorted(grouped[ino], key=lambda item: (item.file_offset, item.version)):
                raw = _inflate(data, frag) if frag.file_offset == cursor else None
                if raw is None:
                    parts = []
                    break
                parts.append(raw)
                cursor += len(raw)
            if parts:
                found.append(Jffs2File(path, b"".join(parts)))
    return tuple(sorted(found, key=lambda item: item.path))
