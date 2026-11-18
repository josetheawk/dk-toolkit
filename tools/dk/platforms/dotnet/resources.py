"""Managed .resources files (System.Resources.ResourceReader format) and the files inside them."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from .pe import FormatError

MAGIC = 0xBEEFCACE

# Version 2 type codes (System.Resources.ResourceTypeCode) → names
_TYPE_CODES = {0: "null", 1: "string", 2: "bool", 3: "char", 4: "byte", 5: "sbyte", 6: "short", 7: "ushort",
               8: "int", 9: "uint", 10: "long", 11: "ulong", 12: "float", 13: "double", 14: "decimal",
               15: "DateTime", 16: "TimeSpan", 0x20: "byte[]", 0x21: "Stream"}
_FIXED = {2: "<?", 3: "<H", 4: "<B", 5: "<b", 6: "<h", 7: "<H", 8: "<i", 9: "<I", 10: "<q", 11: "<Q",
          12: "<f", 13: "<d", 15: "<q", 16: "<q"}


@dataclass(frozen=True)
class Entry:
    name: str
    type: str
    value: str | int | float | bool | None   # simple values, decoded
    raw: bytes                                # the value's bytes (serialized object, byte array, stream)


def _7bit(data: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if at >= len(data) or shift > 28:
            raise FormatError("bad 7-bit encoded length")
        b = data[at]
        at += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, at


def _string(data: bytes, at: int, encoding: str) -> tuple[str, int]:
    length, at = _7bit(data, at)
    if at + length > len(data):
        raise FormatError("string runs past the end")
    return data[at:at + length].decode(encoding, "replace"), at + length


def parse(data: bytes) -> list[Entry]:
    """Entries in data order. Raises FormatError if `data` isn't a .resources file."""
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != MAGIC:
        raise FormatError("not a .resources file")
    _, _, skip = struct.unpack_from("<III", data, 0)
    at = 12 + skip
    version, count, ntypes = struct.unpack_from("<III", data, at)
    at += 12
    types = []
    for _ in range(ntypes):
        name, at = _string(data, at, "utf-8")
        types.append(name)
    at = (at + 7) & ~7                          # "PAD" alignment
    at += 4 * count                             # name hashes
    positions = struct.unpack_from(f"<{count}I", data, at)
    at += 4 * count
    (data_start,) = struct.unpack_from("<I", data, at)
    names_start = at + 4
    located = []
    for pos in positions:
        name, p = _string(data, names_start + pos, "utf-16-le")
        located.append((data_start + struct.unpack_from("<I", data, p)[0], name))
    located.sort()
    entries = []
    for i, (offset, name) in enumerate(located):
        end = located[i + 1][0] if i + 1 < len(located) else len(data)
        entries.append(_entry(data, offset, end, name, version, types))
    return entries


def _entry(data: bytes, at: int, end: int, name: str, version: int, types: list[str]) -> Entry:
    code, at = _7bit(data, at)
    if version == 1:                            # v1: an index into the type table, then a serialized object
        type_name = types[code] if code < len(types) else f"type#{code}"
        return Entry(name, _short_type(type_name), None, data[at:end])
    if code == 1:
        text, _ = _string(data, at, "utf-8")
        return Entry(name, "string", text, b"")
    if code in _FIXED:
        (value,) = struct.unpack_from(_FIXED[code], data, at)
        return Entry(name, _TYPE_CODES[code], chr(value) if code == 3 else value, b"")
    if code in (0x20, 0x21):
        (length,) = struct.unpack_from("<i", data, at)
        return Entry(name, _TYPE_CODES[code], None, data[at + 4:at + 4 + length])
    if code >= 0x40:
        index = code - 0x40
        type_name = types[index] if index < len(types) else f"type#{index}"
        return Entry(name, _short_type(type_name), None, data[at:end])
    return Entry(name, _TYPE_CODES.get(code, f"code{code:#x}"), None, data[at:end])


def _short_type(assembly_qualified: str) -> str:
    return assembly_qualified.split(",")[0].strip()


# Files inside serialized objects

def byte_arrays(blob: bytes) -> list[bytes]:
    """Byte arrays held in a BinaryFormatter stream (ArraySinglePrimitive records of Byte)."""
    out, at = [], 0
    while True:
        at = blob.find(b"\x0f", at)
        if at < 0 or at + 10 > len(blob):
            return out
        _, length = struct.unpack_from("<ii", blob, at + 1)
        if blob[at + 9] == 0x02 and 0 < length <= len(blob) - (at + 10):
            out.append(blob[at + 10:at + 10 + length])
            at += 10 + length
        else:
            at += 1


def sniff(data: bytes) -> tuple[str, str]:
    """(file extension, short description) from a file's magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack_from(">II", data, 16)
        return "png", f"png {w}x{h}"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg", "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack_from("<HH", data, 6)
        return "gif", f"gif {w}x{h}"
    if data[:2] == b"BM":
        return "bmp", "bmp"
    if data[:4] == b"\0\0\1\0" and len(data) >= 6:
        return "ico", f"ico, {struct.unpack_from('<H', data, 4)[0]} image(s)"
    if data[:2] == b"MZ":
        return "bin", "PE executable (MZ header)"
    return "bin", "data"


def contained_files(entry: Entry) -> list[tuple[str, str, bytes]]:
    """(extension, description, bytes) for each file found in one entry's value."""
    if not entry.raw:
        return []
    arrays = [entry.raw] if entry.type in ("byte[]", "Stream") else byte_arrays(entry.raw)
    return [(*sniff(a), a) for a in arrays]


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """A file name from untrusted text: no separators, no leading dots, never empty."""
    cleaned = _UNSAFE.sub("_", name).lstrip(".")
    return cleaned[:120] or "_"
