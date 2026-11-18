"""Known containers and streams inside a flat image, each confirmed by its own structure.

A magic alone is not a hit: the header around it must check out (a CRC, sane fields, a
length that fits) or the stream must actually decode. Compressed streams are decoded with
the standard library to find where they end and how big they inflate.
"""

from __future__ import annotations

import bz2
import lzma
import struct
import zlib
from dataclasses import dataclass
from typing import Callable

from .layout import Span

_CHUNK = 1 << 20
_MAX_OUT = 512 << 20        # stop measuring a stream that inflates past this


@dataclass(frozen=True)
class Hit:
    offset: int             # first byte of the container or stream
    kind: str
    body: int               # first byte of what is worth carving (after a header, if any)
    end: int | None         # one past the last byte, when the structure says so
    note: str = ""


def _stream_end(data: bytes, start: int, decoder) -> tuple[int, int] | None:
    """(end offset, inflated size) of the stream at `start`, or None if it does not decode to its end."""
    pos = start
    out = 0
    try:
        while pos < len(data):
            chunk = data[pos:pos + _CHUNK]
            out += len(decoder.decompress(chunk))
            if decoder.eof:
                return pos + len(chunk) - len(decoder.unused_data), out
            if out > _MAX_OUT:
                return None
            pos += len(chunk)
    except Exception:  # each decoder has its own error type; any of them means "not this stream"
        return None
    return None


def _decoded(kind: str, data: bytes, at: int, make: Callable[[], object]) -> Hit | None:
    found = _stream_end(data, at, make())
    if found is None:
        return None
    end, size = found
    return Hit(at, kind, at, end, f"inflates to {size}")


def _gzip(data: bytes, at: int) -> Hit | None:
    flags, xfl, os_ = data[at + 3], data[at + 8], data[at + 9]
    if data[at + 2] != 8 or flags & 0xE0 or xfl not in (0, 2, 4) or not (os_ <= 13 or os_ == 255):
        return None
    return _decoded("gzip", data, at, lambda: zlib.decompressobj(31))


def _xz(data: bytes, at: int) -> Hit | None:
    flags = data[at + 6:at + 8]
    if len(data) < at + 12 or struct.unpack_from("<I", data, at + 8)[0] != zlib.crc32(flags):
        return None
    return _decoded("xz", data, at, lambda: lzma.LZMADecompressor(lzma.FORMAT_XZ))


def _bzip2(data: bytes, at: int) -> Hit | None:
    if not 0x31 <= data[at + 3] <= 0x39 or data[at + 4:at + 10] != b"\x31\x41\x59\x26\x53\x59":
        return None
    return _decoded("bzip2", data, at, bz2.BZ2Decompressor)


def _lzma_alone(data: bytes, at: int) -> Hit | None:
    """`.lzma` (LZMA-alone): props 0x5D, a power-of-two dictionary, then a size or all ones."""
    if len(data) < at + 13:
        return None
    dictionary, size = struct.unpack_from("<IQ", data, at + 1)
    if dictionary & (dictionary - 1) or not 1 << 16 <= dictionary <= 1 << 26:
        return None
    if size != (1 << 64) - 1 and size > 1 << 30:
        return None
    return _decoded("lzma", data, at, lambda: lzma.LZMADecompressor(lzma.FORMAT_ALONE))


def _zstd(data: bytes, at: int) -> Hit | None:
    try:
        from compression import zstd
    except ImportError:
        return Hit(at, "zstd", at, None, "no decoder (Python < 3.14)")
    return _decoded("zstd", data, at, zstd.ZstdDecompressor)


def _uimage(data: bytes, at: int) -> Hit | None:
    """U-Boot legacy image: 64-byte big-endian header whose CRC covers itself with the CRC zeroed."""
    if len(data) < at + 64:
        return None
    header = bytearray(data[at:at + 64])
    stored = struct.unpack_from(">I", header, 4)[0]
    header[4:8] = b"\0\0\0\0"
    if zlib.crc32(header) != stored:
        return None
    size, load, entry = struct.unpack_from(">III", header, 12)
    comp = {0: "none", 1: "gzip", 2: "bzip2", 3: "lzma", 4: "lzo", 5: "lz4", 6: "zstd"}.get(header[31], str(header[31]))
    name = header[32:64].split(b"\0", 1)[0].decode("ascii", "replace")
    end = at + 64 + size
    note = f"'{name}' load 0x{load:08x} entry 0x{entry:08x} comp {comp}"
    return Hit(at, "uimage", at + 64, end if end <= len(data) else None, note)


def _fdt(data: bytes, at: int) -> Hit | None:
    if len(data) < at + 40:
        return None
    total, off_struct, off_strings, _, version, last = struct.unpack_from(">IIIIII", data, at + 4)
    if not 16 <= version <= 17 or last > version or not 40 <= off_struct < total or off_strings >= total:
        return None
    if at + total > len(data):
        return None
    return Hit(at, "fdt", at, at + total, f"device tree v{version}")


_ELF_MACHINES = {2: "sparc", 3: "x86", 8: "mips", 20: "ppc", 21: "ppc64", 40: "arm", 62: "x86_64", 183: "arm64", 243: "riscv"}


def _elf(data: bytes, at: int) -> Hit | None:
    if len(data) < at + 52:
        return None
    cls, order, version = data[at + 4], data[at + 5], data[at + 6]
    if cls not in (1, 2) or order not in (1, 2) or version != 1:
        return None
    e = "<" if order == 1 else ">"
    etype, machine = struct.unpack_from(e + "HH", data, at + 16)
    if etype not in (1, 2, 3, 4):
        return None
    if cls == 1:
        shoff = struct.unpack_from(e + "I", data, at + 32)[0]
        shentsize, shnum = struct.unpack_from(e + "HH", data, at + 46)
    else:
        if len(data) < at + 64:
            return None
        shoff = struct.unpack_from(e + "Q", data, at + 40)[0]
        shentsize, shnum = struct.unpack_from(e + "HH", data, at + 58)
    end = at + shoff + shentsize * shnum if shoff and shnum else None
    if end is not None and end > len(data):
        end = None
    kind = {1: "relocatable", 2: "executable", 3: "shared", 4: "core"}[etype]
    return Hit(at, "elf", at, end, f"{8 * 4 * cls}-bit {'LE' if order == 1 else 'BE'} {_ELF_MACHINES.get(machine, machine)} {kind}")


def _squashfs(data: bytes, at: int) -> Hit | None:
    if len(data) < at + 48:
        return None
    e = "<" if data[at:at + 4] == b"hsqs" else ">"
    major = struct.unpack_from(e + "H", data, at + 28)[0]
    used = struct.unpack_from(e + "Q", data, at + 40)[0]
    if major not in (2, 3, 4) or not 96 <= used <= len(data) - at:
        return None
    comp = {1: "gzip", 2: "lzma", 3: "lzo", 4: "xz", 5: "lz4", 6: "zstd"}.get(struct.unpack_from(e + "H", data, at + 20)[0], "?")
    return Hit(at, "squashfs", at, at + used, f"v{major} {comp}")


def _cramfs(data: bytes, at: int) -> Hit | None:
    if data[at + 16:at + 32] != b"Compressed ROMFS":
        return None
    e = "<" if data[at] == 0x45 else ">"
    size = struct.unpack_from(e + "I", data, at + 4)[0]
    return Hit(at, "cramfs", at, at + size if at + size <= len(data) else None)


def _zip(data: bytes, at: int) -> Hit | None:
    if len(data) < at + 30:
        return None
    version, _, method = struct.unpack_from("<HHH", data, at + 4)
    if version > 63 or method not in (0, 8, 12, 14, 93, 95):
        return None
    name_len = struct.unpack_from("<H", data, at + 26)[0]
    name = data[at + 30:at + 30 + min(name_len, 64)]
    if not name or not all(32 <= b < 127 for b in name):
        return None
    return Hit(at, "zip", at, None, f"first entry {name.decode('ascii')}")


def _gziphead(data: bytes, at: int) -> Hit | None:
    """`<u32 compressed><u32 uncompressed>gziphead` and the payload after it (a vendor boot container)."""
    header = at - 8
    if header < 0 or header % 4:
        return None
    compressed, uncompressed = struct.unpack_from("<II", data, header)
    if not compressed or not 0 < uncompressed <= 1 << 26 or at + 8 + compressed > len(data):
        return None
    member = "gzip member" if data[at + 8:at + 10] == b"\x1f\x8b" else "not a gzip member"
    return Hit(header, "gziphead", at + 8, at + 8 + compressed, f"inflates to {uncompressed}; payload is {member}")


# magic, parser. Order breaks ties at one offset (first wins).
SIGNATURES: tuple[tuple[bytes, Callable[[bytes, int], Hit | None]], ...] = (
    (b"\x27\x05\x19\x56", _uimage),
    (b"\xd0\x0d\xfe\xed", _fdt),
    (b"\x7fELF", _elf),
    (b"hsqs", _squashfs),
    (b"sqsh", _squashfs),
    (b"\x45\x3d\xcd\x28", _cramfs),
    (b"\x28\xcd\x3d\x45", _cramfs),
    (b"gziphead", _gziphead),
    (b"\x1f\x8b", _gzip),
    (b"\xfd7zXZ\x00", _xz),
    (b"BZh", _bzip2),
    (b"\x5d\x00\x00", _lzma_alone),
    (b"\x28\xb5\x2f\xfd", _zstd),
    (b"PK\x03\x04", _zip),
)


def scan(data: bytes, spans: tuple[Span, ...]) -> tuple[Hit, ...]:
    """Every confirmed hit that starts inside one of `spans` (the image's programmed regions)."""
    found: dict[int, Hit] = {}
    for span in spans:
        for magic, parse in SIGNATURES:
            start = span.offset
            while (at := data.find(magic, start, span.end)) >= 0:
                start = at + 1
                if len(data) - at < 16:
                    continue
                hit = parse(data, at)
                if hit is not None and hit.offset not in found:
                    found[hit.offset] = hit
    return tuple(found[key] for key in sorted(found))
