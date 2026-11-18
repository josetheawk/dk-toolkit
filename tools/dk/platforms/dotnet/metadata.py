"""ECMA-335 metadata: the CLI header, the metadata root, its heaps and the #~ table stream.

Rows are dicts keyed by column name. A simple index is a 1-based row number (0 = null);
a coded index is a (table name, row) pair.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass

from .pe import FormatError

METADATA_SIGNATURE = 0x424A5342      # "BSJB"

# Coded index kinds: (tag bits, tables by tag; None = tag not used)
CODED: dict[str, tuple[int, tuple[str | None, ...]]] = {
    "TypeDefOrRef": (2, ("TypeDef", "TypeRef", "TypeSpec")),
    "HasConstant": (2, ("Field", "Param", "Property")),
    "HasCustomAttribute": (5, ("MethodDef", "Field", "TypeRef", "TypeDef", "Param", "InterfaceImpl", "MemberRef",
                               "Module", "DeclSecurity", "Property", "Event", "StandAloneSig", "ModuleRef",
                               "TypeSpec", "Assembly", "AssemblyRef", "File", "ExportedType", "ManifestResource",
                               "GenericParam", "GenericParamConstraint", "MethodSpec")),
    "HasFieldMarshal": (1, ("Field", "Param")),
    "HasDeclSecurity": (2, ("TypeDef", "MethodDef", "Assembly")),
    "MemberRefParent": (3, ("TypeDef", "TypeRef", "ModuleRef", "MethodDef", "TypeSpec")),
    "HasSemantics": (1, ("Event", "Property")),
    "MethodDefOrRef": (1, ("MethodDef", "MemberRef")),
    "MemberForwarded": (1, ("Field", "MethodDef")),
    "Implementation": (2, ("File", "AssemblyRef", "ExportedType")),
    "CustomAttributeType": (3, (None, None, "MethodDef", "MemberRef", None)),
    "ResolutionScope": (2, ("Module", "ModuleRef", "AssemblyRef", "TypeRef")),
    "TypeOrMethodDef": (1, ("TypeDef", "MethodDef")),
}

# Tables in id order. Column types: u8/u16/u32, str/guid/blob (heap indexes),
# @Table (simple index into Table), or a CODED kind.
_SCHEMA = """\
00 Module Generation:u16 Name:str Mvid:guid EncId:guid EncBaseId:guid
01 TypeRef ResolutionScope:ResolutionScope TypeName:str TypeNamespace:str
02 TypeDef Flags:u32 TypeName:str TypeNamespace:str Extends:TypeDefOrRef FieldList:@Field MethodList:@MethodDef
03 FieldPtr Field:@Field
04 Field Flags:u16 Name:str Signature:blob
05 MethodPtr Method:@MethodDef
06 MethodDef RVA:u32 ImplFlags:u16 Flags:u16 Name:str Signature:blob ParamList:@Param
07 ParamPtr Param:@Param
08 Param Flags:u16 Sequence:u16 Name:str
09 InterfaceImpl Class:@TypeDef Interface:TypeDefOrRef
0A MemberRef Class:MemberRefParent Name:str Signature:blob
0B Constant Type:u8 Padding:u8 Parent:HasConstant Value:blob
0C CustomAttribute Parent:HasCustomAttribute Type:CustomAttributeType Value:blob
0D FieldMarshal Parent:HasFieldMarshal NativeType:blob
0E DeclSecurity Action:u16 Parent:HasDeclSecurity PermissionSet:blob
0F ClassLayout PackingSize:u16 ClassSize:u32 Parent:@TypeDef
10 FieldLayout Offset:u32 Field:@Field
11 StandAloneSig Signature:blob
12 EventMap Parent:@TypeDef EventList:@Event
13 EventPtr Event:@Event
14 Event EventFlags:u16 Name:str EventType:TypeDefOrRef
15 PropertyMap Parent:@TypeDef PropertyList:@Property
16 PropertyPtr Property:@Property
17 Property Flags:u16 Name:str Type:blob
18 MethodSemantics Semantics:u16 Method:@MethodDef Association:HasSemantics
19 MethodImpl Class:@TypeDef MethodBody:MethodDefOrRef MethodDeclaration:MethodDefOrRef
1A ModuleRef Name:str
1B TypeSpec Signature:blob
1C ImplMap MappingFlags:u16 MemberForwarded:MemberForwarded ImportName:str ImportScope:@ModuleRef
1D FieldRVA RVA:u32 Field:@Field
1E EncLog Token:u32 FuncCode:u32
1F EncMap Token:u32
20 Assembly HashAlgId:u32 MajorVersion:u16 MinorVersion:u16 BuildNumber:u16 RevisionNumber:u16 Flags:u32 PublicKey:blob Name:str Culture:str
21 AssemblyProcessor Processor:u32
22 AssemblyOS OSPlatformID:u32 OSMajorVersion:u32 OSMinorVersion:u32
23 AssemblyRef MajorVersion:u16 MinorVersion:u16 BuildNumber:u16 RevisionNumber:u16 Flags:u32 PublicKeyOrToken:blob Name:str Culture:str HashValue:blob
24 AssemblyRefProcessor Processor:u32 AssemblyRef:@AssemblyRef
25 AssemblyRefOS OSPlatformId:u32 OSMajorVersion:u32 OSMinorVersion:u32 AssemblyRef:@AssemblyRef
26 File Flags:u32 Name:str HashValue:blob
27 ExportedType Flags:u32 TypeDefId:u32 TypeName:str TypeNamespace:str Implementation:Implementation
28 ManifestResource Offset:u32 Flags:u32 Name:str Implementation:Implementation
29 NestedClass NestedClass:@TypeDef EnclosingClass:@TypeDef
2A GenericParam Number:u16 Flags:u16 Owner:TypeOrMethodDef Name:str
2B MethodSpec Method:MethodDefOrRef Instantiation:blob
2C GenericParamConstraint Owner:@GenericParam Constraint:TypeDefOrRef
"""

TABLES: tuple[tuple[int, str, tuple[tuple[str, str], ...]], ...] = tuple(
    (int(line.split()[0], 16), line.split()[1], tuple(tuple(c.split(":")) for c in line.split()[2:]))
    for line in _SCHEMA.splitlines()
)
TABLE_IDS = {name: tid for tid, name, _ in TABLES}
TABLE_NAMES = {tid: name for tid, name, _ in TABLES}

_FIXED = {"u8": "B", "u16": "H", "u32": "I"}
HEAP_BITS = {"str": 0x01, "guid": 0x02, "blob": 0x04}   # #~ HeapSizes: 4-byte indexes into that heap


def column_size(ctype: str, counts: dict[str, int], heap_sizes: int) -> int:
    """Byte width of one column, from the row counts and heap-size flags (II.24.2.6)."""
    if ctype in _FIXED:
        return struct.calcsize(_FIXED[ctype])
    if ctype in HEAP_BITS:
        return 4 if heap_sizes & HEAP_BITS[ctype] else 2
    if ctype.startswith("@"):
        return 4 if counts.get(ctype[1:], 0) > 0xFFFF else 2
    bits, tables = CODED[ctype]
    largest = max(counts.get(t, 0) for t in tables if t)
    return 2 if largest < 1 << (16 - bits) else 4


def decode_coded(ctype: str, value: int) -> tuple[str | None, int]:
    bits, tables = CODED[ctype]
    tag = value & ((1 << bits) - 1)
    return (tables[tag] if tag < len(tables) else None), value >> bits


def encode_coded(ctype: str, ref: tuple[str | None, int]) -> int:
    bits, tables = CODED[ctype]
    table, rid = ref
    return rid << bits | (tables.index(table) if rid else 0)


def compressed_uint(data: bytes | memoryview, at: int) -> tuple[int, int]:
    """(value, bytes used) of an ECMA-335 compressed unsigned integer (II.23.2)."""
    b0 = data[at]
    if b0 & 0x80 == 0:
        return b0, 1
    if b0 & 0xC0 == 0x80:
        return (b0 & 0x3F) << 8 | data[at + 1], 2
    if b0 & 0xE0 == 0xC0:
        return (b0 & 0x1F) << 24 | data[at + 1] << 16 | data[at + 2] << 8 | data[at + 3], 4
    raise FormatError(f"bad compressed integer lead byte {b0:#x}")


@dataclass(frozen=True)
class CliHeader:
    runtime: tuple[int, int]
    metadata: tuple[int, int]
    flags: int
    entry_token: int
    resources: tuple[int, int]
    strong_name: tuple[int, int]

    FLAG_NAMES = ((0x1, "ILONLY"), (0x2, "32BITREQUIRED"), (0x4, "IL_LIBRARY"), (0x8, "STRONGNAMESIGNED"),
                  (0x10, "NATIVE_ENTRYPOINT"), (0x10000, "TRACKDEBUGDATA"), (0x20000, "32BITPREFERRED"))

    @classmethod
    def parse(cls, data: bytes) -> CliHeader:
        if len(data) < 44:
            raise FormatError("CLI header truncated")
        _, major, minor, md_rva, md_size, flags, entry, res_rva, res_size, sn_rva, sn_size = \
            struct.unpack_from("<IHHIIIIIIII", data, 0)
        return cls((major, minor), (md_rva, md_size), flags, entry, (res_rva, res_size), (sn_rva, sn_size))

    def flag_names(self) -> list[str]:
        return [name for bit, name in self.FLAG_NAMES if self.flags & bit]


def parse_streams(data: bytes) -> tuple[str, dict[str, bytes]]:
    """Metadata root (II.24.2.1) → (runtime version string, {stream name: bytes})."""
    if len(data) < 16:
        raise FormatError("metadata root truncated")
    signature, _, _, _, length = struct.unpack_from("<IHHII", data, 0)
    if signature != METADATA_SIGNATURE:
        raise FormatError(f"bad metadata signature {signature:#x}")
    version = data[16:16 + length].split(b"\0")[0].decode("utf-8", "replace")
    at = 16 + length
    _, nstreams = struct.unpack_from("<HH", data, at)
    at += 4
    streams: dict[str, bytes] = {}
    for _ in range(nstreams):
        offset, size = struct.unpack_from("<II", data, at)
        end = data.index(b"\0", at + 8) - (at + 8)
        name = data[at + 8:at + 8 + end].decode("ascii", "replace")
        at += 8 + ((end + 4) & ~3)
        streams.setdefault(name, data[offset:offset + size])   # the first of duplicate names wins
    return version, streams


def parse_tables(stream: bytes) -> tuple[dict[str, list[dict]], int]:
    """The #~ (or #-) stream → ({table name: rows}, heap-size flags)."""
    _, _, _, heap_sizes, _, valid, _ = struct.unpack_from("<IBBBBQQ", stream, 0)
    at = 24
    counts: dict[str, int] = {}
    for tid in range(64):
        if valid >> tid & 1:
            if tid not in TABLE_NAMES:
                raise FormatError(f"unknown metadata table {tid:#x}")
            counts[TABLE_NAMES[tid]] = struct.unpack_from("<I", stream, at)[0]
            at += 4
    if heap_sizes & 0x40:           # extra data dword (seen in uncompressed #- streams)
        at += 4
    tables: dict[str, list[dict]] = {}
    for _, name, columns in TABLES:
        widths = [column_size(ctype, counts, heap_sizes) for _, ctype in columns]
        rows = []
        for _ in range(counts.get(name, 0)):
            row = {}
            for (col, ctype), width in zip(columns, widths):
                value = int.from_bytes(stream[at:at + width], "little")
                at += width
                row[col] = decode_coded(ctype, value) if ctype in CODED else value
            rows.append(row)
        tables[name] = rows
    return tables, heap_sizes


@dataclass
class Metadata:
    version: str
    tables: dict[str, list[dict]]
    strings_heap: bytes
    blob_heap: bytes
    us_heap: bytes
    guid_heap: bytes

    @classmethod
    def parse(cls, data: bytes) -> Metadata:
        version, streams = parse_streams(data)
        table_stream = streams.get("#~", streams.get("#-"))
        if table_stream is None:
            raise FormatError("metadata has no #~ table stream")
        tables, _ = parse_tables(table_stream)
        return cls(version, tables, streams.get("#Strings", b""), streams.get("#Blob", b""),
                   streams.get("#US", b""), streams.get("#GUID", b""))

    def rows(self, table: str) -> list[dict]:
        return self.tables.get(table, [])

    def row(self, table: str, rid: int) -> dict | None:
        rows = self.rows(table)
        return rows[rid - 1] if 0 < rid <= len(rows) else None

    def string(self, offset: int) -> str:
        end = self.strings_heap.find(b"\0", offset)
        return self.strings_heap[offset:end if end >= 0 else None].decode("utf-8", "replace")

    def blob(self, offset: int) -> bytes:
        if offset >= len(self.blob_heap):
            return b""
        length, used = compressed_uint(self.blob_heap, offset)
        return self.blob_heap[offset + used:offset + used + length]

    def user_string(self, offset: int) -> str:
        length, used = compressed_uint(self.us_heap, offset)
        raw = self.us_heap[offset + used:offset + used + length]
        return raw[:len(raw) & ~1].decode("utf-16-le", "replace")    # odd trailing byte is a flag

    def user_strings(self) -> list[tuple[int, str]]:
        """Every #US entry as (offset, text), in heap order."""
        out, at = [], 1
        while at < len(self.us_heap):
            try:
                length, used = compressed_uint(self.us_heap, at)
            except (FormatError, IndexError):
                break
            if length == 0:          # zero padding at the end of the heap
                at += used
                continue
            out.append((at, self.user_string(at)))
            at += used + length
        return out

    def guid(self, index: int) -> str:
        raw = self.guid_heap[(index - 1) * 16:index * 16] if index else b""
        return str(uuid.UUID(bytes_le=raw)) if len(raw) == 16 else ""
