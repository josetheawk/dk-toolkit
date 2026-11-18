"""A loaded .NET assembly: PE + CLI header + metadata, with names for every token."""

from __future__ import annotations

import struct
from bisect import bisect_right
from functools import cached_property
from pathlib import Path

from . import sig
from .il import Body, parse_body
from .metadata import TABLE_IDS, TABLE_NAMES, CliHeader, Metadata
from .pe import CLR_DIRECTORY, PE, FormatError

USER_STRING_TABLE = 0x70


def token(table: str, rid: int) -> int:
    return TABLE_IDS[table] << 24 | rid


class Assembly:
    def __init__(self, data: bytes):
        self.pe = PE.parse(data)
        rva, size = self.pe.directory(CLR_DIRECTORY)
        if not rva:
            raise FormatError("no CLI header: not a .NET assembly")
        self.cli = CliHeader.parse(self.pe.read(rva, max(size, 72)))
        self.md = Metadata.parse(self.pe.read(*self.cli.metadata))

    @classmethod
    def load(cls, path: Path) -> Assembly:
        return cls(path.read_bytes())

    # Ownership: TypeDef rows own runs of fields and methods (II.22.37)

    def _starts(self, column: str) -> list[int]:
        return [row[column] for row in self.md.rows("TypeDef")]

    def _run(self, column: str, table: str, typedef: int) -> range:
        starts = self._starts(column)
        end = starts[typedef] if typedef < len(starts) else len(self.md.rows(table)) + 1
        return range(starts[typedef - 1], max(end, starts[typedef - 1]))

    def methods_of(self, typedef: int) -> range:
        return self._run("MethodList", "MethodDef", typedef)

    def fields_of(self, typedef: int) -> range:
        return self._run("FieldList", "Field", typedef)

    def _owner(self, column: str, rid: int) -> int:
        return bisect_right(self._starts(column), rid)    # starts are non-decreasing

    def method_owner(self, method: int) -> int:
        return self._owner("MethodList", method)

    def field_owner(self, field: int) -> int:
        return self._owner("FieldList", field)

    @cached_property
    def _enclosing(self) -> dict[int, int]:
        return {row["NestedClass"]: row["EnclosingClass"] for row in self.md.rows("NestedClass")}

    # Names

    def typedef_name(self, rid: int) -> str:
        row = self.md.row("TypeDef", rid)
        if row is None:
            return f"TypeDef#{rid}"
        name = self.md.string(row["TypeName"])
        if rid in self._enclosing:
            return f"{self.typedef_name(self._enclosing[rid])}/{name}"
        ns = self.md.string(row["TypeNamespace"])
        return f"{ns}.{name}" if ns else name

    def typeref_name(self, rid: int) -> str:
        row = self.md.row("TypeRef", rid)
        if row is None:
            return f"TypeRef#{rid}"
        name = self.md.string(row["TypeName"])
        scope_table, scope = row["ResolutionScope"]
        if scope_table == "TypeRef" and scope:
            return f"{self.typeref_name(scope)}/{name}"
        ns = self.md.string(row["TypeNamespace"])
        return f"{ns}.{name}" if ns else name

    def type_name(self, ref: tuple[str | None, int]) -> str:
        table, rid = ref
        if not rid:
            return ""
        if table == "TypeDef":
            return self.typedef_name(rid)
        if table == "TypeRef":
            return self.typeref_name(rid)
        if table == "TypeSpec":
            row = self.md.row("TypeSpec", rid)
            try:
                return sig.SigReader(self.md.blob(row["Signature"]), self.type_name).type() if row else f"TypeSpec#{rid}"
            except (FormatError, IndexError):
                return f"TypeSpec#{rid}"
        return f"{table}#{rid}"

    def typeref_scope(self, rid: int) -> str:
        """The assembly (or module) a TypeRef resolves into."""
        row = self.md.row("TypeRef", rid)
        while row is not None and row["ResolutionScope"][0] == "TypeRef":
            row = self.md.row("TypeRef", row["ResolutionScope"][1])
        if row is None:
            return ""
        table, scope = row["ResolutionScope"]
        if table == "AssemblyRef":
            return self.md.string(self.md.row("AssemblyRef", scope)["Name"])
        if table == "ModuleRef":
            return self.md.string(self.md.row("ModuleRef", scope)["Name"])
        return ""

    def method_name(self, rid: int) -> str:
        row = self.md.row("MethodDef", rid)
        if row is None:
            return f"MethodDef#{rid}"
        return f"{self.typedef_name(self.method_owner(rid))}::{self.md.string(row['Name'])}"

    def method_signature(self, rid: int) -> str:
        row = self.md.row("MethodDef", rid)
        return self._signed(row["Signature"], self.method_name(rid)) if row else f"MethodDef#{rid}"

    def field_name(self, rid: int) -> str:
        row = self.md.row("Field", rid)
        if row is None:
            return f"Field#{rid}"
        return f"{self.typedef_name(self.field_owner(rid))}::{self.md.string(row['Name'])}"

    def memberref_parent(self, rid: int) -> str:
        row = self.md.row("MemberRef", rid)
        if row is None:
            return ""
        table, parent = row["Class"]
        if table == "MethodDef":
            return self.method_name(parent)
        if table == "ModuleRef":
            return self.md.string(self.md.row("ModuleRef", parent)["Name"])
        return self.type_name((table, parent))

    def memberref_name(self, rid: int) -> str:
        row = self.md.row("MemberRef", rid)
        return f"{self.memberref_parent(rid)}::{self.md.string(row['Name'])}" if row else f"MemberRef#{rid}"

    def memberref_signature(self, rid: int) -> str:
        row = self.md.row("MemberRef", rid)
        return self._signed(row["Signature"], self.memberref_name(rid)) if row else f"MemberRef#{rid}"

    def _signed(self, blob_offset: int, name: str) -> str:
        blob = self.md.blob(blob_offset)
        try:
            if sig.is_field_sig(blob):
                return f"{sig.field_type(blob, self.type_name)} {name}"
            return sig.SigReader(blob, self.type_name).method(name)
        except (FormatError, IndexError):
            return name

    def token_name(self, tok: int, full: bool = True) -> str:
        """Readable text for an IL token operand."""
        table, rid = tok >> 24, tok & 0xFFFFFF
        if table == USER_STRING_TABLE:
            return _quote(self.md.user_string(rid))
        name = TABLE_NAMES.get(table, f"table{table:#x}")
        if name in ("TypeDef", "TypeRef", "TypeSpec"):
            return self.type_name((name, rid))
        if name == "MethodDef":
            return self.method_signature(rid) if full else self.method_name(rid)
        if name == "MemberRef":
            return self.memberref_signature(rid) if full else self.memberref_name(rid)
        if name == "Field":
            row = self.md.row("Field", rid)
            return self._signed(row["Signature"], self.field_name(rid)) if row and full else self.field_name(rid)
        if name == "MethodSpec":
            row = self.md.row("MethodSpec", rid)
            if row is None:
                return f"MethodSpec#{rid}"
            target = row["Method"]
            base = self.token_name(token(target[0], target[1]), full)
            try:
                return base + " " + sig.method_spec(self.md.blob(row["Instantiation"]), self.type_name)
            except (FormatError, IndexError):
                return base
        if name == "StandAloneSig":
            row = self.md.row("StandAloneSig", rid)
            try:
                return sig.SigReader(self.md.blob(row["Signature"]), self.type_name).method() if row else name
            except (FormatError, IndexError):
                return f"StandAloneSig#{rid}"
        return f"{name}#{rid}"

    # Code and data

    def body(self, method: int) -> Body | None:
        row = self.md.row("MethodDef", method)
        if row is None or not row["RVA"]:
            return None                          # abstract, extern, runtime-provided
        return parse_body(self.pe.view(row["RVA"]))

    def locals(self, body: Body) -> list[str]:
        if not body.local_sig:
            return []
        row = self.md.row("StandAloneSig", body.local_sig & 0xFFFFFF)
        try:
            return sig.local_types(self.md.blob(row["Signature"]), self.type_name) if row else []
        except (FormatError, IndexError):
            return ["?"]

    def custom_attributes(self, parent: tuple[str, int]) -> list[tuple[str, list[str]]]:
        """(attribute type, rendered arguments) for every attribute on `parent`."""
        out = []
        for row in self.md.rows("CustomAttribute"):
            if row["Parent"] != parent:
                continue
            table, ctor = row["Type"]
            ctor_row = self.md.row(table, ctor) if table else None
            if ctor_row is None:
                continue
            attr_type = self.memberref_parent(ctor) if table == "MemberRef" else self.typedef_name(self.method_owner(ctor))
            try:
                args = sig.custom_attribute_args(self.md.blob(row["Value"]), self.md.blob(ctor_row["Signature"]))
            except (FormatError, IndexError, struct.error, UnicodeDecodeError):
                args = ["<" + self.md.blob(row["Value"]).hex() + ">"]
            out.append((attr_type, args))
        return out

    def manifest_resources(self) -> list[tuple[str, bytes | None]]:
        """(name, bytes) per manifest resource; bytes is None for one stored in another file."""
        out = []
        base = self.cli.resources[0]
        for row in self.md.rows("ManifestResource"):
            name = self.md.string(row["Name"])
            if row["Implementation"][1]:
                out.append((name, None))
                continue
            (length,) = struct.unpack("<I", self.pe.read(base + row["Offset"], 4))
            out.append((name, self.pe.read(base + row["Offset"] + 4, length)))
        return out


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return '"' + "".join(c if c.isprintable() or c == " " else f"\\u{ord(c):04x}" for c in escaped) + '"'
