"""dotnet platform: a synthetic assembly built here byte by byte, plus a golden pass over the
project's own assembly when the project is a .NET one (skipped otherwise)."""

import argparse
import contextlib
import io
import re
import struct
import tempfile
import unittest
from pathlib import Path

from tools.dk.core.command import MISMATCH, OK, USAGE
from tools.dk.core.config import ConfigError, load
from tools.dk.platforms.dotnet import cmds, resources
from tools.dk.platforms.dotnet.assembly import Assembly
from tools.dk.platforms.dotnet.firstdiff import first_diff
from tools.dk.platforms.dotnet.il import decode, parse_body
from tools.dk.platforms.dotnet.listing import method_listing, xrefs
from tools.dk.platforms.dotnet.metadata import CODED, METADATA_SIGNATURE, TABLES, column_size, encode_coded
from tools.dk.platforms.dotnet.pe import FormatError
from tools.dk.platforms.dotnet.sig import SigReader, field_type

TEXT_RVA, TEXT_RAW = 0x2000, 0x200


def cu(n: int) -> bytes:
    """ECMA-335 compressed unsigned integer."""
    if n < 0x80:
        return bytes([n])
    if n < 0x4000:
        return struct.pack(">H", 0x8000 | n)
    return struct.pack(">I", 0xC0000000 | n)


class Heaps:
    def __init__(self):
        self.strings, self.blob, self.us, self.guid = bytearray(b"\0"), bytearray(b"\0"), bytearray(b"\0"), bytearray()

    def s(self, text: str) -> int:
        self.strings += text.encode() + b"\0"
        return len(self.strings) - len(text.encode()) - 1

    def b(self, data: bytes) -> int:
        at = len(self.blob)
        self.blob += cu(len(data)) + data
        return at

    def u(self, text: str) -> int:
        at = len(self.us)
        raw = text.encode("utf-16-le")
        self.us += cu(len(raw) + 1) + raw + b"\0"
        return at

    def g(self, raw: bytes) -> int:
        self.guid += raw
        return len(self.guid) // 16


def pack_tables(tables: dict) -> bytes:
    counts = {name: len(rows) for name, rows in tables.items() if rows}
    valid = sum(1 << tid for tid, name, _ in TABLES if name in counts)
    out = bytearray(struct.pack("<IBBBBQQ", 0, 2, 0, 0, 1, valid, 0))
    out += b"".join(struct.pack("<I", counts[name]) for _, name, _ in TABLES if name in counts)
    for _, name, columns in TABLES:
        for row in tables.get(name, []):
            for col, ctype in columns:
                value = row.get(col, (CODED[ctype][1][0], 0) if ctype in CODED else 0)
                if ctype in CODED:
                    value = encode_coded(ctype, value)
                out += value.to_bytes(column_size(ctype, counts, 0), "little")
    return bytes(out)


def metadata_root(table_stream: bytes, heaps: Heaps) -> bytes:
    def pad4(b: bytes) -> bytes:
        return bytes(b) + b"\0" * (-len(b) % 4)
    streams = [("#~", pad4(table_stream)), ("#Strings", pad4(heaps.strings)), ("#US", pad4(heaps.us)),
               ("#GUID", bytes(heaps.guid)), ("#Blob", pad4(heaps.blob))]
    version = b"v2.0.50727\0\0"
    head_size = 16 + len(version) + 4 + sum(8 + ((len(n) + 4) & ~3) for n, _ in streams)
    header = bytearray(struct.pack("<IHHII", METADATA_SIGNATURE, 1, 1, 0, len(version)) + version)
    header += struct.pack("<HH", 0, len(streams))
    offset, body = head_size, bytearray()
    for name, data in streams:
        raw = name.encode() + b"\0"
        header += struct.pack("<II", offset, len(data)) + raw + b"\0" * (-len(raw) % 4)
        body += data
        offset += len(data)
    return bytes(header + body)


def make_pe(text: bytes, cli_rva: int, cli_size: int) -> bytes:
    opt = bytearray(0xE0)
    struct.pack_into("<H", opt, 0, 0x10B)
    struct.pack_into("<I", opt, 28, 0x400000)
    struct.pack_into("<I", opt, 60, TEXT_RAW)
    struct.pack_into("<H", opt, 68, 2)
    struct.pack_into("<I", opt, 92, 16)
    struct.pack_into("<II", opt, 96 + 8 * 14, cli_rva, cli_size)
    dos = bytearray(0x80)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x14C, 1, 0x4DC07B49, 0, 0, len(opt), 0x102)
    section = struct.pack("<8sIIIIIIHHI", b".text", len(text), TEXT_RVA, len(text), TEXT_RAW, 0, 0, 0, 0, 0x60000020)
    headers = bytes(dos) + coff + bytes(opt) + section
    return headers.ljust(TEXT_RAW, b"\0") + text


def make_resources(types: list[str], entries: list[tuple[str, int, bytes]]) -> bytes:
    """A version-2 .resources file; entries are (name, type code, value bytes after the code)."""
    def s7(data: bytes) -> bytes:
        return cu(len(data)) if len(data) < 0x80 else bytes([0x80 | len(data) & 0x7F, len(data) >> 7])
    reader = s7(b"System.Resources.ResourceReader") + b"System.Resources.ResourceReader"
    head = struct.pack("<III", 0xBEEFCACE, 1, len(reader)) + reader
    head += struct.pack("<III", 2, len(entries), len(types)) + b"".join(s7(t.encode()) + t.encode() for t in types)
    head += (b"PAD" * 3)[:(-len(head)) % 8]       # align to 8 bytes
    names, data, positions = bytearray(), bytearray(), []
    for name, code, value in entries:
        positions.append(len(names))
        raw = name.encode("utf-16-le")
        names += s7(raw) + raw + struct.pack("<I", len(data))
        data += bytes([code]) + value
    data_start = len(head) + 8 * len(entries) + 4 + len(names)
    index = struct.pack(f"<{len(entries)}I", *range(len(entries))) + struct.pack(f"<{len(entries)}I", *positions)
    return head + index + struct.pack("<I", data_start) + bytes(names) + bytes(data)


PNG = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 3, 2) + b"\0" * 9


def build(eh_data_size: int = 12, helper_constant: int = 42) -> bytes:
    """A small assembly: Demo.Program with Main (tiny body), Send (fat body + try/catch), a
    nested type, a field, external SmtpClient calls, an assembly attribute and a resource."""
    h = Heaps()
    us_smtp, _ = h.u("smtp.example.com"), h.u("never loaded")
    main = bytes([0x72]) + struct.pack("<I", 0x70000000 | us_smtp) + bytes([0x73]) + struct.pack("<I", 0x0A000001) + b"\x26\x2a"
    send_code = (b"\x00\x02\x7b" + struct.pack("<I", 0x04000001) + b"\x26\xde\x06\x26\xde\x03\x00\x00\x00"
                 b"\x20" + struct.pack("<i", 587) + b"\xfe\x01\x45" + struct.pack("<Iii", 2, 0, 1) + b"\x00\x2a")
    send = struct.pack("<HHII", 0x301B, 2, len(send_code), 0) + send_code
    send += b"\0" * (-len(send) % 4)
    send += struct.pack("<BBH", 0x01, eh_data_size, 0) + struct.pack("<HHBHBI", 0, 1, 9, 0x0A, 3, 0x01000003)
    helper = bytes([(3 << 2) | 2, 0x1F, helper_constant, 0x2A])
    resource = make_resources(["System.Drawing.Bitmap, System.Drawing, Version=2.0.0.0"], [
        ("greeting", 1, cu(2) + b"hi"),
        ("../logo", 0x40, b"\x00\x01" + b"\x0f" + struct.pack("<ii", 3, len(PNG)) + b"\x02" + PNG + b"\x0b"),
    ])

    # .text layout: CLI header, bodies, resources, metadata
    cli_at = TEXT_RVA
    main_at = cli_at + 72
    send_at = main_at + 4 * ((1 + len(main) + 3) // 4)
    helper_at = send_at + len(send)
    res_at = helper_at + 4
    md_at = res_at + 4 + len(resource) + (-len(resource) % 4)

    void_sig, ctor_str = h.b(b"\x00\x01\x01\x1d\x0e"), h.b(b"\x20\x01\x01\x0e")
    tables = {
        "Module": [{"Name": h.s("demo.exe"), "Mvid": h.g(bytes(range(16)))}],
        "TypeRef": [
            {"ResolutionScope": ("AssemblyRef", 1), "TypeName": h.s("Object"), "TypeNamespace": h.s("System")},
            {"ResolutionScope": ("AssemblyRef", 2), "TypeName": h.s("SmtpClient"), "TypeNamespace": h.s("System.Net.Mail")},
            {"ResolutionScope": ("AssemblyRef", 1), "TypeName": h.s("Exception"), "TypeNamespace": h.s("System")},
            {"ResolutionScope": ("AssemblyRef", 1), "TypeName": h.s("AssemblyTitleAttribute"),
             "TypeNamespace": h.s("System.Reflection")},
            {"ResolutionScope": ("TypeRef", 1), "TypeName": h.s("Inner"), "TypeNamespace": 0},
        ],
        "TypeDef": [
            {"TypeName": h.s("<Module>"), "FieldList": 1, "MethodList": 1},
            {"Flags": 0x100001, "TypeName": h.s("Program"), "TypeNamespace": h.s("Demo"), "Extends": ("TypeRef", 1),
             "FieldList": 1, "MethodList": 1},
            {"TypeName": h.s("Nested"), "Extends": ("TypeRef", 1), "FieldList": 2, "MethodList": 3},
        ],
        "Field": [{"Flags": 0x1, "Name": h.s("count"), "Signature": h.b(b"\x06\x08")}],
        "MethodDef": [
            {"RVA": main_at, "Flags": 0x16, "Name": h.s("Main"), "Signature": void_sig, "ParamList": 1},
            {"RVA": send_at, "Flags": 0x6, "Name": h.s("Send"), "Signature": h.b(b"\x20\x00\x01"), "ParamList": 1},
            {"RVA": helper_at, "Flags": 0x16, "Name": h.s("Helper"), "Signature": h.b(b"\x00\x00\x08"), "ParamList": 1},
        ],
        "MemberRef": [
            {"Class": ("TypeRef", 2), "Name": h.s(".ctor"), "Signature": ctor_str},
            {"Class": ("TypeRef", 2), "Name": h.s("Send"), "Signature": h.b(b"\x20\x04\x01\x0e\x0e\x0e\x0e")},
            {"Class": ("TypeRef", 4), "Name": h.s(".ctor"), "Signature": ctor_str},
        ],
        "CustomAttribute": [{"Parent": ("Assembly", 1), "Type": ("MemberRef", 3),
                             "Value": h.b(b"\x01\x00" + cu(10) + b"Demo Title" + b"\x00\x00")}],
        "Assembly": [{"MajorVersion": 1, "MinorVersion": 2, "BuildNumber": 3, "RevisionNumber": 4, "Name": h.s("demo")}],
        "AssemblyRef": [{"MajorVersion": 2, "Name": h.s("mscorlib")}, {"MajorVersion": 2, "Name": h.s("System")}],
        "ManifestResource": [{"Offset": 0, "Flags": 1, "Name": h.s("Demo.Program.resources"),
                              "Implementation": ("File", 0)}],
        "NestedClass": [{"NestedClass": 3, "EnclosingClass": 2}],
    }
    md = metadata_root(pack_tables(tables), h)
    cli = struct.pack("<IHHIIIIIIII", 72, 2, 5, md_at, len(md), 1, 0x06000001, res_at, 4 + len(resource), 0, 0).ljust(72, b"\0")

    text = bytearray(cli)
    text += b"\0" * (main_at - cli_at - len(text)) + bytes([(len(main) << 2) | 2]) + main
    text += b"\0" * (send_at - cli_at - len(text)) + send
    text += helper
    text += struct.pack("<I", len(resource)) + resource
    text += b"\0" * (md_at - cli_at - len(text)) + md
    return make_pe(bytes(text), cli_at, 72)


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.asm = Assembly(build())

    def test_headers_and_names(self):
        asm = self.asm
        self.assertEqual(asm.md.version, "v2.0.50727")
        self.assertEqual(asm.cli.flag_names(), ["ILONLY"])
        self.assertEqual(asm.method_signature(1), "static void Demo.Program::Main(string[])")
        self.assertEqual(asm.method_name(3), "Demo.Program/Nested::Helper")
        self.assertEqual(asm.typeref_name(5), "System.Object/Inner")
        self.assertEqual(asm.typeref_scope(2), "System")
        self.assertEqual(asm.token_name(0x0A000002), "void System.Net.Mail.SmtpClient::Send(string, string, string, string)")
        self.assertEqual(asm.token_name(0x04000001), "int Demo.Program::count")
        self.assertEqual(list(asm.methods_of(2)), [1, 2])
        self.assertEqual(list(asm.methods_of(3)), [3])
        self.assertEqual(asm.md.guid(1), "03020100-0504-0706-0809-0a0b0c0d0e0f")
        self.assertEqual(asm.custom_attributes(("Assembly", 1)), [("System.Reflection.AssemblyTitleAttribute", ["'Demo Title'"])])

    def test_user_strings_and_xrefs(self):
        self.assertEqual([t for _, t in self.asm.md.user_strings()], ["smtp.example.com", "never loaded"])
        used, problems = xrefs(self.asm)
        self.assertEqual(problems, [])
        self.assertEqual(used[0x70000001], {1})
        self.assertEqual(used[0x0A000001], {1})
        self.assertEqual(used[0x04000001], {2})

    def test_fat_body_and_clause_count_conventions(self):
        for data_size in (12, 16):       # header excluded / included: both mean one clause
            body = Assembly(build(eh_data_size=data_size)).body(2)
            self.assertEqual(len(body.clauses), 1, data_size)
            c = body.clauses[0]
            self.assertEqual((c.kind, c.try_start, c.try_end, c.handler_start, c.handler_end, c.extra),
                             ("catch", 1, 0x0A, 0x0A, 0x0D, 0x01000003))
        listing = "\n".join(method_listing(self.asm, 2))
        self.assertIn(".try IL_0001 to IL_000a catch System.Exception handler IL_000a to IL_000d", listing)
        self.assertIn("IL_0008: leave.s IL_0010", listing)
        self.assertIn("IL_0010: ldc.i4 587", listing)
        self.assertIn("IL_0015: ceq", listing)
        self.assertIn("IL_0017: switch (IL_0024, IL_0025)", listing)

    def test_decode_errors(self):
        with self.assertRaises(FormatError):
            decode(b"\x24")                  # unassigned opcode
        with self.assertRaises(FormatError):
            decode(b"\x20\x01\x00")          # ldc.i4 operand cut short
        with self.assertRaises(FormatError):
            parse_body(b"\x00")

    def test_signatures(self):
        names = lambda ref: f"{ref[0]}{ref[1]}"
        self.assertEqual(field_type(b"\x06\x1d\x0e", names), "string[]")
        self.assertEqual(SigReader(b"\x00\x02\x01\x10\x08\x14\x08\x02\x00\x00", names).method("f"), "static void f(int&, int[,])")
        self.assertEqual(SigReader(b"\x15\x12\x05\x01\x08", names).type(), "TypeRef1<int>")

    def test_resources(self):
        (name, blob), = self.asm.manifest_resources()
        self.assertEqual(name, "Demo.Program.resources")
        greeting, logo = resources.parse(blob)
        self.assertEqual((greeting.name, greeting.type, greeting.value), ("greeting", "string", "hi"))
        self.assertEqual((logo.name, logo.type), ("../logo", "System.Drawing.Bitmap"))
        (ext, desc, data), = resources.contained_files(logo)
        self.assertEqual((ext, desc, data), ("png", "png 3x2", PNG))
        self.assertEqual(resources.safe_name("../logo"), "_logo")
        self.assertEqual(resources.safe_name("..."), "_")
        with self.assertRaises(FormatError):
            resources.parse(b"not resources")

    def test_first_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = Path(tmp, "a.exe"), Path(tmp, "b.exe"), Path(tmp, "c.exe")
            a.write_bytes(build())
            b.write_bytes(build(eh_data_size=16))
            c.write_bytes(build(helper_constant=43))
            self.assertIn("IL identical", first_diff(None, b, a, 5)[0])
            self.assertEqual(first_diff(None, c, a, 5),
                             ["differs: static int Demo.Program/Nested::Helper() at instruction 0: "
                              "ours 'ldc.i4.s 43', target 'ldc.i4.s 42'"])


DK_TOML = """\
[project]
name = "demo"
game_id = "DEMO"
platform = "dotnet"

[target]
original = "orig/demo.exe"
built = "build/demo.exe"
sha1 = "{sha1}"
"""


class CommandTest(unittest.TestCase):
    def run_cmd(self, run, cfg, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run(cfg, argparse.Namespace(**kw))
        return code, out.getvalue()

    def project(self, tmp: str, data: bytes | None):
        root = Path(tmp)
        (root / "dk.toml").write_text(DK_TOML.format(sha1="0" * 40))
        if data is not None:
            (root / "orig").mkdir()
            (root / "orig/demo.exe").write_bytes(data)
        return load(root)

    def test_info_strings_il_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.project(tmp, build())
            code, out = self.run_cmd(cmds.run_info, cfg)
            self.assertEqual(code, OK)
            self.assertIn("entry  static void Demo.Program::Main(string[])", out)
            self.assertIn("[System.Reflection.AssemblyTitle] 'Demo Title'", out)
            self.assertRegex(out, r"mail,network\s+System\.Net\.Mail\.SmtpClient::\.ctor\n\s+← Demo\.Program::Main")
            self.assertRegex(out, r"System\.Net\.Mail\.SmtpClient::Send\n\s+← \(not called\)")

            code, out = self.run_cmd(cmds.run_strings, cfg, unreferenced=False)
            self.assertEqual(code, OK)
            self.assertIn('"smtp.example.com"\n        ← Demo.Program::Main', out)
            self.assertIn('"never loaded"\n        ← (unreferenced)', out)

            code, out = self.run_cmd(cmds.run_il, cfg, names=["Nested"], out=None)
            self.assertEqual(code, OK)
            self.assertIn("IL_0000: ldc.i4.s 42", out)
            self.assertEqual(self.run_cmd(cmds.run_il, cfg, names=["NoSuchThing"], out=None)[0], USAGE)
            code, _ = self.run_cmd(cmds.run_il, cfg, names=[], out=str(Path(tmp, "il")))
            self.assertEqual(sorted(p.name for p in Path(tmp, "il").iterdir()),
                             ["Demo.Program.il", "Demo.Program_Nested.il", "_Module_.il"])

            code, out = self.run_cmd(cmds.run_resources, cfg, extract=str(Path(tmp, "res")))
            self.assertEqual(code, OK)
            self.assertIn("holds png 3x2", out)
            written = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file() and "res" in p.parts)
            self.assertEqual(written, ["res/Demo.Program.resources", "res/Demo.Program/_logo.png"])

    def test_bad_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_cmd(cmds.run_info, self.project(tmp, None))[0], USAGE)
        with tempfile.TemporaryDirectory() as tmp:
            native = make_pe(b"\0" * 16, 0, 0)
            code, out = self.run_cmd(cmds.run_info, self.project(tmp, native))
            self.assertEqual(code, USAGE)
            self.assertIn("not a .NET assembly", out)

    def test_il_reports_undecodable_body(self):
        data = bytearray(build())
        at = data.index(bytes([(3 << 2) | 2, 0x1F, 42, 0x2A]))
        data[at + 1] = 0x24                  # Helper's first opcode → unassigned
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self.run_cmd(cmds.run_il, self.project(tmp, bytes(data)), names=["Helper"], out=None)
        self.assertEqual(code, MISMATCH)
        self.assertIn("decode stopped: IL_0000: unknown opcode 0x24", out)


_UNRESOLVED = re.compile(r"\b(TypeDef|TypeRef|TypeSpec|MethodDef|MemberRef|Field|MethodSpec|StandAloneSig)#\d+")


class GoldenTest(unittest.TestCase):
    """The project's own assembly, when this is a .NET project: everything must decode and resolve."""

    def test_project_assembly(self):
        try:
            cfg = load()
        except ConfigError:
            self.skipTest("no dk.toml")
        if cfg.platform != "dotnet" or not cfg.original.is_file():
            self.skipTest("not a .NET project, or the original is missing")
        asm = Assembly.load(cfg.original)
        used, problems = xrefs(asm)
        self.assertEqual(problems, [])
        for tok in used:
            self.assertIsNone(_UNRESOLVED.search(asm.token_name(tok)), hex(tok))
        for method in range(1, len(asm.md.rows("MethodDef")) + 1):
            listing = "\n".join(method_listing(asm, method))
            self.assertNotIn("decode stopped", listing)
            self.assertNotIn("body unreadable", listing)
        for name, blob in asm.manifest_resources():
            if blob is not None and name.endswith(".resources"):
                resources.parse(blob)


if __name__ == "__main__":
    unittest.main()
