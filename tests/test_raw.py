"""raw platform: layout, entry points, signatures, JFFS2, disassembly, and the platform chain.

Fixtures are synthetic; nothing here is a dump of a real device.
"""

from __future__ import annotations

import argparse
import bz2
import contextlib
import gzip
import hashlib
import io
import lzma
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from tools.dk import platforms
from tools.dk.core import cli, status
from tools.dk.core.config import load
from tools.dk.platforms.raw import carvecmd, discmd, jffs2, mapcmd
from tools.dk.platforms.raw.analysis import Profile, analyze
from tools.dk.platforms.raw.arm import disassemble_builtin, find_cortex_m
from tools.dk.platforms.raw.layout import Span, fill_spans, programmed_islands
from tools.dk.platforms.raw.signatures import scan

try:
    import capstone  # noqa: F401
    HAVE_CAPSTONE = True
except ImportError:
    HAVE_CAPSTONE = False


def pack(word: int) -> bytes:
    return struct.pack("<I", word)


def branch(addr: int, target: int) -> bytes:
    return pack(0xEA000000 | (((target - addr - 8) >> 2) & 0xFFFFFF))


def arm_boot(link: int = 0x80000000) -> bytearray:
    """An ARM classic vector table at 0: reset is a B to 0x60, the rest `ldr pc` to literals."""
    data = bytearray(b"\x11" * 0x100)
    data[0:4] = branch(0, 0x60)
    for index in range(7):
        data[4 + index * 4:8 + index * 4] = pack(0xE59FF014)
    for index, handler in enumerate(range(0x80, 0x80 + 7 * 4, 4)):
        struct.pack_into("<I", data, 0x20 + index * 4, link + handler)
        data[handler:handler + 4] = pack(0xE3A00000)
    data[0x60:0x64] = pack(0xE3A00000)
    return data


def jffs2_node(e: str, nodetype: int, body: bytes) -> bytes:
    node = bytearray(struct.pack(e + "HHI", 0x1985, nodetype, 12 + len(body)) + b"\0\0\0\0" + body)
    struct.pack_into(e + "I", node, 8, jffs2.crc(bytes(node[:8])))
    return bytes(node) + b"\xff" * (-len(node) % 4)


def jffs2_dirent(e: str, name: bytes, ino: int, pino: int = 1, kind: int = 8) -> bytes:
    body = bytearray(28 + len(name))
    struct.pack_into(e + "III", body, 0, pino, 1, ino)
    body[16] = len(name)
    body[17] = kind
    body[28:] = name
    return jffs2_node(e, 0xE001, bytes(body))


def jffs2_inode(e: str, ino: int, payload: bytes) -> bytes:
    stored = zlib.compress(payload)
    body = bytearray(56 + len(stored))
    struct.pack_into(e + "II", body, 0, ino, 1)
    struct.pack_into(e + "III", body, 32, 0, len(stored), len(payload))
    body[44] = 6
    body[56:] = stored
    return jffs2_node(e, 0xE002, bytes(body))


def uimage(payload: bytes, name: bytes = b"demo") -> bytes:
    header = bytearray(64)
    struct.pack_into(">IIIIIII", header, 0, 0x27051956, 0, 0, len(payload), 0x80008000, 0x80008000, zlib.crc32(payload))
    header[28:32] = bytes([5, 2, 2, 0])       # os linux, arch arm, kernel, comp none
    header[32:32 + len(name)] = name
    struct.pack_into(">I", header, 4, zlib.crc32(header))
    return bytes(header) + payload


def make_raw(root: Path, image: bytes, extra: str = "") -> object:
    (root / "orig").mkdir(parents=True)
    (root / "orig" / "image.bin").write_bytes(image)
    (root / "dk.toml").write_text(
        '[project]\nname = "thing"\nplatform = "raw"\n\n'
        f'[target]\noriginal = "orig/image.bin"\nsha1 = "{hashlib.sha1(image).hexdigest()}"\n\n'
        f"[platform]\n{extra}"
    )
    return load(root)


def run(command, cfg, argv: list[str]) -> tuple[int, str]:
    parser = argparse.ArgumentParser()
    if command.configure:
        command.configure(parser)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = command.run(cfg, parser.parse_args(argv))
    return code, out.getvalue()


class LayoutTest(unittest.TestCase):
    def test_gaps_of_any_fill_byte_split_the_image(self):
        data = b"A" * 10 + b"\x00" * 300 + b"B" * 10 + b"\xff" * 300 + b"C" * 10
        gaps = fill_spans(data, (0x00, 0xFF), 256)
        self.assertEqual(gaps, (Span(10, 300), Span(320, 300)))
        self.assertEqual(programmed_islands(len(data), gaps), (Span(0, 10), Span(310, 10), Span(620, 10)))
        self.assertEqual(fill_spans(data, (0xFF,), 256), (Span(320, 300),))

    def test_strings_copies_and_magics(self):
        block = b"COPY" * 64
        data = b"NRTS\x00" + b"bootcmd=run\x00" + b"\xff" * 4096 + block + b"\xff" * 4096 + block
        report = analyze(data, Profile(fill=(0xFF,)))
        self.assertEqual([m.text for m in report.magics], ["NRTS"])
        self.assertEqual([(c.offset, c.same_as) for c in report.copies], [(len(data) - 256, 4113)])
        self.assertIn("bootcmd=run", [s.text for s in report.strings])


class ArmTest(unittest.TestCase):
    def test_vector_table_gives_the_link_address(self):
        report = analyze(bytes(arm_boot()))
        self.assertEqual([(t.offset, t.link_base, t.reset_file, t.handlers[0]) for t in report.arm],
                         [(0, 0x80000000, 0x60, 0x80)])

    def test_cortex_m_at_the_start_of_an_island(self):
        buf = bytearray(b"\xff" * 0x200)
        for index, word in enumerate([0x20001000, 0x101, 0x111, 0x121, 0, 0, 0x131, 0]):
            struct.pack_into("<I", buf, index * 4, word)
        found = find_cortex_m(bytes(buf), (Span(0, len(buf)),))
        self.assertEqual([(t.stack, t.reset) for t in found], [(0x20001000, 0x100)])

    def test_builtin_disassembly_decodes_the_vector_idiom(self):
        lines = disassemble_builtin(branch(0, 0x60) + pack(0xE59FF014) + pack(0xEB000000), 0x80000000)
        self.assertEqual(lines, ["80000000  160000ea  b 0x80000060",
                                 "80000004  14f09fe5  ldr pc, [pc, #0x14]",
                                 "80000008  000000eb  bl 0x80000010"])


class Jffs2Test(unittest.TestCase):
    def test_crc_is_not_zlib_crc32(self):
        # A little-endian clean marker (type 0x2003, totlen 12) as mkfs.jffs2 writes it.
        header = bytes.fromhex("851903200c000000")
        self.assertEqual(jffs2.crc(header), 0xE41EB0B1)
        self.assertNotEqual(zlib.crc32(header), 0xE41EB0B1)

    def test_files_are_rebuilt_in_both_byte_orders(self):
        payload = b"hello jffs2 payload!!"
        for e, endian in (("<", "le"), (">", "be")):
            image = (jffs2_node(e, 0x2003, b"") + jffs2_dirent(e, b"etc", 2, kind=4)
                     + jffs2_dirent(e, b"hello.txt", 3, pino=2) + jffs2_inode(e, 3, payload))
            report = analyze(image)
            self.assertEqual([(fs.endian, fs.nodes, fs.bad_crc) for fs in report.jffs2], [(endian, 4, 0)])
            self.assertEqual([(f.path, f.data) for f in report.files], [("etc/hello.txt", payload)])

    def test_a_wrong_header_crc_is_counted(self):
        bad = bytearray(jffs2_dirent("<", b"x.txt", 2))
        bad[8] ^= 1
        image = jffs2_node("<", 0x2003, b"") + bytes(bad)
        self.assertEqual([fs.bad_crc for fs in analyze(image).jffs2], [1])


class SignatureTest(unittest.TestCase):
    def scan_one(self, blob: bytes, at: int = 0x40) -> list[tuple]:
        data = b"\x00" * at + blob + b"\x00" * 64
        return [(h.offset, h.kind, h.body, h.end) for h in scan(data, (Span(0, len(data)),))]

    def test_compressed_streams_end_where_they_decode(self):
        payload = bytes(range(256)) * 40
        for kind, blob in (("gzip", gzip.compress(payload, mtime=0)), ("xz", lzma.compress(payload)),
                           ("bzip2", bz2.compress(payload)), ("lzma", lzma.compress(payload, format=lzma.FORMAT_ALONE))):
            with self.subTest(kind):
                self.assertEqual(self.scan_one(blob), [(0x40, kind, 0x40, 0x40 + len(blob))])

    def test_uimage_header_crc_is_checked(self):
        image = uimage(b"K" * 100)
        self.assertEqual(self.scan_one(image), [(0x40, "uimage", 0x80, 0x80 + 100)])
        broken = bytearray(image)
        broken[40] ^= 1
        self.assertEqual(self.scan_one(bytes(broken)), [])

    def test_magic_without_its_structure_is_not_a_hit(self):
        self.assertEqual(self.scan_one(b"\x1f\x8b\x08\xff" + b"\x00" * 32), [])      # reserved gzip flags
        self.assertEqual(self.scan_one(b"\x00" * 8 + b"gziphead" + b"\x00" * 16), [])  # zero lengths
        self.assertEqual(self.scan_one(b"\x7fELF\x09\x09\x01" + b"\x00" * 60), [])    # bad class

    def test_gziphead_body_is_the_payload(self):
        blob = struct.pack("<II", 4, 8) + b"gziphead" + b"WXYZ"
        self.assertEqual(self.scan_one(blob), [(0x40, "gziphead", 0x50, 0x54)])


class PlatformChainTest(unittest.TestCase):
    def test_spi_nor_builds_on_raw(self):
        self.assertEqual([m.__name__.rsplit(".", 1)[-1] for m in platforms.chain("spi_nor")], ["spi_nor", "raw"])
        self.assertIs(platforms.hook("spi_nor", "profile"), platforms.load("spi_nor").profile)
        self.assertIs(platforms.hook("spi_nor", "status_checks"), platforms.load("raw").status_checks)
        self.assertIsNone(platforms.hook("gc_mwcc", "BASE"))

    def test_each_platform_lists_only_commands_that_apply(self):
        gc = {c.name for c in cli._commands("gc_mwcc")}
        self.assertTrue({"check", "build", "progress", "status"} <= gc)
        self.assertFalse({"map", "carve"} & gc)
        for name in ("raw", "spi_nor"):
            cmds = {c.name for c in cli._commands(name)}
            self.assertTrue({"map", "carve", "dis", "status", "doc-lint", "toolkit"} <= cmds, name)
            self.assertFalse({"check", "build", "progress", "floor"} & cmds, name)


class CommandTest(unittest.TestCase):
    def test_map_carve_dis_on_padded_image(self):
        image = bytes(arm_boot()) + b"\x00" * 8192 + gzip.compress(b"rootfs" * 500, mtime=0) + b"\x00" * 8192
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_raw(root, image)
            code, text = run(mapcmd.COMMAND, cfg, [])
            self.assertEqual(code, 0)
            self.assertIn("padding: byte 0x00 0xff, runs >= 4096: 2", text)
            self.assertIn("at 0x00000000  link 0x80000000  reset 0x00000060", text)
            self.assertIn("gzip      body 0x00002100-", text)
            self.assertIn("next: ./dk carve", text)

            self.assertEqual(run(carvecmd.COMMAND, cfg, [])[0], 0)
            regions = sorted(p.name for p in (root / "build" / "raw" / "regions").iterdir())
            self.assertIn("00000000.island", regions)
            self.assertIn("00002100.gzip", regions)
            self.assertEqual(run(carvecmd.COMMAND, cfg, ["--check"])[0], 0)
            (root / "build" / "raw" / "regions" / "00000000.island").write_bytes(b"x")
            self.assertEqual(run(carvecmd.COMMAND, cfg, ["--check"])[0], 1)

            code, text = run(discmd.COMMAND, cfg, ["--engine", "builtin", "--length", "8"])
            self.assertEqual(code, 0)
            self.assertIn("80000000  160000ea  b 0x80000060", text)
            code, text = run(discmd.COMMAND, cfg, ["--at", "0x2100", "--length", "4"])
            self.assertEqual(code, 2)
            self.assertIn("pass --arch", text)
            self.assertIn("next: ./dk dis --at 0x2100 --arch arm", text)

    @unittest.skipUnless(HAVE_CAPSTONE, "capstone not installed")
    def test_dis_takes_any_arch(self):
        image = b"\x00" * 16 + bytes.fromhex("3c1c0042") + b"\x00" * 16   # mips: lui gp, 0x42
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_raw(Path(tmp), image, "min_gap = 8\n")
            code, text = run(discmd.COMMAND, cfg, ["--arch", "mips", "--length", "4"])
        self.assertEqual(code, 0)
        self.assertIn("arch mips engine capstone", text)
        self.assertIn("lui", text)

    def test_status_names_the_chain_and_needs_no_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_raw(Path(tmp), bytes(arm_boot()))
            self.assertEqual(cfg.game_id, "")
            code, text = run(status.COMMAND, cfg, [])
        self.assertEqual(code, 0)
        self.assertIn("thing (platform raw)", text)
        self.assertIn("[x] image present and verified", text)
        self.assertNotIn("build.ninja", text)
        self.assertIn("next: ./dk carve", text)


if __name__ == "__main__":
    unittest.main()
