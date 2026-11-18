"""SPI NOR platform: raw analysis with erased-flash rules. Fixtures are synthetic; nothing here is a flash dump."""

from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from tests.test_raw import arm_boot, jffs2_dirent, jffs2_node, run
from tools.dk.core import status
from tools.dk.core.config import load
from tools.dk.platforms.raw import carvecmd, discmd, mapcmd
from tools.dk.platforms.spi_nor import geometry


def sample_image() -> bytes:
    """ARM vectors, a rejected gziphead string, one real gziphead, JFFS2, and a duplicated block."""
    data = bytearray(b"\xff" * 0x4600)   # ends on the second copied block, so trailing 0xFF does not glue onto it
    data[0:0x100] = arm_boot()
    data[0xB0:0xB0 + 12] = b"bootcmd=run\x00"
    data[0xC0:0xC0 + 10] = b"/dev/null\x00"
    data[0xE8:0xE8 + 8] = b"gziphead"    # zero length words before it: not a container
    header = struct.pack("<II", 4, 8) + b"gziphead" + b"WXYZ"
    data[0x1100:0x1100 + len(header)] = header
    fs = jffs2_node("<", 0x2003, b"") + jffs2_dirent("<", b"demo.txt", 2)
    data[0x2200:0x2200 + len(fs)] = fs
    block = b"COPY" * 64
    data[0x3400:0x3400 + len(block)] = block
    data[0x4500:0x4500 + len(block)] = block
    return bytes(data)


def make_spi(root: Path, image: bytes, *, sha1: str | None = None, extra: str = "") -> object:
    (root / "orig").mkdir(parents=True)
    (root / "orig" / "flash.bin").write_bytes(image)
    (root / "dk.toml").write_text(
        '[project]\nname = "flash"\nplatform = "spi_nor"\n\n'
        f'[target]\noriginal = "orig/flash.bin"\nsha1 = "{sha1 or hashlib.sha1(image).hexdigest()}"\n\n'
        f'[platform]\npart = "TESTPART"\n{extra}'
    )
    return load(root)


class SpiNorTest(unittest.TestCase):
    def test_geometry_comes_from_the_length(self):
        self.assertEqual(geometry(8 * 1024 * 1024), "geometry: 64 Mbit SPI NOR (page 256, sector 4096, block 65536)")
        self.assertEqual(geometry(128 * 1024), "geometry: 1 Mbit SPI NOR (page 256, sector 4096, block 65536)")
        self.assertIn("not a standard", geometry(1000))

    def test_map_carve_and_dis(self):
        image = sample_image()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_spi(root, image)
            code, text = run(mapcmd.COMMAND, cfg, [])
            self.assertEqual(code, 0)
            for want in ("part: TESTPART (label from dk.toml, not read from the image)",
                         "erased: byte 0xff, runs >= 4096: 4",
                         "at 0x00000000  link 0x80000000  reset 0x00000060",
                         "at 0x00001100  gziphead  body 0x00001110-0x00001114  inflates to 8; payload is not a gzip member",
                         "header crc ok", "demo.txt", "same as 0x00003400", "bootcmd=run", "next: ./dk carve"):
                self.assertIn(want, text)

            self.assertEqual(run(carvecmd.COMMAND, cfg, [])[0], 0)
            manifest = (root / "build" / "spi" / "manifest.txt").read_text().splitlines()
            self.assertEqual(manifest[0], f"image_sha1 {hashlib.sha1(image).hexdigest()}")
            payload = root / "build" / "spi" / "regions" / "00001110.gziphead"
            self.assertEqual(payload.read_bytes(), b"WXYZ")
            self.assertEqual(run(carvecmd.COMMAND, cfg, ["--check"])[0], 0)
            payload.write_bytes(b"WXYY")
            self.assertEqual(run(carvecmd.COMMAND, cfg, ["--check"])[0], 1)

            code, text = run(discmd.COMMAND, cfg, ["--engine", "builtin", "--length", "8"])
            self.assertEqual(code, 0)
            self.assertIn("80000000  160000ea  b 0x80000060", text)
            self.assertIn("arch arm engine builtin", text)

    def test_map_reports_a_sha1_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_spi(Path(tmp), sample_image(), sha1="0" * 40)
            code, text = run(mapcmd.COMMAND, cfg, ["--limit", "5"])
        self.assertEqual(code, 1)
        self.assertIn("mismatch dk.toml", text)

    def test_erased_byte_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_spi(Path(tmp), b"\x11\x22" + b"\x00" * 5000, extra="erased = 0\n")
            code, text = run(mapcmd.COMMAND, cfg, [])
        self.assertEqual(code, 0)
        self.assertIn("programmed: 1\n  0x00000000-0x00000002", text)

    def test_status_says_it_builds_on_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_spi(Path(tmp), sample_image())
            code, text = run(status.COMMAND, cfg, [])
        self.assertEqual(code, 0)
        self.assertIn("flash (platform spi_nor on raw)", text)
        self.assertIn("next: ./dk carve", text)


if __name__ == "__main__":
    unittest.main()
