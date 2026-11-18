"""The pre-export scan: names, real addresses and copied code words in the files an export would publish."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.dk.core import exportscan

CODE = [0x7C0802A6, 0x9421FFF0, 0x93E1000C, 0x7C7F1B78, 0x4E800020]       # synthetic "original" words


def binary(tmp: Path) -> Path:
    p = tmp / "orig.bin"
    p.write_bytes(b"".join(w.to_bytes(4, "big") for w in [0, 0, *CODE, 0x20, 0x28, 0x58, 0, 0, 0]))
    return p


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = exportscan.word_index(binary(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_keeps_instruction_like_triples_only(self):
        self.assertIn(tuple(CODE[:3]), self.index)
        self.assertNotIn((0x20, 0x28, 0x58), self.index)                  # small constants: in any binary
        self.assertNotIn((0, 0, 0), self.index)

    def scan(self, text: str, allow=frozenset()):
        return exportscan.scan_files({"tests/x.py": text.encode()}, ["Zorblax", "docs/zb"], {"80123450"},
                                     self.index, set(allow))

    def test_each_kind_is_found(self):
        listing = "\n".join(f"/* 8010{i:04X} 0000{i:04X}  {' '.join(f'{w:08X}'[k:k + 2] for k in (0, 2, 4, 6))} */\tinsn"
                            for i, w in enumerate(CODE[:3]))
        kinds = {h.kind for h in self.scan(listing)}
        self.assertEqual(kinds, {"code"})
        self.assertEqual([h.kind for h in self.scan("w = [0x7C0802A6, 0x9421FFF0,\n     0x93E1000C]\n")], ["code"])
        self.assertEqual([h.kind for h in self.scan("bytes = '7C 08 02 A6 94 21 FF F0 93 E1 00 0C'\n")], ["code"])
        hits = self.scan("# the ZORBLAX camera at 0x80123450, see docs/zb/x.md\n")
        self.assertEqual(sorted((h.kind, h.what) for h in hits),
                         [("addresses", "80123450"), ("names", "ZORBLAX"), ("names", "docs/zb")])

    def test_far_apart_words_synthetic_addresses_and_allowed_hits_pass(self):
        self.assertEqual(self.scan("a = 0x7C0802A6\n\n\nb = 0x9421FFF0\n\n\nc = 0x93E1000C\n"), [])
        self.assertEqual(self.scan("addr = 0x80700000\n"), [])
        self.assertEqual(self.scan("Zorblax 0x80123450\n", allow={"zorblax", "80123450"}), [])

    def test_banned_words_come_from_dk_toml(self):
        from tests.helpers import make_project
        with tempfile.TemporaryDirectory() as d:
            cfg = make_project(Path(d))
            text = (Path(d) / "dk.toml").read_text()
            (Path(d) / "dk.toml").write_text(text + '\n[toolkit]\nexport_banned = ["Zorblax", "ab"]\n')
            from tools.dk.core.config import load
            words = exportscan.banned_words(load(Path(d)))
            self.assertIn("Zorblax", words)
            self.assertNotIn("ab", words)                                   # too short to scan for
            self.assertIn(cfg.name, words)


if __name__ == "__main__":
    unittest.main()
