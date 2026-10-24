import struct
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_dol
from tools.dk.platforms.gc_mwcc.dol import Dol
from tools.dk.platforms.gc_mwcc.dtk_symbols import rename_symbols
from tools.dk.platforms.gc_mwcc.rtti import analyze, mangle

TEXT = 0x80003100
DATA = 0x80004000
SDATA = 0x80005000
FN = [TEXT + 4 * i for i in range(4)]


def build_fixture() -> tuple[bytes, tuple[int, int, int]]:
    """Base <- Derived (Derived also has Mixin at +0x10). Plus a decoy {string, 0} pair."""
    strings = b"Base\0\0\0\0Derived\0Mixin\0\0\0NotAClass\0\0\0"
    s_base, s_derived, s_mixin, s_decoy = DATA, DATA + 8, DATA + 16, DATA + 24
    data = bytearray(strings.ljust(0x40, b"\0"))
    bases_derived = DATA + len(data)
    data += struct.pack(">5I", SDATA + 0, 0, SDATA + 16, 0x10, 0)       # Base@0, Mixin@0x10, end
    vt_base = DATA + len(data)
    data += struct.pack(">4I", SDATA + 0, 0, FN[0], FN[1])
    vt_derived = DATA + len(data)
    data += struct.pack(">4I", SDATA + 8, 0, FN[2], FN[1])
    vt_derived_mixin = DATA + len(data)
    data += struct.pack(">3I", SDATA + 8, (-0x10) & 0xFFFFFFFF, FN[3])
    data += struct.pack(">I", 0x12345678)
    sdata = struct.pack(">8I",
                        s_base, 0,                    # RTTI Base
                        s_derived, bases_derived,     # RTTI Derived
                        s_mixin, 0,                   # RTTI Mixin (no vtable, reached via bases)
                        s_decoy, 0)                   # decoy: never referenced
    text = struct.pack(">4I", *([0x4E800020] * 4))
    dol = make_dol([("T1", TEXT, text), ("D5", DATA, bytes(data)), ("D6", SDATA, sdata)])
    return dol, (vt_base, vt_derived, vt_derived_mixin)


class RttiTest(unittest.TestCase):
    def test_hierarchy(self):
        raw, (vt_base, vt_derived, vt_mixin) = build_fixture()
        classes = analyze(Dol.parse(raw), set(FN))
        by_name = {c.name: c for c in classes.values()}
        self.assertEqual(sorted(by_name), ["Base", "Derived", "Mixin"])   # decoy rejected
        derived = by_name["Derived"]
        self.assertEqual(derived.bases, [(SDATA, 0), (SDATA + 16, 0x10)])
        self.assertEqual([(v.address, v.offset, v.slots) for v in derived.vtables],
                         [(vt_derived, 0, [FN[2], FN[1]]), (vt_mixin, -0x10, [FN[3]])])
        self.assertEqual(derived.primary.address, vt_derived)
        self.assertEqual(by_name["Base"].primary.slots, [FN[0], FN[1]])
        self.assertEqual(by_name["Mixin"].vtables, [])

    def test_mangle(self):
        self.assertEqual(mangle("CarManager"), "10CarManager")
        self.assertEqual(mangle("AI::AICop"), "Q22AI5AICop")
        self.assertIsNone(mangle("AI::LocationManager<AI::CopLocation>"))
        self.assertIsNone(mangle("A::"))


class RenameTest(unittest.TestCase):
    def test_only_auto_names_and_no_duplicates(self):
        text = (
            "lbl_80004000 = .data:0x80004000; // type:object size:0x8\n"
            "KeepMe = .data:0x80004008; // type:object size:0x8\n"
            "lbl_80004010 = .data:0x80004010; // type:object size:0x8\n"
            "__vt__4Base = .data:0x80004020; // type:object size:0x8\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "symbols.txt"
            path.write_text(text)
            new, applied = rename_symbols(path, {0x80004000: "__RTTI__4Base", 0x80004008: "X", 0x80004010: "__vt__4Base"})
        self.assertEqual(applied, 1)
        self.assertIn("__RTTI__4Base = .data:0x80004000; // type:object size:0x8\n", new)
        self.assertIn("KeepMe = .data:0x80004008", new)          # user name kept
        self.assertIn("lbl_80004010 = .data:0x80004010", new)    # name already taken -> skipped


if __name__ == "__main__":
    unittest.main()
