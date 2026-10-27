import unittest

from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
from tools.dk.platforms.gc_mwcc.linkproof import prove, sda_bases


def words(*hexes: str) -> bytes:
    return bytes.fromhex("".join(hexes))


class LinkProofTest(unittest.TestCase):
    def test_sda_bases_from_register_setup(self):
        init = words("3c20803a", "60215740", "3c408039", "6042ab20", "3da08039", "61ad4a40", "4e800020") + bytes(40)
        self.assertEqual(sda_bases(lambda a, n: init[a - 0x100:a - 0x100 + n], 0x100), (0x80394A40, 0x8039AB20))

    def test_relocated_like_the_final_link(self):
        # lis r3, g@ha; addi r3, r3, g@l; lfs f1, @12@sda21(r2); bl ext; blr
        ours = Function("f", ".text", words("3c600000", "38630000", "c0200000", "48000001", "4e800020"), [
            Reloc(0x2, 6, "g", 0), Reloc(0x6, 4, "g", 0),
            Reloc(0x8, 109, "@12", 0, ".sdata2", 0x4),        # our literal, placed by the unit's split
            Reloc(0xC, 10, "ext", 0)])
        names = {"g": (0x80250010, ".data"), "ext": (0x80100000, ".text"), "@12": (0x80000000, ".data")}
        placed = {".text": 0x80001000, ".sdata2": 0x80395000}
        sda = (0x80394A40, 0x8039AB20)
        original = words("3c608025", "38630010", "c022a4e4", "480feff5", "4e800020")
        dol = lambda a, n: original[:n]
        # "@12" is local to our object: it must not resolve to another unit's symbols.txt "@12"
        self.assertTrue(prove(ours, 0x80001000, {"@12"}, placed, names, sda, dol).ok)
        moved = prove(ours, 0x80001000, set(), placed, names, sda, dol)
        self.assertFalse(moved.ok)
        self.assertIn("+0x8", moved.detail)
        missing = prove(ours, 0x80001000, {"@12"}, placed, {k: v for k, v in names.items() if k != "ext"}, sda, dol)
        self.assertIn("undefined at link", missing.detail)


if __name__ == "__main__":
    unittest.main()


class LiteralRowsTest(unittest.TestCase):
    def test_marks_the_first_literal_out_of_order(self):
        from tools.dk.platforms.gc_mwcc.diffcmd import literal_rows
        from tools.dk.platforms.gc_mwcc.elf import ElfSection, ElfSymbol

        class Fake:
            def __init__(self, data, names, users):
                self.sections = (ElfSection(0, "", 0, 0, 0, 0, 0, b""), ElfSection(1, ".text", 1, 0, 8, 0, 0, bytes(8)),
                                 ElfSection(2, ".sdata2", 1, 0, len(data), 0, 0, data))
                self.symbols = tuple([ElfSymbol(n, 4 * i, 4, 1, 0, 2) for i, n in enumerate(names)] +
                                     [ElfSymbol(f, 0, 4, 2, 1, 1) for f in users])
                self._users = users

            def section(self, name):
                return next((s for s in self.sections if s.name == name), None)

            def functions(self):
                return {f: Function(f, ".text", bytes(4), [Reloc(0, 109, "x", 0, ".sdata2", off)])
                        for f, off in self._users.items()}

        one, two = bytes.fromhex("3f800000"), bytes.fromhex("40000000")
        target = Fake(one + two, ["lbl_a", "lbl_b"], {"f": 0})
        ours = Fake(two + one, ["@1", "@2"], {"f": 4})
        rows = literal_rows(target, ours, ".sdata2")
        self.assertTrue(rows[1].lstrip().startswith("!= +0x0"))
        self.assertIn("(1f) (f)", rows[1])
