import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_dol, make_project
from tools.dk.platforms.gc_mwcc import matching as M
from tools.dk.platforms.gc_mwcc.elf import Elf, ElfSection, ElfSymbol

CODE = bytes.fromhex("38600001388000024e800020")      # li r3, 1; li r4, 2; blr


def our_object(name: str, force_active: bool) -> Elf:
    """One weak function, flagged force-active in mwcc's `.comment` (8 bytes per symbol, bit 0x08 at +5)."""
    comment = bytearray(b"CodeWarrior".ljust(0x2C, b"\0") + bytes(16))
    if force_active:
        comment[0x2C + 8 + 5] = 0x08
    sections = (ElfSection(0, "", 0, 0, 0, 0, 0, b""), ElfSection(1, ".text", 1, 0, len(CODE), 0, 0, CODE),
                ElfSection(2, ".comment", 1, 0, len(comment), 0, 0, bytes(comment)))
    return Elf(sections, (ElfSymbol("", 0, 0, 0, 0, 0), ElfSymbol(name, 0, len(CODE), 2, 2, 1)))


class ProjectCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = make_project(root, files={
            "config/symbols.txt": "fn_80003100 = .text:0x80003100; // type:function size:0xC\n"
                                  "named__Fv = .text:0x80003200; // type:function size:0xC\n",
            "config/splits.txt": "",
            "objdiff.json": '{"units": []}',
        })
        dol = make_dol([("T0", 0x80003100, CODE + bytes(0x100 - len(CODE)) + CODE)])
        (root / "orig").mkdir()
        (root / "orig" / "main.dol").write_bytes(dol)
        M._INDEX.clear()
        M._ORIGINAL.clear()
        M._LINKED.clear()

    def tearDown(self):
        self.tmp.cleanup()


class UnnamedCopyTest(ProjectCase):
    def test_a_kept_weak_copy_names_the_placeholder_holding_the_original(self):
        notes = M.unnamed_copies(self.cfg, our_object("__dt__4ItemFv", True), ["__dt__4ItemFv"])
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].startswith("rename fn_80003100 to __dt__4ItemFv"))

    def test_nothing_when_the_name_is_known_or_the_copy_is_not_kept(self):
        self.assertEqual(M.unnamed_copies(self.cfg, our_object("named__Fv", True), ["named__Fv"]), [])
        self.assertEqual(M.unnamed_copies(self.cfg, our_object("__dt__4ItemFv", False), ["__dt__4ItemFv"]), [])
        self.assertEqual(M.unnamed_copies(self.cfg, our_object("__dt__4ItemFv", True), []), [])


def with_relocs(objects: list[tuple[str, int, int, int]], relocs: list[tuple[int, str]], undefined: list[str]) -> Elf:
    """A `.data` section holding `objects` (name, offset, size, type) and relocations
    (offset, symbol) against them or against `undefined` names."""
    import struct
    syms = [ElfSymbol("", 0, 0, 0, 0, 0)] + [ElfSymbol(n, off, size, typ, 1, 1) for n, off, size, typ in objects]
    syms += [ElfSymbol(n, 0, 0, 0, 1, 0) for n in undefined]
    index = {x.name: i for i, x in enumerate(syms)}
    rela = b"".join(struct.pack(">IIi", off, (index[name] << 8) | 1, 0) for off, name in relocs)
    sections = (ElfSection(0, "", 0, 0, 0, 0, 0, b""), ElfSection(1, ".data", 1, 0, 0x40, 0, 0, bytes(0x40)),
                ElfSection(2, ".rela.data", 4, 0, len(rela), 0, 1, rela))
    return Elf(sections, tuple(syms))


class UndefinedAtLinkTest(ProjectCase):
    def test_a_vtable_slot_symbols_txt_lacks_names_the_placeholder_to_rename(self):
        ours = with_relocs([("__vt__3Foo", 0, 0x10, 1)], [(8, "vf1__3FooFv")], ["vf1__3FooFv"])
        target = with_relocs([("__vt__3Foo", 0, 0x10, 1)], [(8, "fn_80003100")], ["fn_80003100"])
        notes = M.undefined_at_link(self.cfg, target, ours)
        self.assertEqual(notes, ["undefined at link: vf1__3FooFv is not in symbols.txt: rename fn_80003100 to vf1__3FooFv"])

    def test_the_units_own_data_used_as_an_extern_is_undefined(self):
        ours = with_relocs([("__vt__3Foo", 0, 0x10, 1)], [(8, "lbl_80004000")], ["lbl_80004000"])
        target = with_relocs([("__vt__3Foo", 0, 0x10, 1), ("lbl_80004000", 0x20, 0xC, 1)], [(8, "lbl_80004000")], [])
        notes = M.undefined_at_link(self.cfg, target, ours)
        self.assertEqual(len(notes), 1)
        self.assertIn("ours references lbl_80004000, which the original defines in this unit", notes[0])

    def test_names_the_original_also_leaves_to_its_link_are_fine(self):
        ours = with_relocs([("__vt__3Foo", 0, 0x10, 1)], [(8, "_stack_addr")], ["_stack_addr"])
        target = with_relocs([("__vt__3Foo", 0, 0x10, 1)], [(8, "_stack_addr")], ["_stack_addr"])
        self.assertEqual(M.undefined_at_link(self.cfg, target, ours), [])


class DroppedByLinkTest(unittest.TestCase):
    def test_unreferenced_functions_get_the_force_active_fix(self):
        from tools.dk.platforms.gc_mwcc.acceptcmd import dropped_lines
        lines = dropped_lines({"a__Fv": 0x40, "b__Fv": 0x10, "w__Fv": 0x8, "gap_x": 4}, {"b__Fv"}, {"a__Fv": 1, "w__Fv": 2})
        self.assertEqual(lines[0], "the link lacks 0x48 bytes of this unit's functions: a__Fv (0x40), w__Fv (0x8)")
        self.assertTrue(lines[1].endswith("./dk symbol force-active a__Fv"))
        self.assertIn("w__Fv", lines[2])
        self.assertEqual(dropped_lines({"a__Fv": 4}, {"a__Fv"}, {}), [])


class DataShortfallTest(ProjectCase):
    def test_a_section_that_links_short_names_what_only_discarded_copies_used(self):
        from types import SimpleNamespace
        from tests.test_zero_filled_kept import GLOBAL, LOCAL, WEAK, make_object
        from tools.dk.platforms.gc_mwcc.acceptcmd import data_shortfall
        root = Path(self.tmp.name)
        (root / "config" / "symbols.txt").write_text(
            "before = .sdata:0x80001000; // type:object size:0x8\n"
            "kept = .sdata:0x80001008; // type:object size:0x8\n"
            "after = .sdata:0x80001018; // type:object size:0x8\n")
        (root / "config" / "splits.txt").write_text("u.cpp:\n\t.sdata      start:0x80001008 end:0x80001018\n")
        target = make_object({".sdata": bytes(0x10)}, [("kept", ".sdata", 0, 8, GLOBAL), ("lbl_80001010", ".sdata", 8, 8, LOCAL)])
        ours = make_object({".data": bytes(4), ".sdata": bytes(0x10)},
                           [("__vt__1X", ".data", 0, 4, WEAK), ("kept", ".sdata", 0, 8, GLOBAL), ("@1", ".sdata", 8, 8, LOCAL)],
                           [(".data", 0, "@1")])
        (root / "t.o").write_bytes(target)
        (root / "b.o").write_bytes(ours)
        linked = Elf((ElfSection(0, "", 0, 0, 0, 0, 0, b""), ElfSection(1, ".sdata", 1, 0, 0x18, 0, 0, b"")),
                     (ElfSymbol("", 0, 0, 0, 0, 0), ElfSymbol("before", 0x80001000, 8, 1, 1, 1),
                      ElfSymbol("after", 0x80001010, 8, 1, 1, 1)))
        unit = {"target_path": "t.o", "base_path": "b.o", "metadata": {"source_path": "src/u.cpp"}}
        lines = data_shortfall(SimpleNamespace(root=root, platform_options=self.cfg.platform_options), unit, linked)
        self.assertEqual(lines, ["this unit's .sdata links 0x8 bytes short of the original's 0x10: the original kept "
                                 "what only discarded weak copies use: @1 (0x8)"])


class KeptByOriginalTest(unittest.TestCase):
    def test_small_data_only_a_discarded_duplicate_uses_but_the_original_kept(self):
        from tests.test_zero_filled_kept import LOCAL, WEAK, make_object
        name = b"GButton\0"
        ours = Elf.parse(make_object({".text": bytes(8), ".sdata": name},
                                     [("ClassName__7GButtonFv", ".text", 0, 8, WEAK), ("@1", ".sdata", 0, 8, LOCAL)],
                                     [(".text", 0, "@1")]))
        target = Elf.parse(make_object({".text": bytes(8), ".sdata": name}, [("lbl_80001000", ".sdata", 0, 8, LOCAL)]))
        [note] = M.kept_by_original(target, ours, {"ClassName__7GButtonFv"})
        self.assertIn("the original keeps 0x8 bytes of .sdata that only discarded weak copies use (@1)", note)
        self.assertEqual(M.kept_by_original(target, ours, set()), [])          # merely dead-stripped: kept anyway
        in_data = Elf.parse(make_object({".text": bytes(8), ".data": name},
                                        [("ClassName__7GButtonFv", ".text", 0, 8, WEAK), ("@1", ".data", 0, 8, LOCAL)],
                                        [(".text", 0, "@1")]))
        self.assertEqual(M.kept_by_original(target, in_data, {"ClassName__7GButtonFv"}), [])


class VtableOrderTest(unittest.TestCase):
    def test_a_vtable_behind_weak_copies_the_target_lacks_names_the_key_function_fix(self):
        from tests.test_zero_filled_kept import GLOBAL, WEAK, make_object
        ours = Elf.parse(make_object({".data": bytes(0x18)}, [("__vt__1G", ".data", 0, 0xC, WEAK), ("__vt__1L", ".data", 0xC, 0xC, GLOBAL)]))
        target = Elf.parse(make_object({".data": bytes(0x18)}, [("__vt__1L", ".data", 0, 0xC, GLOBAL)]))
        [note] = M.vtables_out_of_order(target, ours)
        self.assertTrue(note.startswith("__vt__1L sits after the weak copies __vt__1G in ours, earlier in the target"))
        self.assertIn("the first one L declares", note)
        self.assertEqual(M.vtables_out_of_order(ours, ours), [])


if __name__ == "__main__":
    unittest.main()
