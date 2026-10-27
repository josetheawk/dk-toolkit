import tempfile
import tomllib
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.matching import UnitScore, FunctionScore, retarget
from tools.dk.platforms.gc_mwcc.matchcmds import _version_key
from tools.dk.platforms.gc_mwcc.units import (
    add_object, add_split_unit, read_units_file, remove_object, remove_split_unit, render_units_file,
    replace_split_unit, set_status,
)

SPLITS = """\
Sections:
\t.text       type:code align:32
\t.data       type:data align:32

b.c: comment:0
\t.text       start:0x80003200 end:0x80003300 align:16

d.c:
\t.text       start:0x80003500 end:0x80003600
"""

UNITS = """\
# header kept

[[lib]]
name = "Lib"
mw_version = "GC/1.3.2"
cflags = "base"
objects = [
    { path = "b.c", status = "NonMatching" },
]
"""


class SplitsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "splits.txt"
        self.path.write_text(SPLITS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_insert_by_address_keeps_existing_lines(self):
        text = add_split_unit(self.path, "c.c", [(".text", 0x80003300, 0x80003500), (".data", 0x80004000, 0x80004010)])
        self.assertIn("b.c: comment:0\n\t.text       start:0x80003200 end:0x80003300 align:16\n", text)
        self.assertLess(text.index("b.c:"), text.index("c.c:"))
        self.assertLess(text.index("c.c:"), text.index("d.c:"))
        self.assertIn("\t.data       start:0x80004000 end:0x80004010", text)

    def test_insert_first_and_last(self):
        first = add_split_unit(self.path, "a.c", [(".text", 0x80003100, 0x80003200)])
        self.assertLess(first.index("a.c:"), first.index("b.c:"))
        last = add_split_unit(self.path, "e.c", [(".text", 0x80003600, 0x80003700)])
        self.assertTrue(last.rstrip().endswith("start:0x80003600 end:0x80003700"))

    def test_rejects_overlap_and_duplicate(self):
        with self.assertRaises(ValueError):
            add_split_unit(self.path, "x.c", [(".text", 0x800032F0, 0x80003310)])
        with self.assertRaises(ValueError):
            add_split_unit(self.path, "b.c", [(".text", 0x80009000, 0x80009010)])


class EditSplitsTest(SplitsTest):
    def test_replace_keeps_header_and_others(self):
        text = replace_split_unit(self.path, "b.c", [(".text", 0x80003200, 0x80003280), (".data", 0x80004000, 0x80004004)])
        self.assertIn("b.c: comment:0\n\t.text       start:0x80003200 end:0x80003280\n\t.data       start:0x80004000 end:0x80004004\n", text)
        self.assertIn("d.c:\n\t.text       start:0x80003500 end:0x80003600", text)
        with self.assertRaises(ValueError):
            replace_split_unit(self.path, "b.c", [(".text", 0x80003500, 0x80003510)])     # overlaps d.c
        with self.assertRaises(ValueError):
            replace_split_unit(self.path, "zz.c", [(".text", 0x80009000, 0x80009004)])

    def test_replace_refuses_to_drop_a_section_silently(self):
        with_data = replace_split_unit(self.path, "b.c", [(".text", 0x80003200, 0x80003280), (".data", 0x80004000, 0x80004004)])
        self.path.write_text(with_data)
        with self.assertRaises(ValueError):
            replace_split_unit(self.path, "b.c", [(".text", 0x80003200, 0x80003280)])
        text = replace_split_unit(self.path, "b.c", [(".text", 0x80003200, 0x80003280)], drop=(".data",))
        self.assertNotIn("0x80004000", text)

    def test_remove(self):
        text = remove_split_unit(self.path, "b.c")
        self.assertNotIn("b.c", text)
        self.assertIn("d.c:", text)


class UnitsFileTest(unittest.TestCase):
    def test_roundtrip_add_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "units.toml"
            path.write_text(UNITS)
            units = read_units_file(path)
            self.assertEqual(render_units_file(units), UNITS)          # stable rendering
            self.assertTrue(add_object(units, "Lib", "c.c", "NonMatching", {}))
            self.assertFalse(add_object(units, "Lib", "c.c", "NonMatching", {}))
            self.assertTrue(add_object(units, "New", "n.c", "NonMatching", {"cflags": "runtime", "mw_version": None}))
            self.assertTrue(set_status(units, "b.c", "Matching"))
            self.assertTrue(remove_object(units, "n.c"))
            self.assertIsNone(units.lib("New"))                                      # empty lib dropped
            self.assertTrue(add_object(units, "New", "n.c", "NonMatching", {"cflags": "runtime", "mw_version": None}))
            text = render_units_file(units)
            data = tomllib.loads(text)
            self.assertEqual(data["lib"][0]["objects"][0], {"path": "b.c", "status": "Matching"})
            self.assertEqual(data["lib"][1], {"name": "New", "cflags": "runtime", "objects": [{"path": "n.c", "status": "NonMatching"}]})
            self.assertTrue(text.startswith("# header kept\n"))
            with self.assertRaises(ValueError):
                add_object(units, "Lib", "z.c", "Done", {})


class MatchingHelpersTest(unittest.TestCase):
    def test_retarget(self):
        cmd = ("build/tools/wibo build/compilers/GC/1.3.2/mwcceppc.exe -O4,p -lang=c -MMD "
               "-c src/a.c -o build/X/src")
        args = retarget(cmd, Path("/tmp/out"), "GC/2.0", ("-inline", "deferred"))
        self.assertEqual(args, ["build/tools/wibo", "build/compilers/GC/2.0/mwcceppc.exe", "-O4,p", "-lang=c",
                                "-inline", "deferred", "-c", "src/a.c", "-o", "/tmp/out"])

    def test_version_order(self):
        versions = ["2.0", "1.3.2r", "1.2.5n", "1.3", "3.0a3", "1.3.2", "1.2.5", "2.0p1", "1.10"]
        self.assertEqual(sorted(versions, key=_version_key),
                         ["1.2.5", "1.2.5n", "1.3", "1.3.2", "1.3.2r", "1.10", "2.0", "2.0p1", "3.0a3"])

    def test_exact_requires_layout(self):
        s = UnitScore("u", [FunctionScore("f", 8, 100.0)], {".data": 100.0})
        self.assertTrue(s.exact)
        s.layout_problems.append("f: moved")
        self.assertFalse(s.exact)
        self.assertEqual(UnitScore("u", [FunctionScore("f", 8, 50.0), FunctionScore("g", 8, 100.0)]).code_percent, 75.0)



class WeakExtrasTest(unittest.TestCase):
    def test_only_weak_extras(self):
        import base64
        from tools.dk.platforms.gc_mwcc.matching import _only_weak_extras
        b64 = lambda raw: base64.b64encode(raw).decode()
        common = {"data": b64(b"x" * 72), "size": "72"}
        weak = [(72, 92, "__vt__4Base")]
        # trailing weak copy only in the compiled object
        left = {"data_diff": [common, {"kind": "DIFF_INSERT", "size": "20"}]}
        right = {"data_diff": [common, {"kind": "DIFF_INSERT", "data": b64(b"\0" * 20), "size": "20"}]}
        self.assertTrue(_only_weak_extras(left, right, weak))
        right_real = {"data_diff": [common, {"kind": "DIFF_INSERT", "data": b64(b"r" * 20), "size": "20"}]}
        self.assertFalse(_only_weak_extras(left, right_real, [(0, 20, "__vt__4Base")]))   # real bytes elsewhere
        # target zero hole the size of the weak copy (original linker zero-filled it)
        hole = {"data_diff": [common, {"kind": "DIFF_DELETE", "data": b64(b"\0" * 20), "size": "20"}]}
        self.assertTrue(_only_weak_extras(hole, right, weak))
        # real target-only bytes, or a zero block of the wrong size, are not explained
        real = {"data_diff": [{"kind": "DIFF_DELETE", "data": b64(b"text\0\0\0\0"), "size": "8"}]}
        self.assertFalse(_only_weak_extras(real, right, weak))
        wrong = {"data_diff": [{"kind": "DIFF_DELETE", "data": b64(b"\0" * 8), "size": "8"}]}
        self.assertFalse(_only_weak_extras(wrong, right, weak))
        # an all-zero base-only chunk reported at another offset is explained by size
        shifted = {"data_diff": [{"kind": "DIFF_INSERT", "data": b64(bytes(8)), "size": "8"}, common]}
        self.assertTrue(_only_weak_extras({"data_diff": [common]}, shifted, [(72, 80, "@RTTI")]))
        # a chunk that starts in the zero padding byte before the first extra
        pad_first = {"data_diff": [{"data": b64(b"x" * 31), "size": "31"},
                                   {"kind": "DIFF_INSERT", "data": b64(bytes(1) + b"v" * 20), "size": "21"}]}
        self.assertTrue(_only_weak_extras({"data_diff": [{"data": b64(b"x" * 31), "size": "31"}]}, pad_first, [(32, 52, "__vt__1G")]))
        nonzero = {"data_diff": [{"kind": "DIFF_INSERT", "data": b64(b"abcdefgh"), "size": "8"}, common]}
        self.assertFalse(_only_weak_extras({"data_diff": [common]}, nonzero, [(72, 80, "@RTTI")]))


class SymbolEditTest(unittest.TestCase):
    TEXT = ("lbl_80250F40 = .data:0x80250F40; // type:object size:0x84\n"
            "__vt__4Base = .data:0x80250FC4; // type:object size:0x14\n")

    def test_split_and_size(self):
        from tools.dk.platforms.gc_mwcc.symbolcmd import size_text, split_text
        text, _ = split_text(self.TEXT, 0x80250F70, None)
        self.assertIn("lbl_80250F40 = .data:0x80250F40; // type:object size:0x30\n", text)
        self.assertIn("lbl_80250F70 = .data:0x80250F70; // type:object size:0x54\n", text)
        with self.assertRaises(ValueError):
            split_text(self.TEXT, 0x80250FC8, None)                 # named symbol: not split
        with self.assertRaises(ValueError):
            split_text(self.TEXT, 0x80250F40, None)                 # not strictly inside
        self.assertIn("__vt__4Base = .data:0x80250FC4; // type:object size:0x18", size_text(self.TEXT, "__vt__4Base", 0x18)[0])

    U64 = ("bound = .sbss:0x80392970; // type:object size:0x4 data:4byte\n"
           "lbl_80392974 = .sbss:0x80392974; // type:object size:0x4 data:4byte\n"
           "named = .sbss:0x80392978; // type:object size:0x4\n")

    def test_size_absorbs_auto_labels_only(self):
        from tools.dk.platforms.gc_mwcc.symbolcmd import size_text
        text, removed = size_text(self.U64, "bound", 0x8)
        self.assertEqual(removed, ["lbl_80392974"])
        self.assertEqual(text, "bound = .sbss:0x80392970; // type:object size:0x8 data:4byte\n"
                               "named = .sbss:0x80392978; // type:object size:0x4\n")
        with self.assertRaises(ValueError):
            size_text(self.U64, "bound", 0xC)                       # would swallow a named symbol
        with self.assertRaises(ValueError):
            size_text(self.U64, "bound", 0x6)                       # would cut an auto label

    def test_add_object_and_label(self):
        from tools.dk.platforms.gc_mwcc.symbolcmd import add_text
        text, removed = add_text(self.U64, "big", ".sbss", 0x80392974, 0x4, "local")
        self.assertEqual(removed, ["lbl_80392974"])
        self.assertIn("bound = .sbss:0x80392970; // type:object size:0x4 data:4byte\n"
                      "big = .sbss:0x80392974; // type:object size:0x4 scope:local\nnamed", text)
        text, _ = add_text(self.U64, "lab", ".sbss", 0x80392970, None, "global")
        self.assertTrue(text.startswith("bound = .sbss:0x80392970; // type:object size:0x4 data:4byte\n"
                                        "lab = .sbss:0x80392970; // type:label scope:global\n"))
        with self.assertRaises(ValueError):
            add_text(self.U64, "named", ".sbss", 0x80392990, 0x4, None)   # name taken
        with self.assertRaises(ValueError):
            add_text(self.U64, "x", ".sbss", 0x80392976, 0x4, None)       # cuts a label and a named one


class RewriteSourcesTest(unittest.TestCase):
    def test_plain_names_rewritten_mangled_reported(self):
        from tests.helpers import make_project
        from tools.dk.platforms.gc_mwcc.renamecmd import rewrite_sources
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp), files={
                "src/game/a.cpp": 'extern "C" void fn_801C3E1C(int);\nextern "C" void fn_800ED408(void*);\n'
                                  "void f() { fn_801C3E1C(1); fn_801C3E1Cx(); }\n",
                "include/game/a.h": "extern int lbl_80391C74;\n",
            })
            changed, manual = rewrite_sources(cfg, [("fn_801C3E1C", "OSAllocFromHeap"), ("lbl_80391C74", "gThing"),
                                                    ("fn_800ED408", "__dl__FPv"), ("Named", "Other")])
            self.assertEqual(changed, ["include/game/a.h", "src/game/a.cpp"])
            text = (Path(tmp) / "src/game/a.cpp").read_text()
            self.assertIn('extern "C" void OSAllocFromHeap(int);', text)
            self.assertIn("fn_801C3E1Cx()", text)                       # whole words only
            self.assertIn("fn_800ED408", text)                           # mangled target: not rewritten
            self.assertEqual(manual, ["src/game/a.cpp: fn_800ED408"])
            self.assertIn("gThing", (Path(tmp) / "include/game/a.h").read_text())

    def test_a_declaration_renamed_into_a_repeat_is_dropped(self):
        from tests.helpers import make_project
        from tools.dk.platforms.gc_mwcc.renamecmd import rewrite_sources
        src = ('extern "C" void OSInitMutex(void*);\nextern int lbl_1;\nextern int lbl_1;\n'
               'extern "C" void fn_801C3E1C(void*);\nvoid f(void* m) { fn_801C3E1C(m); }\n')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp), files={"src/game/a.cpp": src})
            rewrite_sources(cfg, [("fn_801C3E1C", "OSInitMutex")])
            text = (Path(tmp) / "src/game/a.cpp").read_text()
            self.assertEqual(text.count('extern "C" void OSInitMutex(void*);'), 1)
            self.assertIn("OSInitMutex(m);", text)
            self.assertEqual(text.count("extern int lbl_1;"), 2)          # an older repeat is left alone


class FreshnessTest(unittest.TestCase):
    def test_stale_split_output_removed(self):
        import os
        from tests.helpers import make_project
        from tools.dk.platforms.gc_mwcc.freshness import ensure_split_fresh
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_project(Path(tmp), files={"config/symbols.txt": "", "config/splits.txt": "",
                                                 "config/DEMO01/config.yml": "", "build/DEMO01/config.json": "{}"})
            out = Path(tmp) / "build/DEMO01/config.json"
            os.utime(out, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
            self.assertFalse(ensure_split_fresh(cfg))                       # output newer: keep
            os.utime(Path(tmp) / "config/symbols.txt", ns=(3_000_000_000_000_000_000, 3_000_000_000_000_000_000))
            self.assertTrue(ensure_split_fresh(cfg))                        # symbols edited later: force re-split
            self.assertFalse(out.exists())


class CheckAllTest(unittest.TestCase):
    def test_linked_regression_fails_wip_only_reported(self):
        from tools.dk.platforms.gc_mwcc.matchcmds import check_all_rows
        from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore
        from tools.dk.core.command import BUILD_ERROR, MISMATCH, OK
        unit = lambda n, done: {"name": n, "metadata": {"complete": done}}
        good = UnitScore("a", [FunctionScore("f", 8, 100.0)], {".data": 100.0})
        half = UnitScore("w", [FunctionScore("g", 8, 100.0), FunctionScore("h", 8, 50.0)])
        units = [unit("a", True), unit("w", False)]
        lines, code = check_all_rows(units, {"a": good, "w": half})
        self.assertEqual(code, OK)
        self.assertEqual(lines[0], "linked units: 1/1 exact")
        self.assertIn("1/2", lines[-1])
        _, code = check_all_rows(units, {"a": half, "w": good})
        self.assertEqual(code, MISMATCH)
        _, code = check_all_rows(units, {"a": None, "w": good})
        self.assertEqual(code, BUILD_ERROR)

    def test_only_trailing_padding(self):
        import base64
        from tools.dk.platforms.gc_mwcc.matching import _only_trailing_padding
        b64 = lambda raw: base64.b64encode(raw).decode()
        common = {"data": b64(b"x" * 554), "size": "554"}
        tail = lambda raw: {"kind": "DIFF_DELETE", "data": b64(raw), "size": str(len(raw))}
        ours = {"data_diff": [common, {"kind": "DIFF_DELETE", "size": "6"}]}
        self.assertTrue(_only_trailing_padding({"data_diff": [common, tail(bytes(6))]}, ours))
        self.assertFalse(_only_trailing_padding({"data_diff": [common, tail(b"\0\0\0\0\0\1")]}, ours))   # real byte
        self.assertFalse(_only_trailing_padding({"data_diff": [common, tail(bytes(32))]}, ours))          # not padding
        self.assertFalse(_only_trailing_padding({"data_diff": [tail(bytes(4)), common]}, ours))          # not at the end

    def test_discarded_follows_stripped_functions(self):
        from tools.dk.platforms.gc_mwcc.matching import _discarded
        referrers = {"@lit_a": {"StrippedFn"}, "@lit_shared": {"StrippedFn", "KeptFn"},
                     "@rtti": {"__vt__4Base"}, "@name": {"@rtti"}}
        objects = {"@lit_a", "@lit_shared", "@rtti", "@name", "@unreferenced"}
        self.assertEqual(_discarded(objects, referrers, {"__vt__4Base"}), {"__vt__4Base", "@rtti", "@name"})
        both = _discarded(objects, referrers, {"__vt__4Base", "StrippedFn"})
        self.assertIn("@lit_a", both)                     # only the stripped function used it
        self.assertNotIn("@lit_shared", both)             # a kept function still uses it
        self.assertNotIn("@unreferenced", both)

    def test_stripped_only_data_is_never_zero_filled(self):
        import base64
        from tools.dk.platforms.gc_mwcc.matching import _only_weak_extras
        b64 = lambda raw: base64.b64encode(raw).decode()
        hole = {"data_diff": [{"data": b64(b"x" * 8), "size": "8"}, {"kind": "DIFF_DELETE", "data": b64(bytes(4)), "size": "4"}]}
        ours = {"data_diff": [{"data": b64(b"x" * 8), "size": "8"}, {"kind": "DIFF_INSERT", "data": b64(b"\x3f\x80\0\0"), "size": "4"}]}
        ranges = [(8, 12, "@37")]
        self.assertFalse(_only_weak_extras(hole, ours, ranges, zero_fillable=set()))   # a missing 0.0f literal stays a diff

    def test_packed_section_matches(self):
        from tools.dk.platforms.gc_mwcc.elf import Elf, ElfSection, ElfSymbol
        from tools.dk.platforms.gc_mwcc.matching import _packed_section_matches
        def obj(data, syms, nobits=False):
            sec = ElfSection(1, ".sdata2", 8 if nobits else 1, 0, len(data), 0, 0, b"" if nobits else data)
            return Elf((ElfSection(0, "", 0, 0, 0, 0, 0, b""), sec),
                       tuple(ElfSymbol(n, v, sz, 1, 0, 1) for n, v, sz in syms))
        f = lambda x: bytes.fromhex(x)
        # ours: 0.0f, 0.5f (stripped-only), 1.0f, pad, 0.5 (double); the link keeps 0.0f, 1.0f, 0.5
        ours = obj(f("00000000 3f000000 3f800000 00000000 3fe0000000000000"),
                   [("@1", 0, 4), ("@2", 4, 4), ("@3", 8, 4), ("@4", 16, 8)])
        target = obj(f("00000000 3f800000 3fe0000000000000"), [("lbl_a", 0, 4), ("lbl_b", 4, 4), ("lbl_c", 8, 8)])
        self.assertTrue(_packed_section_matches(target, ours, ".sdata2", {"@2"}))
        self.assertFalse(_packed_section_matches(target, ours, ".sdata2", set()))          # 0.5f really is extra
        wrong = obj(f("00000000 3f800000 4000000000000000"), [("lbl_a", 0, 4), ("lbl_b", 4, 4), ("lbl_c", 8, 8)])
        self.assertFalse(_packed_section_matches(wrong, ours, ".sdata2", {"@2"}))          # a wrong literal
        # bss: kept objects must start at target labels (a u64 after a u32 aligns to 8)
        ours_bss = obj(bytes(24), [("a", 0, 4), ("t", 4, 4), ("b", 8, 8), ("unused", 16, 8)], nobits=True)
        target_bss = obj(bytes(16), [("a", 0, 4), ("gap", 4, 4), ("b", 8, 4), ("b_hi", 12, 4)], nobits=True)
        self.assertTrue(_packed_section_matches(target_bss, ours_bss, ".sdata2", {"t", "unused"}))
        self.assertFalse(_packed_section_matches(target_bss, ours_bss, ".sdata2", {"t"}))


class RenameLocalTest(unittest.TestCase):
    TEXT = ("SamplingHandler = .text:0x801CAC88; // type:function size:0x40 scope:local\n"
            "fn_801DB9DC = .text:0x801DB9DC; // type:function size:0x40\n"
            "PADRead = .text:0x801CB000; // type:function size:0x40 scope:global\n"
            "fn_801DBA1C = .text:0x801DBA1C; // type:function size:0x40\n")

    def test_local_duplicates_only_with_flag_and_only_of_locals(self):
        from tools.dk.platforms.gc_mwcc.renamecmd import rename_text
        _, errors = rename_text(self.TEXT, [("fn_801DB9DC", "SamplingHandler")])
        self.assertTrue(errors)
        text, errors = rename_text(self.TEXT, [("fn_801DB9DC", "SamplingHandler")], local=True)
        self.assertEqual(errors, [])
        self.assertIn("SamplingHandler = .text:0x801DB9DC; // type:function size:0x40 scope:local\n", text)
        text, errors = rename_text(self.TEXT, [("fn_801DBA1C", "PADRead")], local=True)
        self.assertEqual(errors, [])                              # the existing one becomes local too
        self.assertIn("PADRead = .text:0x801CB000; // type:function size:0x40 scope:local\n", text)
        self.assertIn("PADRead = .text:0x801DBA1C; // type:function size:0x40 scope:local\n", text)

    def test_nobits_order_is_checked(self):
        from tools.dk.platforms.gc_mwcc.elf import Elf, ElfSection, ElfSymbol
        from tools.dk.platforms.gc_mwcc.matching import _nobits_layout_problems
        def obj(syms):
            sec = ElfSection(1, ".sbss", 8, 0, 12, 0, 0, b"")
            return Elf((ElfSection(0, "", 0, 0, 0, 0, 0, b""), sec), tuple(ElfSymbol(n, v, 4, 1, 0, 1) for n, v in syms))
        target = obj([("CPUFifo", 0), ("GPFifo", 4), ("__GXCurrentThread", 8)])
        self.assertEqual(_nobits_layout_problems(target, obj([("CPUFifo", 0), ("GPFifo", 4), ("__GXCurrentThread", 8)]), {}), [])
        wrong = obj([("__GXCurrentThread", 0), ("GPFifo", 4), ("CPUFifo", 8)])
        self.assertEqual(len(_nobits_layout_problems(target, wrong, {})), 2)
        # a discarded object before the others: compare packed offsets
        extra = obj([("unused", 0), ("CPUFifo", 4), ("GPFifo", 8), ("__GXCurrentThread", 12)])
        self.assertEqual(_nobits_layout_problems(target, extra, {".sbss": {"unused"}}), [])


class UnitFlagsTest(unittest.TestCase):
    def test_set_and_clear_flags(self):
        import tempfile
        from pathlib import Path
        from tools.dk.platforms.gc_mwcc.units import read_units_file, render_units_file, set_flags
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "units.toml"
            path.write_text('[[lib]]\nname = "game"\nobjects = [\n    { path = "game/a.cpp", status = "NonMatching", cflags = "game", mw_version = "GC/1.3.2" },\n]\n')
            units = read_units_file(path)
            changes = set_flags(units, "game/a.cpp", "GC/1.3", None, ["-char", "signed"])
            self.assertEqual(len(changes), 2)
            text = render_units_file(units)
            self.assertIn('mw_version = "GC/1.3"', text)
            self.assertIn('extra_cflags = ["-char", "signed"]', text)
            path.write_text(text)
            units = read_units_file(path)
            self.assertEqual(set_flags(units, "game/a.cpp", None, None, []), ["extra_cflags: ['-char', 'signed'] -> (none)"])
            with self.assertRaises(KeyError):
                set_flags(units, "game/nope.cpp", "GC/1.3", None, None)


class ByAddressTest(unittest.TestCase):
    TEXT = ("ResetFunctionInfo = .data:0x80270000; // type:object size:0x8 scope:local\n"
            "ResetFunctionInfo = .data:0x80271000; // type:object size:0x8 scope:local\n")

    def test_rename_and_size_one_of_two_same_named_symbols(self):
        from tools.dk.platforms.gc_mwcc.renamecmd import rename_text
        from tools.dk.platforms.gc_mwcc.symbolcmd import size_text
        text, errors = rename_text(self.TEXT, [("0x80271000", "ResetFunctionInfo2")])
        self.assertEqual(errors, [])
        self.assertIn("ResetFunctionInfo = .data:0x80270000;", text)
        self.assertIn("ResetFunctionInfo2 = .data:0x80271000;", text)
        sized, _ = size_text(self.TEXT, "0x80270000", 0x10)
        self.assertIn("ResetFunctionInfo = .data:0x80270000; // type:object size:0x10", sized)
        self.assertIn("ResetFunctionInfo = .data:0x80271000; // type:object size:0x8", sized)


class RelocatedDataTest(unittest.TestCase):
    """Pointers into a section that lost a weak object mid-way point at shifted offsets of the
    same objects: equal when they point at the same bytes, not when the bytes differ."""

    class Fake:
        def __init__(self, sections, symbols, relocs):
            from tools.dk.platforms.gc_mwcc.elf import ElfSection, ElfSymbol
            self.sections = tuple([ElfSection(0, "", 0, 0, 0, 0, 0, b"")] +
                                  [ElfSection(i + 1, n, 1, 0, len(d), 0, 0, d) for i, (n, d) in enumerate(sections)])
            self.symbols = tuple(ElfSymbol(n, v, sz, 1, rest[0] if rest else 0, idx) for n, v, sz, idx, *rest in symbols)
            self._relocs = relocs

        def section(self, name):
            return next((s for s in self.sections if s.name == name), None)

        def relocations(self, index):
            return self._relocs.get(index, [])

    def test_shifted_pointers_to_equal_objects(self):
        from tools.dk.platforms.gc_mwcc.elf import Reloc
        from tools.dk.platforms.gc_mwcc.matching import _relocated_section_matches
        name = b"RaceManager\0"
        target = self.Fake([(".sdata", bytes(8)), (".data", name)],
                           [("lbl_a", 0, 12, 2), ("__RTTI__x", 0, 8, 1)],
                           {1: [Reloc(0, 1, "lbl_a", 0, ".data", 0)]})
        ours = self.Fake([(".sdata", bytes(8)), (".data", bytes(0x2C) + name)],
                         [("__vt__weak", 0, 0x2C, 2), ("@153", 0x2C, 12, 2), ("__RTTI__x", 0, 8, 1)],
                         {1: [Reloc(0, 1, "@153", 0, ".data", 0x2C)]})
        self.assertTrue(_relocated_section_matches(target, ours, ".sdata", {}))
        other = self.Fake([(".sdata", bytes(8)), (".data", bytes(0x2C) + b"Different\0\0\0")],
                          [("@153", 0x2C, 12, 2), ("__RTTI__x", 0, 8, 1)],
                          {1: [Reloc(0, 1, "@153", 0, ".data", 0x2C)]})
        self.assertFalse(_relocated_section_matches(target, other, ".sdata", {}))

    def test_private_copies_pair_by_bytes_global_objects_by_name(self):
        from tools.dk.platforms.gc_mwcc.elf import Reloc
        from tools.dk.platforms.gc_mwcc.matching import _relocated_section_matches
        rtti = bytes(8)
        target = self.Fake([(".data", bytes(4)), (".sdata", rtti)], [("lbl_8038E04C", 0, 8, 2, 1)],
                           {1: [Reloc(0, 1, "lbl_8038E04C", 0, ".sdata", 0)]})
        local = self.Fake([(".data", bytes(4)), (".sdata", rtti)], [("__RTTI__7GObject", 0, 8, 2, 0)],
                          {1: [Reloc(0, 1, "__RTTI__7GObject", 0, ".sdata", 0)]})
        self.assertTrue(_relocated_section_matches(target, local, ".data", {}))      # the unit's own copy
        exported = self.Fake([(".data", bytes(4)), (".sdata", rtti)], [("__RTTI__7GObject", 0, 8, 2, 1)],
                             {1: [Reloc(0, 1, "__RTTI__7GObject", 0, ".sdata", 0)]})
        self.assertFalse(_relocated_section_matches(target, exported, ".data", {}))  # a global must be the same one


class ForceActiveTest(unittest.TestCase):
    def test_list_edits_and_per_unit_composition(self):
        from tools.dk.platforms.gc_mwcc import dtkconfig
        base = "object: sys/main.dol\nwrite_asm: true\n"
        one = dtkconfig.set_force_active(base, ["b", "a"])
        self.assertEqual(dtkconfig.force_active_names(one), ["a", "b"])
        self.assertEqual(dtkconfig.set_force_active(one, []), base)               # empty list: block gone
        self.assertEqual(dtkconfig.force_active_names("force_active: [x, y]\n"), ["x", "y"])
        head = dtkconfig.set_force_active(base, ["other_unit_old"])
        work = dtkconfig.set_force_active(base, ["mine", "other_unit_new"])
        composed = dtkconfig.compose(head, work, lambda n: n == "mine")
        self.assertEqual(dtkconfig.force_active_names(composed), ["mine", "other_unit_old"])


class SharedTreeWaitTest(unittest.TestCase):
    def test_only_other_units_compile_failures_count(self):
        from tools.dk.platforms.gc_mwcc.freshness import others_broken
        out = ("FAILED: [code=2] build/GH2E69/src/game/fe/SingleChallengeScreen.o\n"
               "FAILED: build/GH2E69/src/game/fe/Mine.o\nFAILED: build/GH2E69/main.elf\n")
        self.assertEqual(others_broken(out, {"build/GH2E69/src/game/fe/Mine.o"}),
                         {"build/GH2E69/src/game/fe/SingleChallengeScreen.o"})
        self.assertEqual(others_broken("FAILED: build/GH2E69/main.elf\n", set()), set())    # a link error is ours


if __name__ == "__main__":
    unittest.main()


class VtableSlotTest(unittest.TestCase):
    """An override declared in the wrong slot leaves the vtable's bytes equal (relocations are
    zero bytes) but points two slots at other functions: check must say so."""

    def fake(self, slots, defined, size=0x14):
        from tools.dk.platforms.gc_mwcc.elf import ElfSection, ElfSymbol, Reloc

        class Fake:
            sections = (ElfSection(0, "", 0, 0, 0, 0, 0, b""), ElfSection(1, ".text", 1, 0, 0, 0, 0, b""),
                        ElfSection(2, ".data", 1, 0, size, 0, 0, bytes(size)))
            symbols = tuple([ElfSymbol("__vt__3Foo", 0, size, 1, 1, 2)] +
                            [ElfSymbol(n, i * 4, 4, 2, 1, 1) for i, n in enumerate(defined)] +
                            [ElfSymbol(n, 0, 0, 0, 1, 0) for _, n in slots if n not in defined])

            def relocations(self, index):
                return [Reloc(off, 1, n, 0) for off, n in slots] if index == 2 else []
        return Fake()

    def test_wrong_slot_and_tolerated_placeholders(self):
        from tools.dk.platforms.gc_mwcc.matching import vtable_problems
        target = self.fake([(0, "__RTTI__3Foo"), (8, "vf2__3FooFv"), (0xC, "vf3__3BarFv"), (0x10, "fn_80001000")],
                           ["vf2__3FooFv"])
        ours = self.fake([(0, "@12"), (8, "vf2__3BarFv"), (0xC, "vf3__3FooFv"), (0x10, "vf4__3BarFv")],
                         ["vf3__3FooFv"])
        self.assertEqual(vtable_problems(target, ours), [
            "__vt__3Foo+0x8: target vf2__3FooFv, ours vf2__3BarFv",
            "__vt__3Foo+0xC: target vf3__3BarFv, ours vf3__3FooFv"])
        # slot 0 is RTTI (data); 0x10 an unnamed function elsewhere that ours cannot be checked against
        ok = self.fake([(0, "@12"), (8, "vf2__3FooFv"), (0xC, "vf3__3BarFv"), (0x10, "vf4__3BarFv")], ["vf2__3FooFv"])
        self.assertEqual(vtable_problems(target, ok), [])


class RelocationEquivalenceTest(unittest.TestCase):
    """Our `.data.0+0x8` (or an `@12` literal) and dtk's `lbl_...` at the same offset of the same
    section, with the same bytes there, link to the same instruction."""

    def elf(self, data, symbols):
        from tools.dk.platforms.gc_mwcc.elf import ElfSection, ElfSymbol

        class Fake:
            sections = (ElfSection(0, "", 0, 0, 0, 0, 0, b""), ElfSection(1, ".text", 1, 0, 8, 0, 0, bytes(8)),
                        ElfSection(2, ".sdata2", 1, 0, len(data), 0, 0, data))
            def section(self, name):
                return next((s for s in self.sections if s.name == name), None)

            def relocations(self, index):
                return []
        f = Fake()
        f.symbols = tuple(ElfSymbol(n, v, sz, 1, 0, 2) for n, v, sz in symbols)
        return f

    def test_same_offset_and_bytes_is_equivalent(self):
        from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
        from tools.dk.platforms.gc_mwcc.matching import relocation_difference
        code = bytes.fromhex("c0220000") + bytes.fromhex("4e800020")         # lfs f1, x@sda21(r2); blr
        floats = bytes.fromhex("3f800000" "40000000")
        target = self.elf(floats, [("lbl_80390000", 0, 4), ("lbl_80390004", 4, 4)])
        theirs = Function("f", ".text", code, [Reloc(0, 109, "lbl_80390004", 0, ".sdata2", 4)])
        same = Function("f", ".text", code, [Reloc(0, 109, "@12", 0, ".sdata2", 4)])
        self.assertIsNone(relocation_difference(target, self.elf(floats, [("@12", 4, 4)]), theirs, same))
        moved = Function("f", ".text", code, [Reloc(0, 109, "@12", 0, ".sdata2", 0)])
        self.assertIn("target's order", relocation_difference(target, self.elf(floats, [("@12", 0, 4)]), theirs, moved))
        other = self.elf(bytes.fromhex("3f800000" "40400000"), [("@12", 4, 4)])
        self.assertIn("different bytes", relocation_difference(target, other, theirs, same))
        table = self.elf(floats, [("@12", 4, 4)])
        table.relocations = lambda index: [Reloc(4, 1, ".text", 0x40, ".text", 0)] if index == 2 else []
        self.assertIn("its pointers differ", relocation_difference(target, table, theirs, same))   # a jump table's labels
        register = Function("f", ".text", bytes.fromhex("c0420000") + code[4:], same.relocs)   # lfs f2
        self.assertEqual(relocation_difference(target, self.elf(floats, [("@12", 4, 4)]), theirs, register), "")

    def test_notes_say_what_to_change(self):
        from tools.dk.platforms.gc_mwcc.elf import Function, Reloc
        from tools.dk.platforms.gc_mwcc.matching import relocation_difference
        call = bytes.fromhex("48000001") + bytes.fromhex("4e800020")         # bl x; blr
        target = self.elf(b"", [])
        member = Function("f", ".text", call, [Reloc(0, 10, "Run__3FooFv", 0)])
        c_name = Function("f", ".text", call, [Reloc(0, 10, "fn_80001000", 0)])
        note = relocation_difference(target, target, member, c_name, {0x80001000: "Run__3FooFv"}.get)
        self.assertIn("still uses the C name fn_80001000", note)
        load = bytes.fromhex("c0220000") + bytes.fromhex("4e800020")
        outside = Function("f", ".text", load, [Reloc(0, 109, "lbl_80390004", 0)])        # undefined in the target
        ours = Function("f", ".text", load, [Reloc(0, 109, "@12", 0, ".sdata2", 0)])
        self.assertIn("outside this unit's split", relocation_difference(target, self.elf(bytes(4), [("@12", 0, 4)]), outside, ours))


class NearLinkTest(unittest.TestCase):
    def test_lists_units_close_to_exact_with_reasons(self):
        from tools.dk.platforms.gc_mwcc.matchcmds import near_rows
        units = [{"name": "main/a", "metadata": {}}, {"name": "main/b", "metadata": {}},
                 {"name": "main/linked", "metadata": {"complete": True}}]
        a = UnitScore("main/a", [FunctionScore("f", 0x10, 100.0), FunctionScore("g", 0x20, 99.5, "+0x4: literal order")])
        b = UnitScore("main/b", [FunctionScore(n, 4, 50.0) for n in "wxyz"])          # four open: not near
        rows = near_rows(units, {"main/a": a, "main/b": b, "main/linked": a})
        self.assertEqual(rows[0], "near link (1 WIP unit(s) at most 3 function(s) from exact):")
        self.assertIn("main/a: 1 function(s) open, 0x20 bytes", rows[1])
        self.assertIn("g 99.50%: +0x4: literal order", rows[2])
