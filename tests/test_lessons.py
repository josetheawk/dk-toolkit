import unittest
from pathlib import Path
from string import Template
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import lessons
from tools.dk.platforms.gc_mwcc.lessons import (LESSONS, before, build_pair, definition_span, grid, insns, merge_compilers,
                                                only_registers, reordered, roles_swapped)

REPO = Path(__file__).resolve().parents[1]

RIGHT = "    0  lfs f1, 0x0(r3)\n    4  lfs f0, lbl_80400000@sda21(r0)\n    8  fmuls f0, f1, f0\n    C  blr"
WRONG = "    0  lfs f1, lbl_80400000@sda21(r0)\n    4  lfs f0, 0x0(r3)\n    8  fmuls f0, f1, f0\n    C  blr"


class CheckTest(unittest.TestCase):
    def test_roles_see_operands_swapped_through_the_load_order(self):
        r, w = insns(RIGHT), insns(WRONG)
        self.assertEqual(r[1], ("lfs", "f0, L@sda21(r0)"))
        self.assertTrue(roles_swapped(r, w, ("fmuls",)))       # the same text, the operands' roles swapped
        self.assertFalse(roles_swapped(r, r, ("fmuls",)))
        self.assertTrue(reordered(r, w, ("lfs",)))

    def test_registers_only_and_call_order(self):
        a = insns("    0  mr r31, r3\n    4  bl Use__Fi\n    8  mr r3, r31")
        b = insns("    0  mr r30, r3\n    4  bl Use__Fi\n    8  mr r3, r30")
        self.assertTrue(only_registers(a, b))
        self.assertFalse(only_registers(a, a))
        c = insns("    0  bl Other__Fi\n    4  bl On__Fi")
        self.assertTrue(before(c, "Other", "On"))
        self.assertFalse(before(c, "On", "Other"))


class CatalogTest(unittest.TestCase):
    def test_every_template_expands_and_defines_its_function(self):
        ids = [les.id for les in LESSONS]
        self.assertEqual(len(ids), len(set(ids)))
        for les in LESSONS:
            self.assertTrue(les.params, les.id)
            for shape in les.shapes:
                for p in les.params:
                    decls = Template(shape.decls).substitute(p)
                    right, wrong = Template(shape.right).substitute(p), Template(shape.wrong).substitute(p)
                    self.assertNotEqual(right, wrong, les.id)
                    for code in (right, wrong):
                        self.assertNotIn("$", code, les.id)
                        self.assertIsNotNone(definition_span(decls + "\n" + code + "\n", p["F"]), (les.id, code))
                        for device in ("#pragma", "volatile", "asm", "(void)0"):
                            self.assertNotIn(device, code, les.id)

    def test_grid_is_deterministic_and_varies_names(self):
        a = grid(5, "salt", S=("A", "B", "C"), T=("int", "short"))
        self.assertEqual(a, grid(5, "salt", S=("A", "B", "C"), T=("int", "short")))
        self.assertGreater(len({p["F"] for p in a}), 3)
        self.assertGreater(len({(p["S"], p["T"]) for p in a}), 2)

    def test_merge_compilers(self):
        rec = {"rule": "x", "function": "f__Fv", "wrong": "a", "exact": "b", "asm": "A", "compiler": "GC/1.3",
               "provenance": {}}
        out = merge_compilers([rec, {**rec, "compiler": "GC/1.3.2"}, {**rec, "function": "g__Fv"},
                               {**rec, "function": "g__Fv", "compiler": "GC/1.3.2", "asm": "B"}])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["provenance"]["compilers"], ["GC/1.3", "GC/1.3.2"])
        self.assertEqual(out[1]["provenance"]["differs_on"], ["GC/1.3.2"])


class CompilerTest(unittest.TestCase):
    """Both shapes through the real compiler (skipped without it)."""

    def setUp(self):
        for p in (REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe", REPO / "build" / "tools" / "wibo",
                  REPO / "build" / "tools" / "dtk", REPO / "build" / "tools" / "objdiff-cli"):
            if not p.exists():
                self.skipTest("compiler or tools not installed (./dk setup)")
        self.cfg = SimpleNamespace(root=REPO)

    def test_a_rule_that_holds_gives_repairs_both_ways(self):
        les = lessons.BY_ID["chained-assign"]
        pr = build_pair(self.cfg, les, 0, 0, les.params[0], "GC/1.3.2", lessons.FLAGS)
        self.assertEqual(pr.status, "kept", pr.note)
        self.assertEqual(len(pr.records), 2)
        rec = pr.records[0]
        self.assertEqual((rec["source"], rec["rule"], rec["kind"]), ("lessons", "chained-assign", "repair"))
        self.assertLess(rec["wrong_percent"], 100.0)
        self.assertIn("= p->", rec["exact"])                     # the idiom is the answer of the first record
        self.assertTrue(rec["rows"] and rec["asm"].startswith(".fn"))
        self.assertIn("(the function being matched)", rec["decls"])
        self.assertEqual(rec["provenance"]["direction"], "idiom")
        self.assertEqual(pr.records[1]["exact"], pr.records[0]["wrong"])

    def test_shapes_that_compile_alike_are_not_kept(self):
        les = lessons.Lesson("toy", "test", "s", "i", lambda r, w: True,
                             [lessons.Shape("", "int $F(int a) {\n    return a + 1;\n}",
                                            "int $F(int a) {\n    int b = a + 1;\n    return b;\n}")], [{"F": "Toy"}])
        pr = build_pair(self.cfg, les, 0, 0, les.params[0], "GC/1.3.2", lessons.FLAGS)
        self.assertEqual(pr.status, "same asm")
        self.assertEqual(pr.records, [])


if __name__ == "__main__":
    unittest.main()
