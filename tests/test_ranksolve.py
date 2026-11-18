import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc.ranksolve import fit, longest_run, reorder_edits, report, solve, solver_candidates
from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.regs import DwarfFunction, Var, debug_refs, find_function, parse_dwarf1

REPO = Path(__file__).resolve().parents[1]
from tools.dk.platforms.gc_mwcc.rewrites import Site


def dwarf(*vars_):
    return DwarfFunction("f", "f__Fv", [Var(k, n, r) for k, n, r in vars_])


def pairs(**m):
    return {ours: Counter({t: 4}) for ours, t in m.items()}


def site(body: str) -> Site:
    src = "void f(int p) {\n" + body + "}\n"
    return Site(src, cbody.parse_body(src, src.index("{")))


class FitTest(unittest.TestCase):
    def test_longest_run(self):
        self.assertEqual(longest_run([0, 2, 1, 3]), {0, 1, 3})
        self.assertEqual(longest_run([]), set())

    def test_declaration_order_from_the_target_registers(self):
        d = dwarf(("param", "p", "r29"), ("local", "a", "r31"), ("local", "b", "r30"))
        f = fit(d, pairs(r31="r30", r30="r31", r29="r29"), "r", True)
        self.assertEqual(f.current, ["a", "b"])
        self.assertEqual(f.order, ["b", "a"])
        self.assertIn("declare in this order (first gets the highest register): b (r31), a (r30)", f.notes)

    def test_merges_ours_versus_target(self):
        # c ranks above a and b in ours (merged with a temporary); in the target it follows them
        d = dwarf(("local", "a", "r30"), ("local", "b", "r29"), ("local", "c", "r31"))
        f = fit(d, pairs(r30="r31", r29="r30", r31="r29"), "r", True)
        self.assertEqual(f.merged_ours, ["c"])
        self.assertEqual(f.merged_target, [])
        self.assertTrue(any(n.startswith("c ranks as a temporary in ours") for n in f.notes))
        self.assertEqual(f.order, ["a", "b", "c"])

    def test_volatile_locals_take_ascending_registers(self):
        d = dwarf(("local", "s", "r10"), ("local", "k", "r9"))
        f = fit(d, pairs(r10="r9", r9="r10"), "r", False)
        self.assertEqual(f.order, ["s", "k"])           # the target follows declaration order (s lower)
        self.assertEqual(f.merged_ours, ["k"])          # ours ranks k first: merged with a temporary
        d = dwarf(("local", "k", "r9"), ("local", "s", "r10"))
        f = fit(d, pairs(r9="r10", r10="r9"), "r", False)
        self.assertEqual(f.order, ["s", "k"])
        self.assertIn("first gets the lowest register", f.notes[0])

    def test_unexplained_temporaries_are_reported(self):
        d = dwarf(("local", "a", "r30"))
        f = fit(d, pairs(r30="r31", r31="r29"), "r", True)
        self.assertTrue(any(n.startswith("missing rule: the temporary in r31") for n in f.notes))
        self.assertTrue(report([f])[0].startswith("callee-saved GPRs"))

    def test_banks_are_separate(self):
        d = dwarf(("local", "x", "f31"), ("local", "a", "r31"), ("local", "i", "r9"))
        banks = [(f.cls, f.callee_saved) for f in solve(d, pairs(f31="f31", r31="r31", r9="r9"))]
        self.assertEqual(banks, [("r", True), ("f", True), ("r", False)])


class CandidateTest(unittest.TestCase):
    def test_minimal_moves_and_hoisting(self):
        s = site("    int a = p;\n    g(a);\n    int b;\n    b = p + 1;\n    g(b);\n")
        new = cbody.apply_edits(s.src, reorder_edits(s, ["b", "a"]))
        self.assertIn("    int b;\n    int a = p;\n    g(a);\n    b = p + 1;", new)
        s = site("    g(p);\n    int a = p;\n    {\n        int b = 2;\n        g(a, b);\n    }\n")
        new = cbody.apply_edits(s.src, reorder_edits(s, ["b", "a"], hoist=True))
        self.assertIn("    int b;\n    int a;\n    g(p);\n    a = p;\n    {\n        b = 2;", new)

    def test_a_declaration_never_moves_into_a_block(self):
        s = site("    {\n        int a = p;\n        g(a);\n    }\n    int b = 1;\n    g(b);\n")
        self.assertIsNone(reorder_edits(s, ["a", "b"]))           # a is already first: nothing to move
        s = site("    int a = p;\n    {\n        int b = 1;\n        g(a, b);\n    }\n")
        self.assertIsNotNone(reorder_edits(s, ["b", "a"]))      # b moves out to the enclosing block

    def test_solver_candidates_put_the_locals_in_target_order(self):
        s = site("    int a = p;\n    int b = p * 2;\n    g(a, b);\n")
        d = dwarf(("local", "a", "r31"), ("local", "b", "r30"))
        got = list(solver_candidates(s, d, pairs(r31="r30", r30="r31")))
        self.assertTrue(got)
        label, new = got[0]
        self.assertIn("in target order: b, a", label)
        self.assertIn("    int b;\n    int a = p;\n    b = p * 2;", new)



class VolatileRuleTest(unittest.TestCase):
    """What the solver assumes for volatile registers, against the real compiler (skipped without it)."""

    def regs(self, code: str, symbol: str) -> dict[str, str]:
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            self.skipTest("compiler not installed (./dk setup)")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "t.cpp"
            src.write_text(code)
            cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware",
                   "-lang=c++", "-sym", "on", "-c", str(src), "-o", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            elf = Elf.parse((Path(tmp) / "t.o").read_bytes())
        f = find_function(parse_dwarf1(elf.section(".debug").data, debug_refs(elf)), symbol)
        return {v.name: v.reg for v in f.vars if v.kind == "local"}

    def test_named_volatile_locals_take_ascending_registers_in_declaration_order(self):
        body = "for (int i = 0; i < n; i++) { s += p[i]; k ^= p[i]; m *= p[i]; } return s * k + m; }"
        a = self.regs("int f(int* p, int n) { int s = 0; int k = 0; int m = 1; " + body, "f__FPii")
        b = self.regs("int f(int* p, int n) { int m = 1; int s = 0; int k = 0; " + body, "f__FPii")
        self.assertEqual([a["s"], a["k"], a["m"]], ["r10", "r11", "r12"])
        self.assertEqual([b["m"], b["s"], b["k"]], ["r10", "r11", "r12"])

    def test_the_same_for_floats(self):
        body = "for (int i = 0; i < n; i++) { s += p[i]; k *= p[i]; } return s * k; }"
        a = self.regs("float f(float* p, int n) { float s = 0.0f; float k = 1.0f; " + body, "f__FPfi")
        b = self.regs("float f(float* p, int n) { float k = 1.0f; float s = 0.0f; " + body, "f__FPfi")
        self.assertEqual((a["s"], a["k"]), ("f6", "f7"))
        self.assertEqual((b["k"], b["s"]), ("f6", "f7"))


if __name__ == "__main__":
    unittest.main()
