"""`regs --why-temp` (whytemp.py) and oracle contract v3 (`--temps`: each temp's creating phase).

Synthetic: hand-made records with `origin` fields and solver rules. Golden (skipped without the project's
compiler and a runner whose probe lists "temps" in DK_ORACLE): a toy function with a common subexpression
compiled under the real runner, with and without temp origins."""

import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import oracle, whytemp

REPO = Path(__file__).resolve().parents[1]

SRC = textwrap.dedent("""\
    void f(S* s, int k) {
        int x = s->a * k;
        int y = k > 0 ? s->b : 0;
        g(s->a * k);
        h(y && x);
    }
    """)


def v(index, kind, name="", vline=None, **origin):
    out = {"index": index, "cls": "gpr", "kind": kind, "name": name, "line": vline}
    if origin:
        out["origin"] = origin
    return out


def rule(a, b, cell="temp-temp", flip=True):
    return {"before": {"index": a["index"], "kind": a["kind"], "name": a["name"]},
            "after": {"index": b["index"], "kind": b["kind"], "name": b["name"]}, "flip": flip, "cell": cell}


class LeverTest(unittest.TestCase):
    def test_cse_after_linear_is_phase_blocked(self):
        a = v(55, "fe-temp", "@9", 2, phase="cse0", line=4, first_line=2)
        b = v(61, "fe-temp", "@3", 3, phase="linear", line=3)
        verdict, lv = whytemp.levers(a, b)
        self.assertIn("different phases", verdict)
        self.assertIn("linearize temp always precedes", verdict)
        self.assertTrue(any("@9 earlier" in x and "linearize" in x for x in lv))
        self.assertTrue(any("@3 later" in x and "if/else" in x for x in lv))

    def test_same_phase_cse_names_second_occurrences(self):
        a = v(40, "fe-temp", "@7", 2, phase="cse0", line=9, first_line=2)
        b = v(41, "fe-temp", "@8", 3, phase="cse0", line=5, first_line=3)
        verdict, lv = whytemp.levers(a, b)
        self.assertIn("same phase (cse0)", verdict)
        self.assertIn("2nd occurrence (line 9)", lv[0])
        self.assertTrue(any("after line 9" in x for x in lv))
        self.assertTrue(any("first use come second" in x for x in lv))

    def test_same_statement_linear_names_operand_cost(self):
        a = v(40, "fe-temp", "@7", 3, phase="linear", line=3)
        b = v(41, "fe-temp", "@8", 3, phase="linear", line=3)
        _verdict, lv = whytemp.levers(a, b)
        self.assertTrue(any("costlier operand" in x for x in lv))

    def test_loops_by_test_position_and_inline_roles(self):
        _v, lv = whytemp.levers(v(1, "fe-temp", "@1", phase="loop", line=8), v(2, "fe-temp", "@2", phase="loop", line=4))
        self.assertIn("inner loop before its outer loop", lv[0])
        a = v(1, "fe-temp", "@1", phase="front", role="arg", inline="Helper", line=3)
        b = v(2, "fe-temp", "@2", phase="front", role="local", inline="Helper", line=3)
        _v, lv = whytemp.levers(a, b)
        self.assertTrue(any("result, then the locals, then the arguments" in x for x in lv))

    def test_earlier_phase_already_first_points_at_liveness(self):
        verdict, lv = whytemp.levers(v(1, "fe-temp", "@1", phase="linear"), v(2, "fe-temp", "@2", phase="cse0"))
        self.assertIn("simplify order", verdict)
        self.assertIn("liveness", lv[0])

    def test_cross_tier_and_unknown_phase(self):
        verdict, lv = whytemp.levers(v(1, "local", "parent", 2), v(2, "fe-temp", "@4", 3, phase="cse0", line=4))
        self.assertIn("cross-tier", verdict)
        self.assertTrue(any("remove the temp @4" in x for x in lv))
        self.assertTrue(any("re-read the member" in x for x in lv))
        verdict, _lv = whytemp.levers(v(1, "fe-temp", "@1"), v(2, "fe-temp", "@2"))
        self.assertEqual(verdict, "phase unknown")
        self.assertIn("declaration order", whytemp.levers(v(1, "local", "a"), v(2, "local", "b"))[0])

    def test_explain_reads_origins_from_the_record_and_source_lines(self):
        a = v(55, "fe-temp", "@9", 2, phase="cse0", line=4, first_line=2)
        b = v(61, "fe-temp", "@3", 3, phase="linear", line=3)
        an = {"classes": {"gpr": {"rules": [rule(a, b), rule(b, a, flip=False)]}}}
        rec = {"vregs": [a, b]}
        out = whytemp.explain(an, rec, SRC, (0, 5))
        self.assertEqual(len(out), 1)                       # only the broken rule
        self.assertIn("2nd occurrence (line 4: g(s->a * k);)", out[0]["before"]["text"])
        self.assertIn("the value of a ?: expression", out[0]["after"]["text"])
        self.assertEqual(len(whytemp.explain(an, rec, SRC, (0, 5), flipped_only=False)), 2)
        text = "\n".join(whytemp.render(out))
        self.assertIn("rule 1 [gpr temp-temp] (flipped): colour @9 before @3", text)
        self.assertIn("outside the body", whytemp.describe(v(1, "fe-temp", "@1", 40, phase="front", line=40), SRC.splitlines(), (0, 5)))

    def test_member_body_is_found_by_class(self):
        from tools.dk.platforms.gc_mwcc.regs import member_body_lines
        text = "void A::vf20() {\n    a();\n}\nvoid B::vf20() {\n    b();\n}\n"
        calls = "void h() {\n    if (f(1)) {\n    }\n}\nint f(int x) {\n    return f(x - 1);\n}\n"
        self.assertEqual(member_body_lines(calls, "f__Fi"), (4, 6))     # the definition, not the call
        self.assertEqual(member_body_lines(text, "vf20__1BFv"), (3, 5))
        self.assertEqual(member_body_lines(text, "vf20__1AFv"), (0, 2))

    def test_wrap_passes_temps(self):
        argv = oracle.wrap(["run"], Path("o.json"), "f__Fv", extra=True, temps=True)(["cc", "-c", "a.cpp"])
        self.assertEqual(argv[:6], ["run", "o.json", "--funcs", "f", "--extra", "--temps"])
        self.assertNotIn("--temps", oracle.wrap(["run"], Path("o.json"), None, extra=True)(["cc"]))


def _temps_runner():
    cmd = oracle.runner(None)
    if cmd is None:
        return None
    work = Path(tempfile.mkdtemp())
    try:
        return cmd if "temps" in oracle.modes(cmd, work) else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


FLAGS = ["-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware", "-Cpp_exceptions", "off",
         "-RTTI", "off", "-enum", "int", "-lang=c++"]
TOY = ("int g(int);\nstruct S { int a; int b; int* p; };\nint toy2(S* s, int k) {\n"
       "    int y = g((k > 0 ? s->a : s->b) + k);\n    int z = g(s->p[k] + 1);\n    g(s->p[k] + 2);\n"
       "    return g(y + (k && s->b)) + z;\n}\n")


@unittest.skipUnless((REPO / "build/compilers/GC/1.3.2/mwcceppc.exe").exists() and _temps_runner(),
                     "needs the project's compiler and an oracle runner with contract v3 (--temps) in DK_ORACLE")
class GoldenTest(unittest.TestCase):
    def test_cse_temp_gets_its_phase_and_the_object_is_unchanged(self):
        work = Path(tempfile.mkdtemp())
        try:
            (work / "t.cpp").write_text(TOY)
            objs, recs = [], []
            for temps in (False, True):
                argv = [str(REPO / "build/tools/wibo"), str(REPO / "build/compilers/GC/1.3.2/mwcceppc.exe"), *FLAGS,
                        "-c", "t.cpp", "-o", f"t{int(temps)}.o"]
                subprocess.run(oracle.wrap(_temps_runner(), work / f"v{int(temps)}.json", "toy2__FP1Si", True,
                                           temps=temps)(argv), cwd=work, capture_output=True)
                objs.append((work / f"t{int(temps)}.o").read_bytes())
                recs.append(oracle.find_record(oracle.load(work / f"v{int(temps)}.json"), "toy2__FP1Si", None))
            self.assertEqual(objs[0], objs[1])                       # the hooks change nothing
            self.assertFalse((work / "t.log").exists())              # the compiler's log is cleaned up
            temps = [x for x in recs[1]["vregs"] if x["kind"] == "fe-temp"]
            self.assertTrue(temps and all("origin" in x for x in temps))
            cse = [x["origin"] for x in temps if x["origin"]["phase"] == "cse0"]
            self.assertEqual((cse[0]["line"], cse[0]["first_line"]), (6, 5))   # made at the 2nd occurrence
            self.assertFalse(any("origin" in x for x in recs[0]["vregs"]))
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
