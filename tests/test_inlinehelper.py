import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import closecmd, rewrites
from tools.dk.platforms.gc_mwcc.closecmd import Closer, Eval, Options
from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.inlinehelper import parameters, span_of
from tools.dk.platforms.gc_mwcc.linkproof import Proof
from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore
from tools.dk.platforms.gc_mwcc.ranksolve import below_parameters, inline_notes
from tools.dk.platforms.gc_mwcc.regs import DwarfFunction, RegisterMap, Var, debug_refs, entry_moves, find_function, \
    parse_dwarf1

REPO = Path(__file__).resolve().parents[1]

MEMBER = """struct O { int m4; int m8; };
extern void Drop(O*, int);
struct M {
    O* mE18[5][25];
    int m100C[5];
    void f(O* object);
};
void M::f(O* object) {
    for (int type = 0; type < 5; type++) {
        O** subs = mE18[type];
        for (int i = 0; i < 25; i++) {
            if (subs[i] == object) {
                Drop(object, type);
                if (m100C[type] > 0) {
                    m100C[type]--;
                }
                break;
            }
        }
    }
}
"""


def site(src: str, fn: str, cplusplus: bool = True, members=None, prefer=()) -> rewrites.Site:
    span = next(f for f in cbody.functions(src) if f.name == fn)
    body = cbody.parse_body(src, span.open, cbody.type_names(src))
    return rewrites.Site(src, body, None, cplusplus, members=members, prefer=frozenset(prefer))


def candidates(s: rewrites.Site) -> list[tuple[str, str]]:
    return [(label, new) for _, label, new in rewrites.candidates(s, [rewrites.BY_NAME["inline-helper"]], groups=False)]


class ExtractTest(unittest.TestCase):
    def test_the_inner_loop_becomes_a_helper_with_self_for_members(self):
        got = candidates(site(MEMBER, "M::f", members=frozenset({"mE18", "m100C"})))
        labels = [lab for lab, _ in got]
        self.assertIn("loop at line 11 as inline helper UnkLoop11(this, subs, object, type)", labels)
        self.assertIn("loop at line 11 as inline helper UnkLoop11(this, object, type, subs)", labels)
        new = dict(got)["loop at line 11 as inline helper UnkLoop11(this, subs, object, type)"]
        self.assertIn("static inline void UnkLoop11(M* self, O** subs, O* object, int type) {\n"
                      "    for (int i = 0; i < 25; i++) {", new)
        self.assertIn("            if (self->m100C[type] > 0) {\n                self->m100C[type]--;", new)
        self.assertIn("        O** subs = mE18[type];\n        UnkLoop11(this, subs, object, type);\n    }", new)
        self.assertLess(new.index("static inline void UnkLoop11"), new.index("void M::f"))

    def test_a_local_only_the_loop_uses_moves_with_it(self):
        got = dict(candidates(site(MEMBER, "M::f", members=frozenset({"mE18", "m100C"}))))
        new = next(v for k, v in got.items() if k.startswith("loop and its locals at line 10"))
        self.assertIn("    O** subs = self->mE18[type];\n    for (int i = 0;", new)

    def test_written_variables_go_by_reference_in_cpp_and_by_pointer_in_c(self):
        src = "int f(int* p, int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) {\n        s += p[i];\n    }\n    return s;\n}\n"
        cpp = candidates(site(src, "f"))
        heads = [new[new.index("static"):new.index("{")] for lab, new in cpp if "the rest" not in lab]
        self.assertEqual(heads, ["static inline void UnkLoop3(int n, int& s, int* p) ",      # first use
                                 "static inline void UnkLoop3(int* p, int n, int& s) "])     # declaration order
        c = candidates(site(src, "f", cplusplus=False))
        new = next(new for _, new in c if "UnkLoop3(int n, int* s, int* p)" in new)
        self.assertIn("(*s) += p[i];", new)
        self.assertIn("UnkLoop3(n, &s, p);", new)

    def test_a_search_loop_becomes_a_helper_that_returns(self):
        src = ("int* f(int** p, int n, int k) {\n    int* found = 0;\n    for (int i = 0; i < n; i++) {\n"
               "        if (p[i][0] == k) {\n            found = p[i];\n            break;\n        }\n    }\n"
               "    return found;\n}\n")
        got = dict(candidates(site(src, "f")))
        new = next(v for k, v in got.items() if k.startswith("search loop"))
        self.assertIn("static inline int* UnkLoop3(int n, int** p, int k) {", new)
        self.assertIn("            return p[i];\n        }\n    }\n    return 0;\n}", new)
        self.assertIn("    int* found = UnkLoop3(n, p, k);\n    return found;", new)

    def test_a_loop_that_returns_from_the_function_stays_unless_it_is_the_tail(self):
        src = ("int f(int* p, int n) {\n    for (int i = 0; i < n; i++) {\n        if (p[i]) return i;\n    }\n"
               "    n = 0;\n    return -1;\n}\n")
        self.assertEqual(candidates(site(src, "f")), [])
        src = "int f(int* p, int n) {\n    for (int i = 0; i < n; i++) {\n        if (p[i]) return i;\n    }\n    return -1;\n}\n"
        got = candidates(site(src, "f"))
        self.assertEqual([lab for lab, _ in got], ["loop at line 2 and the rest as inline helper UnkLoop2(n, p)",
                                                   "loop at line 2 and the rest as inline helper UnkLoop2(p, n)"])
        self.assertIn("static inline int UnkLoop2(int n, int* p) {\n    for (int i = 0; i < n; i++) {\n"
                      "        if (p[i]) return i;\n    }\n    return -1;\n}\n", got[0][1])
        self.assertIn("int f(int* p, int n) {\n    return UnkLoop2(n, p);\n}", got[0][1])

    def test_a_void_functions_last_loop_keeps_its_returns(self):
        src = "void f(int* p, int n) {\n    for (int i = 0; i < n; i++) {\n        if (p[i]) return;\n        p[i] = 1;\n    }\n}\n"
        got = candidates(site(src, "f"))
        self.assertTrue(got)
        self.assertIn("static inline void UnkLoop2(int n, int* p) {", got[0][1])
        self.assertIn("    UnkLoop2(n, p);\n}", got[0][1])

    def test_a_break_of_an_enclosing_loop_stays(self):
        src = ("void f(int* p, int n) {\n    for (int j = 0; j < n; j++) {\n        if (p[j]) {\n"
               "            for (int i = 0; i < j; i++) p[i] = 0;\n            break;\n        }\n    }\n}\n")
        labels = [lab for lab, _ in candidates(site(src, "f"))]
        self.assertFalse(any(lab.startswith("guarded loop") for lab in labels), labels)
        self.assertTrue(any(lab.startswith("loop at line 4") for lab in labels))

    def test_a_flagged_locals_statements_become_a_helper(self):
        src = ("void f(int* p, int n) {\n    int t = p[0] * n;\n    p[1] = t;\n    p[2] = t + 1;\n"
               "    p[3] = n;\n}\n")
        self.assertEqual(candidates(site(src, "f")), [])                   # no loop, nothing flagged
        got = candidates(site(src, "f", prefer={"t"}))
        self.assertEqual(got[0][0], "block at line 2 as inline helper UnkBlock2(p, n)")
        self.assertIn("static inline void UnkBlock2(int* p, int n) {\n    int t = p[0] * n;\n    p[1] = t;\n"
                      "    p[2] = t + 1;\n}\n", got[0][1])
        self.assertIn("    UnkBlock2(p, n);\n    p[3] = n;\n}", got[0][1])

    def test_preferred_locals_come_first(self):
        src = ("void f(int* p, int n) {\n    for (int a = 0; a < n; a++) p[a] = 0;\n"
               "    for (int b = 0; b < n; b++) p[b] += 1;\n}\n")
        self.assertTrue(candidates(site(src, "f"))[0][0].startswith("loop at line 2"))
        self.assertTrue(candidates(site(src, "f", prefer={"b"}))[0][0].startswith("loop at line 3"))

    def test_an_unexpected_shape_gives_no_candidates_not_an_error(self):
        from unittest import mock
        from tools.dk.platforms.gc_mwcc import inlinehelper
        with mock.patch.object(inlinehelper, "blocks", side_effect=IndexError("surprise")):
            self.assertEqual(candidates(site(MEMBER, "M::f")), [])

    def test_parameters_of_a_head(self):
        src = 'extern "C" bool A::f(const char* name, int arr[4], void (*cb)(int), S& s) const {\n}\n'
        got = parameters(src, span_of(site(src, "A::f")))
        self.assertEqual([(v.name, v.decl) for v in got],
                         [("name", "const char* name"), ("arr", "int arr[4]"), ("cb", "void (*cb)(int)"), ("s", "S& s")])
        self.assertEqual(got[0].as_param("ref"), "const char*& name")


def dwarf(*vars_):
    return DwarfFunction("f", "f__Fv", [Var(k, n, r) for k, n, r in vars_])


class DetectTest(unittest.TestCase):
    def test_a_local_the_target_ranks_below_every_parameter(self):
        d = dwarf(("param", "object", "r26"), ("local", "type", "r29"), ("local", "i", "r27"))
        pairs = {"r26": Counter({"r27": 5}), "r29": Counter({"r29": 3}), "r27": Counter({"r26": 6})}
        got = below_parameters(d, pairs)
        self.assertEqual([(b.name, b.ours, b.target, b.param) for b in got], [("i", "r27", "r26", "object")])
        self.assertTrue(inline_notes(d, pairs)[0].startswith("inline-helper rule: i (r27) ranks below every parameter"))

    def test_temporaries_and_the_entry_copies(self):
        d = dwarf(("param", "file", "r25"), ("param", "key", "r29"))
        pairs = RegisterMap()
        pairs.update({"r25": Counter({"r27": 2, "r25": 9}), "r29": Counter({"r29": 4}), "r28": Counter({"r26": 7})})
        self.assertEqual(below_parameters(d, pairs), [])            # file's uses say r25: no parameter above r26
        pairs.entry = {"r25": "r27", "r29": "r29"}                  # its entry copy says r27
        got = below_parameters(d, pairs)
        self.assertEqual([(b.name, b.ours, b.target) for b in got], [(None, "r28", "r26")])
        self.assertIn("the temporaries in r28 (target r26)", inline_notes(d, pairs)[0])

    def test_nothing_below_when_the_target_keeps_parameters_last(self):
        d = dwarf(("param", "p", "r29"), ("local", "a", "r31"))
        self.assertEqual(below_parameters(d, {"r29": Counter({"r29": 3}), "r31": Counter({"r30": 3})}), [])

    def test_entry_moves(self):
        def ins(m, *regs):
            return {"instruction": {"parts": [{"opcode": {"mnemonic": m}}] + [{"arg": {"opaque": r}} for r in regs]}}
        target = [ins("mr", "r27", "r3"), ins("mr", "r28", "r4"), ins("bl", "f"), ins("mr", "r30", "r5")]
        ours = [ins("mr", "r25", "r3"), ins("mr", "r26", "r4"), ins("bl", "f"), ins("mr", "r29", "r5")]
        self.assertEqual(entry_moves(target, ours), {"r25": "r27", "r26": "r28"})


class FakeCfg:
    def __init__(self, root):
        self.root, self.quality = root, {"forbidden": []}


class SearchTest(unittest.TestCase):
    def test_the_closer_keeps_an_extraction_that_is_exact(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "src").mkdir()
        src = "int f(int* p, int n) {\n    int s = 0;\n    for (int i = 0; i < n; i++) {\n        s += p[i];\n    }\n    return s;\n}\n"
        (root / "src" / "u.cpp").write_text(src)
        unit = {"name": "u", "metadata": {"source_path": "src/u.cpp"}, "target_path": "t.o"}
        want = lambda t: 100.0 if "static inline void UnkLoop3(int* p, int n, int& s)" in t else 90.0

        class Fake(Closer):
            def evaluate(self, text, slot):
                return Eval(text, want(text), 0 if want(text) == 100.0 else 3,
                            rewrites.Symptoms(Counter({"regs": 3}), Counter({4: 3})), 1)

            def unit_score(self, text, slot):
                p = want(text)
                return UnitScore("u", [FunctionScore("f__FPii", 8, p)]), Proof(p == 100.0, "fake"), ""

        r = Fake(FakeCfg(root), unit, "f__FPii", Options(jobs=2), log=lambda s: None).run()
        self.assertTrue(r.exact, r.reasons)
        self.assertEqual(r.labels, ["loop at line 3 as inline helper UnkLoop3(p, n, s)"])
        self.assertIn("    UnkLoop3(p, n, s);\n    return s;", r.text)

    def test_skip_leaves_a_rule_out(self):
        self.assertIn("inline-helper", rewrites.BY_NAME)
        self.assertEqual(closecmd.Options(skip=("inline-helper",)).skip, ("inline-helper",))


class CompilerTest(unittest.TestCase):
    """The candidates are valid code for the real compiler (skipped without it)."""

    def compiles(self, code: str) -> None:
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            self.skipTest("compiler not installed (./dk setup)")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "t.cpp"
            src.write_text(code)
            cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-inline", "auto", "-fp", "hardware",
                   "-enum", "int", "-Cpp_exceptions", "off", "-lang=c++", "-c", str(src), "-o", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr + code)

    def test_every_candidate_of_a_member_function_compiles(self):
        got = candidates(site(MEMBER, "M::f", members=frozenset({"mE18", "m100C"})))
        self.assertGreaterEqual(len(got), 4)
        for _, new in got:
            self.compiles(new)


if __name__ == "__main__":
    unittest.main()
