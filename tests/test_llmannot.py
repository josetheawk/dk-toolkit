"""Round 2: prompt v8's annotated asm (llmannot), the DIAG line (llmdiag) and their export and llm-try hooks.
Synthetic fixtures only; the DWARF cases compile a made-up unit and skip without the compiler."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import llmannot, llmdata, llmdiag, llmtry
from tools.dk.platforms.gc_mwcc.llmdecls import Index

REPO = Path(__file__).resolve().parents[1]

UNIT = """struct Vec { float x, y; };
class Base { public: int a; virtual ~Base(); virtual int Speed(); virtual void Go(float t); };
class Car : public Base {
public:
    unsigned char flag;
    Vec pos;
    Car* mNext;
    float speed;
    void f(Car* other);
    static Car* Make();
};
union U { int i; float f; };
extern Car* gCar;
extern U gU;
int Helper(Car* c, float x);
void Car::f(Car* other) { flag = other->flag; pos.y = speed; mNext = gCar; gU.i = Helper(this, pos.x); }
"""
HEADER = "extern Car* gCar;\nint Helper(Car* c, float x);\nCar* GetCar(int i);\n"


def obj(name: str, *lines: str):
    return SimpleNamespace(kind="obj", text="\n".join([f".obj {name}, global", *lines, f".endobj {name}"]))


ENTRIES = {"lbl_1": obj("lbl_1", "\t.float 1.5"), "lbl_2": obj("lbl_2", '\t.string "hello"'),
           "lbl_3": obj("lbl_3", "\t.double 4503601774854144"),
           "__vt__3Car": obj("__vt__3Car", "\t.4byte lbl_rtti", "\t.4byte 0x00000000", "\t.4byte __dt__3CarFv",
                             "\t.4byte Speed__3CarFv", "\t.4byte Go__4BaseFf")}


def compact(*insns: str) -> str:
    return "\n".join([".fn f__3CarFP3Car, global"] + [f"{4 * i:5X}  {x}" for i, x in enumerate(insns)]
                     + [".endfn f__3CarFP3Car"])


class AnnotateDwarfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            raise unittest.SkipTest("compiler not installed (./dk setup)")
        from tools.dk.platforms.gc_mwcc.draftfields import Layouts
        from tools.dk.platforms.gc_mwcc.elf import Elf
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "t.cpp").write_text(UNIT)
            cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-fp", "hardware", "-lang=c++",
                   "-sym", "on", "-c", "t.cpp", "-o", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            cls.lay = Layouts(Elf.parse((Path(tmp) / "t.o").read_bytes()))
        cls.index = Index()
        cls.index.add("t.h", HEADER)

    def notes(self, *insns, budget=llmannot.BUDGET):
        ev = llmannot.Evidence(self.lay, self.index, ENTRIES)
        return llmannot.notes(compact(*insns), "f__3CarFP3Car", ev, budget)

    def test_members_through_this_arguments_and_loads(self):
        got = self.notes("lbz r0, 0x8(r4)", "stb r0, 0x8(r3)", "lfs f0, 0x18(r3)", "stfs f0, 0x10(r3)",
                         "lwz r5, 0x14(r3)", "lwz r6, 0x0(r5)", "addi r7, r3, 0xc")
        self.assertEqual(got, {0x0: "arg0->flag (unsigned char)", 0x4: "this->flag (unsigned char)",
                               0x8: "this->speed (float)", 0xC: "this->pos.y (float)", 0x10: "this->mNext (Car*)",
                               0x14: "this->mNext->a (int)", 0x18: "&this->pos"})

    def test_never_guesses(self):
        got = self.notes("lwz r0, 0x8(r3)",          # a 4-byte load of a 1-byte member
                         "lfs f0, 0x0(r3)",          # a float load of an int
                         "lwz r0, 0x40(r3)",         # no member there
                         "lwz r0, 0x0(r9)",          # r9 is not known
                         "stw r0, gU@sda21(r13)",    # a union: which member is unknown
                         "lwz r0, 0x8(r1)")          # the stack
        self.assertEqual(got, {})

    def test_globals_calls_data_and_virtual_calls(self):
        got = self.notes("lwz r3, gCar@sda21(r13)", "lwz r4, 0x14(r3)", "lfs f1, lbl_1@sda21(r2)", "li r4, lbl_2@sda21",
                         "lfd f2, lbl_3@sda21(r2)", "bl Helper__FP3Carf", "bl GetCar__Fi", "lwz r5, 0x14(r3)",
                         "lwz r12, 0x4(r31)", "lwz r12, 0x10(r12)", "mtctr r12", "bctrl")
        self.assertEqual(got[0x0], "gCar (Car*)")
        self.assertEqual(got[0x4], "gCar->mNext (Car*)")
        self.assertEqual(got[0x8], "1.5f")
        self.assertEqual(got[0xC], '"hello"')
        self.assertEqual(got[0x10], "int→float bias")
        self.assertEqual(got[0x14], "int Helper(Car*, float)")
        self.assertEqual(got[0x18], "Car* GetCar(int)")
        self.assertEqual(got[0x1C], "GetCar()->mNext (Car*)")
        self.assertNotIn(0x20, got)                  # r31 is not this here (never assigned from r3)
        got = self.notes("mr r31, r3", "lwz r12, 0x4(r31)", "lwz r12, 0x10(r12)", "mtctr r12", "bctrl")
        self.assertEqual(got, {0x10: "virtual Base::Go(float)"})

    def test_static_method_has_no_this(self):
        ev = llmannot.Evidence(self.lay, self.index, ENTRIES)
        asm = "\n".join([".fn Make__3CarFv, global", "    0  lwz r0, 0x8(r3)", ".endfn Make__3CarFv"])
        self.index.add("c.h", "class Car { public: static Car* Make(); };")
        self.assertEqual(llmannot.notes(asm, "Make__3CarFv", ev), {})

    def test_budget_and_rendering_are_deterministic(self):
        insns = ("lfs f0, 0x18(r3)", "lfs f1, 0x18(r3)", "bl Helper__FP3Carf", "lfs f2, 0x18(r3)")
        full = self.notes(*insns)
        self.assertEqual(full, self.notes(*insns))
        small = self.notes(*insns, budget=len("this->speed (float)") + len("int Helper(Car*, float)") + 8)
        self.assertEqual(small, {0x0: "this->speed (float)", 0x8: "int Helper(Car*, float)"})   # repeats go first
        asm = compact(*insns)
        text = llmannot.apply(asm, full)
        self.assertIn("    0  lfs f0, 0x18(r3)  ; this->speed (float)", text)
        self.assertEqual(llmannot.apply_stored(asm, llmannot.stored(asm, full)), text)
        self.assertEqual(llmannot.apply_stored(asm.replace("lfs f2", "lfs f3"), llmannot.stored(asm, full)).count(";"), 3)


class AnnotateNoLayoutTest(unittest.TestCase):
    def test_without_layouts_only_calls_and_data(self):
        ev = llmannot.Evidence(None, None, ENTRIES)
        got = llmannot.notes(compact("lwz r0, 0x8(r3)", "bl Helper__FP3Carf", "lfs f1, lbl_1@sda21(r2)"),
                             "f__3CarFP3Car", ev)
        self.assertEqual(got, {0x4: "Helper(Car*, float)", 0x8: "1.5f"})

    def test_coverage_helpers(self):
        self.assertEqual(llmannot.named_members(["this->pos.y (float)", "int Helper(Car*, float)", "&this->mArr[0]"]),
                         {"pos", "y", "mArr"})
        self.assertEqual(llmannot.answer_fields("void C::f() { a->b = c.d + 1.0f; e = g(); x->h(); }", {"e", "z"}),
                         {"b", "d", "e"})
        self.assertEqual([llmannot.kind_of(c) for c in ("this->a (int)", "int f(int)", "virtual A::f()", '"x"', "1.5f",
                                                        "gCar (Car*)", "&gObj (Obj)")],
                         ["member", "call", "virtual call", "string", "constant", "global", "global address"])


class DiagTest(unittest.TestCase):
    def rec(self, wrong, exact, rows=(), cls="shape"):
        return {"wrong": wrong, "exact": exact, "rows": list(rows), "diff_class": cls}

    def test_register_swap_and_declaration_order(self):
        rows = ["  ...", f"{'1C':>5}  {'mr r31, r3':<36} {'arg':>3}  mr r30, r3",
                f"{'20':>5}  {'mr r30, r4':<36} {'arg':>3}  mr r31, r4"]
        r = self.rec("void f(A* a, B* b) {\n    A* x = a;\n    B* y = b;\n    g(x, y);\n}",
                     "void f(A* a, B* b) {\n    B* y = b;\n    A* x = a;\n    g(x, y);\n}", rows, "register")
        self.assertEqual(llmdiag.diagnose(r), "DIAG: registers r30/r31 swapped → declare `y` before `x`")

    def test_source_changes(self):
        cases = [
            ("void f() {\n    a = 1;\n    b();\n}", "void f() {\n    b();\n    a = 1;\n}", "statement order"),
            ("int f(int x) {\n    if (x) {\n        return 1;\n    } else {\n        return 2;\n    }\n}",
             "int f(int x) {\n    return x ? 1 : 2;\n}", "ternary"),
            ("void f() {\n    if (a < b) {\n        g();\n    }\n}", "void f() {\n    if (b > a) {\n        g();\n    }\n}",
             "comparison"),
            ("void f() {\n    int t;\n    g();\n    t = h();\n    k(t);\n}", "void f() {\n    g();\n    int t = h();\n    k(t);\n}",
             "join init"),
            ("void f() {\n    int i;\n    k(i);\n}", "void f() {\n    unsigned int i;\n    k(i);\n}", "type"),
            ("void f() {\n    x = x + y;\n}", "void f() {\n    x += y;\n}", "compound"),
            ("void f() {\n    k(a + b);\n}", "void f() {\n    k(b + a);\n}", "operands"),
        ]
        for wrong, exact, want in cases:
            self.assertEqual(llmdiag.diag_class(self.rec(wrong, exact)), want, (wrong, exact))
        self.assertEqual(llmdiag.diagnose({"wrong": "", "exact": "int f();"}), "")
        self.assertEqual(llmdiag.diagnose({**self.rec("a", "b"), "diag": "DIAG: given"}), "DIAG: given")
        long = self.rec("void f() {\n" + "    a();\n" * 3 + "}", "void f() {\n" + "    b" * 200 + "();\n}")
        self.assertLessEqual(len(llmdiag.diagnose(long)), llmdiag.MAX)

    def test_symptom_of_a_shape_miss(self):
        rows = [f"{'8':>5}  {'lwz r3, 0x4(r3)':<36} {'op':>3}  lfs f1, 0x4(r3)", f"{'C':>5}  {'blr':<36} {'-':>3}  "]
        self.assertEqual(llmdiag.symptom(rows), "at 8 the target has `lwz r3, 0x4(r3)`, ours `lfs f1, 0x4(r3)`")
        self.assertEqual(llmdiag.symptom([]), "the instructions differ")


class Round2HooksTest(unittest.TestCase):
    REC = {"unit": "u", "function": "f__3CarFP3Car", "asm": compact("lfs f0, 0x18(r3)", "blr"), "decls": "",
           "wrong": "void Car::f(Car* o) {\n    int t;\n    t = 1;\n}", "wrong_percent": 50.0, "rows": [],
           "exact": "void Car::f(Car* o) {\n    int t = 1;\n}", "diff_class": "shape"}

    def test_prompt_version_8_is_opt_in(self):
        self.assertIn(8, llmtry.PROMPT_VERSIONS)
        self.assertEqual(llmtry.PROMPT_VERSION, 6)
        self.assertEqual(llmdata.view_version(8), 7)
        self.assertEqual(llmdata.view_version(7), 7)
        self.assertFalse(llmtry.Settings().diag)

    def test_export_extras(self):
        plain = llmdata.render(self.REC)
        annot = {("u", "f__3CarFP3Car"): {"x": {"notes": [["0", "lfs f0, 0x18(r3)", "this->speed (float)"]]}}}
        r2 = llmdata.Round2(annot, 1.0, 1.0)
        got = r2.render(self.REC, None, None)
        self.assertTrue(got["annotated"])
        self.assertIn("lfs f0, 0x18(r3)  ; this->speed (float)", got["messages"][1]["content"])
        self.assertTrue(got["messages"][-1]["content"].startswith("DIAG: "))
        self.assertTrue(got["messages"][-1]["content"].split("\n", 1)[1].startswith(llmtry.PREFILL))
        self.assertEqual(got["messages"][0], plain["messages"][0])
        none = llmdata.Round2(annot, 0.0, 0.0).render(self.REC, None, None)
        self.assertEqual(none["messages"], plain["messages"])          # shares at 0: the plain record
        half = llmdata.Round2(annot, 0.5, 0.0)
        picks = [half.asm({**self.REC, "wrong": f"w{i}"}) is not None for i in range(200)]
        self.assertTrue(60 < sum(picks) < 140)
        self.assertEqual(picks, [half.asm({**self.REC, "wrong": f"w{i}"}) is not None for i in range(200)])

    def test_llm_try_parses_a_diag_reply(self):
        code, fmt, diag = llmtry.extract_diag(" r30/r31 swapped → declare `t` first\n```cpp\nint f() {}\n```", "stop")
        self.assertEqual((code, fmt, diag), ("int f() {}\n", "diag", "DIAG: r30/r31 swapped → declare `t` first"))
        self.assertEqual(llmtry.extract_diag("DIAG: x\n```cpp\nint f() {}\n", "stop")[:2], ("int f() {}\n", "diag"))
        self.assertEqual(llmtry.extract_diag(" x\n```cpp\nint f() {", "length")[1], "truncated")
        self.assertEqual(llmtry.extract_diag(" only a line", "stop")[1], "none")
        self.assertEqual(llmtry.extract("DIAG: x\nint f() {\n}")[0], "int f() {\n}\n")
        self.assertEqual(llmtry.diag_prefill(llmtry.NO_THINK + llmtry.PREFILL), llmtry.NO_THINK + "DIAG:")
        st = llmtry.Settings(think=False, diag=True)
        self.assertEqual(st.options(1, 0, True)["stop"], ["\n```\n"])
        self.assertEqual(st.options(1)["stop"], ["```"])


if __name__ == "__main__":
    unittest.main()
