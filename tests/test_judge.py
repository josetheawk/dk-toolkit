import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc.dwarftypes import _modified, local_types
from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.judge import (aligned_decls, aliasing_locals, bodies, copy_only_locals, empty_cases, judge,
                                              offset_casts, parameters, replace_typeof, review, same_type_casts,
                                              source_reasons, unit_reasons)
from tools.dk.platforms.gc_mwcc.matching import FunctionScore, UnitScore

REPO = Path(__file__).resolve().parents[1]


def body(src: str):
    return cbody.parse_body(src, src.index("{"))


def unit(fns: dict[str, float], data=None, layout=()) -> UnitScore:
    return UnitScore("u", [FunctionScore(n, 4, p) for n, p in fns.items()], dict(data or {}), list(layout))


class SourceGateTest(unittest.TestCase):
    def test_devices_added_by_the_candidate(self):
        before = "int f(int a) { return a + 1; }\n"
        after = ("#pragma opt_lifetimes off\nint f(int a) { volatile int k = a; register int r;\n"
                 "    __typeof__(a) t = a; char pad[4]; asm { nop } return k + 1; }\n")
        got = source_reasons(before, after)
        self.assertEqual(got, ["adds 1 volatile", "adds 1 pragma", "adds 1 asm", "adds 1 __typeof__", "adds 1 padding",
                               "adds 1 register keyword"])

    def test_devices_already_there_are_not_counted(self):
        src = "volatile int g;\nint f() { return g; }\n"
        self.assertEqual(source_reasons(src, src.replace("return g;", "return 0 + g;")), [])

    def test_forbidden_quality_rules_apply(self):
        rule = ("forbidden pointer-offset", re.compile(r"\(char\s*\*\)\s*\w+\s*\+\s*0x"))
        self.assertEqual(source_reasons("f() {}", "f() { x = (char*)p + 0x10; }", [rule]), ["adds 1 forbidden pointer-offset"])

    def test_alignment_on_statics_is_a_device(self):
        before = "static int g;\nint f() { return g; }\n"
        for dev in ("static int g __attribute__((aligned(8)));", "static int g __attribute__((__aligned__(8)));",
                    "__declspec(align(8)) static int g;", "static int g ATTRIBUTE_ALIGN(8) = 1;", "int g __attribute__((aligned(8)));"):
            self.assertEqual(source_reasons(before, before.replace("static int g;", dev))[:1],
                             ["adds 1 aligned static (an alignment attribute on a static or file-level object)"], dev)
        self.assertEqual(source_reasons("static int g __attribute__((aligned(8)));\n",
                                        "static int g __attribute__((aligned(8)));\nstatic int h;\n"), [])
        local = "int f() { static int k __attribute__((aligned(8))); return k; }\n"
        self.assertEqual(len(source_reasons(before, local)), 1)

    def test_alignment_of_a_stack_local_needs_judgment(self):
        before = "struct M { float m[4][4]; };\nvoid g(M*);\nvoid f() {\n    g(0);\n}\n"
        after = before.replace("    g(0);", "    M m __attribute__((aligned(32)));\n    g(&m);")
        self.assertEqual(aligned_decls(after), (0, 1))
        self.assertEqual(source_reasons(before, after), [])
        cat, flags = review(before, after, "f")
        self.assertEqual(cat, "needs-judgment")
        self.assertTrue(flags[0].startswith("adds 1 aligned stack local"), flags)
        self.assertEqual(review(after, after, "f"), ("clean", []))          # already there


class CopyOnlyTest(unittest.TestCase):
    def check(self, before: str, after: str) -> list[str]:
        return copy_only_locals(before, body(before), after, body(after))

    def test_copies_and_unread_locals_are_devices(self):
        before = "void f(int a) { g(a); }"
        self.assertEqual(self.check(before, "void f(int a) { int b = a; g(b); }"), ["new local b only copies a"])
        self.assertEqual(self.check(before, "void f(int a) { int b; b = a; g(b); }"), ["new local b only copies a"])
        self.assertEqual(self.check(before, "void f(int a) { int b = 3; g(a); }"), ["new local b is never read"])

    def test_named_values_and_references_are_idioms(self):
        before = "void f(S* p, int i) { g(p->v[i].x * 2 + p->v[i].y); }"
        self.assertEqual(self.check(before, "void f(S* p, int i) { int t = p->v[i].x * 2; g(t + p->v[i].y); }"), [])
        self.assertEqual(self.check(before, "void f(S* p, int i) { V& e = p->v[i]; g(e.x * 2 + e.y); }"), [])


class OffsetCastTest(unittest.TestCase):
    def count(self, stmt: str) -> int:
        src = "void f(S* p, int i, int n) { " + stmt + " }"
        return offset_casts(src, body(src))

    def test_fields_by_offset(self):
        self.assertEqual(self.count("g(*(int*)((char*)p->m4 + i * 0x3DC + 0x248));"), 1)
        self.assertEqual(self.count("*((char*)p->m40 + 0x15E30) = 1;"), 1)
        self.assertEqual(self.count("g((V*)((unsigned char*)p + 8));"), 1)

    def test_byte_arithmetic_and_data_offsets(self):
        self.assertEqual(self.count("copy((char*)p + n, (char*)p->m4 + n + 4, i);"), 0)     # stays bytes
        self.assertEqual(self.count("char* end = (char*)p + size(i) + 0x10;"), 0)
        self.assertEqual(self.count("char* e = (char*)p + (q + 2)->m4;"), 0)                 # the offset is data
        self.assertEqual(self.count("p = (S*)((unsigned char*)p + ((p->m6 * 2 + 3) & ~3));"), 0)   # a size
        self.assertEqual(self.count("char* b = (char*)alloc(n + 1);"), 0)
        src = "void f(S* p, int i) { char* e = (char*)p + i * 0x60 + 0x74; g(e); }"
        self.assertEqual(offset_casts(src, body(src), loose=True), 1)                    # bytes: a look, not a reject


class UnitGateTest(unittest.TestCase):
    def test_exact_and_nothing_worse(self):
        before = unit({"f": 99.0, "g": 100.0, "h": 80.0}, {".data": 100.0})
        self.assertEqual(unit_reasons(before, unit({"f": 100.0, "g": 100.0, "h": 85.0}, {".data": 100.0}), "f"), [])
        got = unit_reasons(before, unit({"f": 100.0, "g": 99.5, "h": 80.0}, {".data": 90.0}, ["x: missing"]), "f")
        self.assertEqual(got, ["g was exact, now 99.50%", ".data 100.00% before, now 90.00%", "new layout problem: x: missing"])
        self.assertEqual(unit_reasons(before, unit({"f": 99.5, "g": 100.0, "h": 80.0}, {".data": 100.0}), "f"),
                         ["f is at 99.50%, not exact"])

    def test_a_function_already_off_may_move_without_a_new_problem(self):
        before = unit({"f": 99.0, "g": 100.0}, layout=["g: at .text+0x36A0 (after stripping), target .text+0x634"])
        moved = unit({"f": 100.0, "g": 100.0}, layout=["g: at .text+0x36DC (after stripping), target .text+0x634"])
        self.assertEqual(unit_reasons(before, moved, "f"), [])
        worse = unit({"f": 100.0, "g": 100.0}, layout=["g: at .text+0x36DC (after stripping), target .text+0x634",
                                                       "h: missing from compiled object"])
        self.assertEqual(unit_reasons(before, worse, "f"), ["new layout problem: h: missing from compiled object"])

    def test_the_verdict_combines_every_gate(self):
        b, a = "void f(int a) { g(a); }", "void f(int a) { volatile int b = a; g(b); }"
        v = judge(b, a, body(b), body(a), unit({"f": 90.0}), unit({"f": 100.0}), "f")
        self.assertFalse(v.ok)
        self.assertEqual(v.reasons, ["adds 1 volatile", "new local b only copies a"])
        self.assertTrue(judge(b, b, body(b), body(b), unit({"f": 90.0}), unit({"f": 100.0}), "f").ok)


class TypeofTest(unittest.TestCase):
    def test_typeof_declarations_get_the_real_type(self):
        src = "void f(S* p, int i) { __typeof__(p->v[i])& e = p->v[i]; __typeof__(i * 2) t = i * 2; __typeof__(x) u; g(e, t); }"
        new, missing = replace_typeof(src, body(src), {"e": "Vec&", "t": "int"})
        self.assertEqual(new, "void f(S* p, int i) { Vec& e = p->v[i]; int t = i * 2; __typeof__(x) u; g(e, t); }")
        self.assertEqual(missing, ["u"])

    def test_modifiers_apply_outermost_first(self):
        self.assertEqual(_modified(bytes([1, 3]), "char"), "const char*")          # pointer to const char
        self.assertEqual(_modified(bytes([3, 1]), "char"), "char* const")          # const pointer to char
        self.assertEqual(_modified(bytes([2]), "Vec"), "Vec&")


class CopyOnlyRefinementTest(unittest.TestCase):
    def flags(self, after: str) -> list[str]:
        return copy_only_locals("int f(int a) { return 0; }", body("int f(int a) { return 0; }"), after, body(after))

    def test_natural_copies_pass(self):
        self.assertEqual(self.flags("int f(int a) { int old = m; m = a; return old; }"), [])      # keeps the old value
        self.assertEqual(self.flags("int f(int a) { int r = a; r += 2; return r; }"), [])         # changed in place
        self.assertEqual(self.flags("int f(int a) { static int s = 0; s = 1; return a; }"), [])   # data, not a local
        self.assertEqual(self.flags("int f(int a) { int on = false; if (a) on = true; return on; }"), [])
        self.assertEqual(self.flags("int f(int a) { int t = a; return t; }"), ["new local t only copies a"])

    def with_params(self, after: str, params: dict[str, str]) -> list[str]:
        empty = body("int f() { return 0; }")
        return copy_only_locals("int f() { return 0; }", empty, after, body(after), params)

    def test_copies_that_keep_a_value_are_code(self):
        p = {"n": "int"}
        loop = "float f(int n) { float best; for (int i = 0; i < n; i++) { float d = g(i); if (i == 0 || d < best) best = d; } return best; }"
        self.assertEqual(self.with_params(loop, p), [])                         # best so far: d is new every iteration
        walk = "int f(int n) { int last; int v = 0; while (v < n) { last = v; v = g(v); } return last; }"
        self.assertEqual(self.with_params(walk, p), [])                         # v changes later in the loop
        self.assertEqual(self.with_params("int f(int n) { int r; if (n) r = a; else r = b; return r; }", p), [])  # a merge
        self.assertEqual(self.with_params("int f(int n) { TxtStr s(lbl_80001234); return g(s); }", p), [])  # a constructor
        self.assertEqual(self.with_params("int f(char c) { int x = c; return x > 0x41; }", {"c": "char"}), [])  # a conversion
        self.assertEqual(self.with_params("int f(int n) { int v = m4; g(n); return v + m4; }", p), [])    # kept across a call
        self.assertEqual(self.with_params("float f(float a) { float old = m74; m88 = old * a; return old; }",
                                          {"a": "float"}), [])                  # kept across a store
        self.assertEqual(self.with_params("int f(int n) { int iv = 0; h(&iv); int t = iv; h(&iv); return t + iv; }", p), [])
        self.assertEqual(self.with_params("int f(int n) { int first = g(n); return n; }", p), [])     # the call is the code
        self.assertEqual(self.with_params("int f(int n) { B* b = cast<B>(n); return n; }", p), [])     # a template's too

    def test_pure_aliases_are_devices(self):
        p = {"n": "int", "value": "int"}
        self.assertEqual(self.with_params("int f(int n) { int k = n; int s = 0; for (int i = 0; i < k; i++) s += i; return s; }", p),
                         ["new local k only copies n"])                         # the loop never changes n
        self.assertEqual(self.with_params("int f(int value) { volatile int v; v = value; return h(v); }", p),
                         ["new local v only copies value"])                     # the volatile spill
        self.assertEqual(self.with_params("int f(int n) { float u = n + 1.0f; return n; }", p), ["new local u is never read"])
        self.assertEqual(self.with_params("int f(int n) { int mine = m370; return mine == -1 || mine > n; }", p),
                         ["new local mine only caches m370 (a member or global nothing changes in between)"])

    def test_a_cached_member_needs_judgment(self):
        src = "struct R { int m370; int f(int n); };\nint R::f(int n) {\n    int mine = m370;\n    return mine > n;\n}\n"
        self.assertEqual(review("struct R { int m370; int f(int n); };\n", src, "R::f"),
                         ("needs-judgment", ["new local mine only caches m370 (a member or global nothing changes in between)"]))
        alias = src.replace("int mine = m370;", "int k = n;").replace("mine > n", "m370 > k")
        self.assertEqual(review("struct R { int m370; int f(int n); };\n", alias, "R::f"),
                         ("reject", ["new local k only copies n", "new local k aliases n (n is not used after it)"]))


class AliasingLocalTest(unittest.TestCase):
    SRC = ("unsigned long f(int base, int* out, void (*cb)(int)) {\n    int n = 0;\n    while (n < base) { n++; }\n"
           "    *out = n;\n    return n;\n}\n")

    def flags(self, before: str, after: str) -> list[str]:
        b_body, a_body, span = bodies(before, after, "f")
        return aliasing_locals(b_body, after, a_body, parameters(after, span))

    def test_parameters(self):
        self.assertEqual(parameters(self.SRC, (0, len(self.SRC))), {"base": "int", "out": "int*", "cb": "void(*)(int)"})
        ctor = "A::A(int x, const char*) : B(x), m4(0) {}"
        self.assertEqual(parameters(ctor, (0, len(ctor))), {"x": "int"})

    def test_a_copy_used_in_place_of_the_parameter(self):
        after = self.SRC.replace("int n = 0;", "int n = 0;\n    int temp_base = base;").replace("n < base", "n < temp_base") \
            .replace("*out = n;", "if (temp_base == 0) temp_base = 10;\n    *out = n;")
        self.assertEqual(self.flags(self.SRC, after), ["new local temp_base aliases base (base is not used after it)"])
        cat, flags = review(self.SRC, after, "f")
        self.assertEqual(cat, "needs-judgment")

    def test_not_flagged(self):
        both = self.SRC.replace("int n = 0;", "int n = 0;\n    int b = base;").replace("n < base", "n < b + base")
        self.assertEqual(self.flags(self.SRC, both), [])                                     # the original still used
        other = self.SRC.replace("int n = 0;", "int n = 0;\n    short b = base;").replace("n < base", "n < b")
        self.assertEqual(self.flags(self.SRC, other), [])                                    # another type: a conversion
        already = self.SRC.replace("int n = 0;", "int n = 0;\n    int b = base;").replace("n < base", "n < b")
        self.assertEqual(self.flags(already, already.replace("n++", "n += 1")), [])           # the source has it
        member = "struct S { int m4; int f(); };\nint S::f() {\n    int v = m4;\n    return v + 1;\n}\n"
        b_body, a_body, span = bodies("struct S { int m4; int f(); };\n", member, "S::f")
        self.assertEqual(aliasing_locals(b_body, member, a_body, parameters(member, span)), [])   # a member, not a variable


class SameTypeCastTest(unittest.TestCase):
    HEAD = ("namespace Ns { struct NpcBase { int x; }; struct Cop : NpcBase {}; }\ntypedef unsigned long u32;\n"
            "struct Chase { Ns::NpcBase* m2C[3]; unsigned int m38; Ns::Cop* m3C;\n"
            "    Ns::NpcBase* NthFollower(unsigned int i) { return i < m38 ? m2C[i] : 0; } };\n")
    SRC = HEAD + "int f(Chase* chase, short s, u32 u) {\n    Ns::NpcBase* first = chase->NthFollower(0);\n    return first->x + s;\n}\n"

    def fake(self, same: set[str], steers: bool = True):
        """A stand-in compiler: a probe compiles when its operand's type is listed as the cast's; the
        function scores 100% with every cast in place and (`steers`) less without one."""
        def probe(text: str):
            m = re.search(r"\*\((.+?)\*\*\)0 = \(__typeof__\((.+?)\)\*\)0", text)
            if m is not None:
                return 100.0 if (m[1], m[2]) in same else None
            return 100.0 if not steers or "(Ns::NpcBase*)chase" in text else 98.5
        return probe

    def casts(self, after: str, probe):
        b_body, a_body, _ = bodies(self.SRC, after, "f")
        return same_type_casts(self.SRC, b_body, after, a_body, probe)

    def test_the_probe_and_the_flags(self):
        after = self.SRC.replace("= chase->NthFollower(0)", "= (Ns::NpcBase*)chase->NthFollower(0)")
        same = {("Ns::NpcBase*", "chase->NthFollower(0)")}
        flag = "cast to its own type changes the code (Ns::NpcBase*)chase->NthFollower(0)"
        self.assertEqual(self.casts(after, self.fake(same)), ([flag], []))
        self.assertEqual(self.casts(after, self.fake(same, steers=False)),                  # no effect: noise to drop
                         ([], ["redundant cast to its own type (Ns::NpcBase*)chase->NthFollower(0)"]))
        self.assertEqual(self.casts(after, self.fake(set())), ([], []))                    # not its type: kept
        self.assertEqual(self.casts(self.SRC, self.fake(same)), ([], []))                  # no cast added
        self.assertEqual(review(self.SRC, after, "f", probe=self.fake(same)), ("reject", [flag]))
        self.assertEqual(judge(self.SRC, after, None, None, None, None, "f", probe=self.fake(same)).reasons, [flag])
        broken = self.casts(after, lambda text: None)                                       # fails safe: needs a look
        self.assertTrue(broken[1][0].startswith("cast check could not run"))

    def test_real_compiler(self):
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            self.skipTest("compiler not installed (./dk setup)")

        def probe(text: str):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "t.cpp").write_text(text)
                cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-fp", "hardware", "-lang=c++",
                       "-c", "t.cpp", "-o", tmp]
                return 100.0 if subprocess.run(cmd, capture_output=True, text=True, cwd=tmp).returncode == 0 else None
        same = self.SRC.replace("= chase->NthFollower(0)", "= (Ns::NpcBase*)chase->NthFollower(0)") \
            .replace("+ s;", "+ (int)(u32)u + (short)s;")
        self.assertEqual(self.casts(same, probe)[1], ["redundant cast to its own type (Ns::NpcBase*)chase->NthFollower(0)",
                                                      "redundant cast to its own type (u32)u",
                                                      "redundant cast to its own type (short)s"])
        other = self.SRC.replace("= chase->NthFollower(0)", "= (Ns::NpcBase*)chase->m3C").replace("+ s;", "+ (int)s;")
        self.assertEqual(self.casts(other, probe), ([], []))                               # an upcast, a widening


class DwarfTypeTest(unittest.TestCase):
    """Local types from the real compiler's debug info (skipped without the project's tools)."""

    def compile(self, code: str, lang: str) -> Elf:
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            self.skipTest("compiler not installed (./dk setup)")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / ("t.cpp" if lang == "c++" else "t.c")
            src.write_text(code)
            cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-fp", "hardware", f"-lang={lang}",
                   "-sym", "on", "-c", str(src), "-o", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return Elf.parse((Path(tmp) / "t.o").read_bytes())

    def test_cplusplus_locals(self):
        elf = self.compile("struct Vec { float x, y; }; struct Car { int n; Vec v[4]; }; extern void g(void*); extern float h(float);\n"
                           "void t(Car* c, int i, const char* s) { __typeof__(c->v[i])& e = c->v[i];\n"
                           "  __typeof__(h(1.0f) * 2.0f) b = h(1.0f) * 2.0f; __typeof__(s) q = s + i; __typeof__(c) w = c + i;\n"
                           "  g(&e); e.x = h(e.y); g(&e); g(&b); g((void*)q); g(w); g((void*)q); g(w); }\n", "c++")
        self.assertEqual(local_types(elf, "t__FP3CariPCc"), {"e": "Vec&", "b": "float", "q": "const char*", "w": "Car*"})

    def test_c_structs_keep_their_tag(self):
        elf = self.compile("struct Vec { float x, y; }; struct Car { int n; struct Vec v[4]; }; extern void g(void*);\n"
                           "void t(struct Car* c, int i) { __typeof__(&c->v[i]) e; __typeof__(c->n * 2) a;\n"
                           "  e = &c->v[i]; a = c->n * 2; g(e); g(&a); g(e); }\n", "c")
        self.assertEqual(local_types(elf, "t", c=True), {"e": "struct Vec*", "a": "int"})


class ReviewFalsePositiveTest(unittest.TestCase):
    """Accepted code the audit (llm-review --audit) flagged wrongly."""
    B = "struct Foo { Foo(int, int); };\nint g;\n"

    def test_constructor_arguments_are_not_comma_operators(self):
        self.assertEqual(review(self.B, self.B + "void f(int a) {\n    Foo x(a, 2);\n}\n", "f"), ("clean", []))
        self.assertIn("adds a comma operator", review(self.B, self.B + "void f(int a) {\n    Foo x((a, 3), 2);\n}\n", "f")[1])
        self.assertIn("adds a comma operator", review(self.B, self.B + "void f(int a) {\n    g = (a, 2);\n}\n", "f")[1])
        self.assertNotIn("adds a comma operator", review(self.B, self.B + "void f(int a) {\n    Foo* p = new (0, \"Foo\") "
                                                         "Foo(a, 2);\n    g = (int)p;\n}\n", "f")[1])
        self.assertIn("adds a comma operator", review(self.B, self.B + "void f(int a) {\n    Foo* p = new (0, \"Foo\") "
                                                      "Foo((a, 1), 2);\n    g = (int)p;\n}\n", "f")[1])

    def test_unwritten_overload_is_not_an_outside_edit(self):
        c, d1, d2 = "struct K { K(); K(int); int m; };\n", "K::K() : m(0) {}\n", "K::K(int v) : m(v) {}\n"
        self.assertEqual(review(c + d1, c + d1 + d2, "__ct__1KFi"), ("clean", []))
        self.assertEqual(review(c + d2, c + d1 + d2, "__ct__1KFv"), ("clean", []))
        self.assertEqual(review(c + d1 + "K::K(int v) : m(v + 1) {}\n", c + d1 + d2, "__ct__1KFi"), ("clean", []))


class EmptyCaseTest(unittest.TestCase):
    B = "extern void g(int);\nvoid f(int a) {\n    switch (a) {\n    case 0:\n        g(1);\n        break;\n    }\n}\n"

    def count(self, src: str) -> int:
        return empty_cases(cbody.parse_body(src, src.index("{")))

    def test_labels_with_no_body_of_their_own(self):
        self.assertEqual(self.count("void f(int a) { switch (a) { case 0: g(); break; case 1: case 2: break; } }"), 2)
        self.assertEqual(self.count("void f(int a) { switch (a) { case 1: {} case 2: ; default: break; } }"), 2)
        self.assertEqual(self.count("void f(int a) { switch (a) { case 1: default: break; } }"), 1)

    def test_shared_bodies_and_defaults_are_natural(self):
        self.assertEqual(self.count("void f(int a) { switch (a) { case 0: case 1: g(); break; default: break; } }"), 0)
        self.assertEqual(self.count("int f(int a) { switch (a) { case 0: return 1; case 1: g(); default: return 0; } }"), 0)
        self.assertEqual(self.count("void f(int a) { if (a) { g(); } }"), 0)
        self.assertEqual(self.count("void f(int a) { switch (a) { case 0: g(); break; case 1: h(); break; case 9: break; } }"), 0)
        self.assertEqual(self.count("void f(int a) { switch (a) { case 0: g(); break; case 1: case 2: break; } }"), 2)

    def test_review_needs_judgment_only_for_added_labels(self):
        added = self.B.replace("        break;\n    }", "        break;\n    case 1:\n    case 2:\n        break;\n    }")
        cat, flags = review(self.B, added, "f")
        self.assertEqual(cat, "needs-judgment")
        self.assertTrue(flags[0].startswith("empty case labels: 2"), flags)
        self.assertEqual(review(added, added, "f"), ("clean", []))        # already there: not the candidate's
        shared = self.B.replace("    case 0:\n", "    case 0:\n    case 1:\n")
        self.assertEqual(review(self.B, shared, "f")[0], "needs-judgment")      # only the odd-shape flag
        self.assertFalse(any(x.startswith("empty case") for x in review(self.B, shared, "f")[1]))


if __name__ == "__main__":
    unittest.main()
