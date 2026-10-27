import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.dk.platforms.gc_mwcc import reorder as R
from tools.dk.platforms.gc_mwcc.elf import Elf, ElfSection, ElfSymbol

REPO = Path(__file__).resolve().parents[1]

SOURCE = textwrap.dedent("""\
    #include "a.h"
    struct S {
        int a;   // a { brace in a comment
    };
    extern "C" void ext(const char* s = "}");

    // B's comment
    void B() {
        ext("{");
    }
    static const char* tbl[] = { "x", "y" };
    void A() { ext(tbl[0]); }
    inline int helper() { return 1; }
    #pragma dont_inline on
    void C() {}
    #pragma dont_inline reset
    """)


def lines_of(text: str, it: R.Item) -> str:
    return "\n".join(text.split("\n")[it.start:it.end + 1])


class SplitTest(unittest.TestCase):
    def test_items_cover_every_line_once(self):
        items = R.split_items(SOURCE)
        self.assertEqual("\n".join(lines_of(SOURCE, it) for it in items), SOURCE)
        covered = [n for it in items for n in range(it.start, it.end + 1)]
        self.assertEqual(covered, list(range(SOURCE.count("\n") + 1)))

    def test_kinds_and_leading_comments(self):
        items = R.split_items(SOURCE)
        kinds = [(it.kind, it.block) for it in items]
        self.assertEqual(kinds[0], ("pp", False))
        struct = items[1]
        self.assertEqual((struct.body, struct.end), (1, 3))       # the `}` of a type body waits for its `;`
        b = next(it for it in items if "B" in it.head and it.block)
        self.assertEqual(b.start, 5)                                # the blank line and the comment lead it
        self.assertEqual(b.body, 7)
        table = next(it for it in items if "tbl" in it.head)
        self.assertTrue(R.is_data(table))

    def test_regions_fold_matched_pragmas_and_ifs(self):
        text = "#if X\nint a;\n#endif\n#pragma push\nvoid f() {}\n#pragma pop\n#pragma force_active on\nint b;\n"
        items = R.group_regions(R.split_items(text))
        self.assertEqual([it.kind for it in items], ["region", "region", "pp", "code", "trivia"])
        self.assertEqual((items[1].start, items[1].end), (3, 5))

    def test_overlapping_regions_merge(self):
        text = "#pragma a on\n#pragma b on\nvoid f() {}\n#pragma a reset\nvoid g() {}\n#pragma b reset\n"
        items = R.group_regions(R.split_items(text))
        self.assertEqual(items[0].kind, "region")
        self.assertEqual((items[0].start, items[0].end), (0, 5))

    def test_unbalanced_braces_are_refused(self):
        with self.assertRaises(R.SplitError):
            R.split_items("void f() {\n")


class ClassifyTest(unittest.TestCase):
    def item(self, text: str) -> R.Item:
        return R.split_items(text)[0]

    def test_declarations_are_not_data(self):
        for text in ("extern int x;\n", 'extern "C" void f(int);\n', "struct S;\n", "struct S { int a; };\n",
                     "typedef int T;\n", "int f(int a, char* b);\n", "void C::g() const;\n", "enum { A, B };\n",
                     "class D : public B { int x; };\n"):
            self.assertFalse(R.is_data(self.item(text)), text)

    def test_definitions_and_macro_invocations_are_data(self):
        for text in ("int x = 3;\n", "static int buf[3];\n", "Foo gFoo(1, 2);\n", 'REGISTER(lbl, "Name", fn);\n',
                     "DECLARE(x);\n", "struct { int a; } gS;\n", "int (*fp)(int);\n", "S tbl[] = { {1}, {2} };\n"):
            self.assertTrue(R.is_data(self.item(text)), text)


class PlanTest(unittest.TestCase):
    def test_functions_follow_the_target_and_take_their_data_along(self):
        lines = {"B__Fv": 7, "A__Fv": 11, "C__Fv": 14}
        offsets = {"A__Fv": 0x0, "B__Fv": 0x20, "C__Fv": 0x40}
        p = R.plan(SOURCE, lines, offsets)
        out = p.text.split("\n")
        self.assertLess(out.index("static const char* tbl[] = { \"x\", \"y\" };"), out.index("void A() { ext(tbl[0]); }"))
        self.assertLess(out.index("void A() { ext(tbl[0]); }"), out.index("// B's comment"))
        self.assertGreater(out.index("inline int helper() { return 1; }"), out.index("// B's comment"))   # outside the moved stretch
        self.assertEqual(out[-4:], ["#pragma dont_inline on", "void C() {}", "#pragma dont_inline reset", ""])
        self.assertEqual(sorted(SOURCE.split("\n")), sorted(out))
        self.assertEqual(p.moved, ["A__Fv", "B__Fv"])

    def test_an_ordered_source_is_left_verbatim(self):
        lines = {"B__Fv": 7, "A__Fv": 11, "C__Fv": 14}
        offsets = {"B__Fv": 0x0, "A__Fv": 0x20, "C__Fv": 0x40}
        self.assertEqual(R.plan(SOURCE, lines, offsets).text, SOURCE)

    def test_a_function_the_target_lacks_stays_behind_its_predecessor(self):
        text = "void a() {}\nvoid gone() {}\nvoid b() {}\n"
        p = R.plan(text, {"a__Fv": 0, "gone__Fv": 1, "b__Fv": 2}, {"a__Fv": 0x40, "b__Fv": 0x0})
        self.assertEqual(p.text, "void b() {}\nvoid a() {}\nvoid gone() {}\n")
        self.assertEqual(p.unknown, ["gone__Fv"])

    def test_unrenamed_functions_match_by_address(self):
        text = "void C::fn_80001000() {}\nvoid C::fn_80000F00() {}\n"
        p = R.plan(text, {"fn_80001000__1CFv": 0, "fn_80000F00__1CFv": 1}, {"fn_80000F00": 0, "fn_80001000": 0x100})
        self.assertEqual(p.text, "void C::fn_80000F00() {}\nvoid C::fn_80001000() {}\n")

    def test_namespace_and_linkage_blocks_are_arranged_inside(self):
        text = 'namespace N {\nvoid b() {}\nvoid a() {}\n}  // N\nextern "C" {\nvoid d() {}\nvoid c() {}\n}\n'
        p = R.plan(text, {"b__1NFv": 1, "a__1NFv": 2, "d": 5, "c": 6}, {"a__1NFv": 0, "b__1NFv": 4, "c": 8, "d": 12})
        self.assertEqual(p.text, 'namespace N {\nvoid a() {}\nvoid b() {}\n}  // N\nextern "C" {\nvoid c() {}\nvoid d() {}\n}\n')
        self.assertEqual(p.functions, 4)

    def test_a_define_moves_up_and_a_prototype_covers_an_earlier_use(self):
        text = "void b() { a(); }\n#define X 1\nvoid a() { X; }\n"
        p = R.plan(text, {"a__Fv": 2, "b__Fv": 0}, {"a__Fv": 0x40, "b__Fv": 0})
        self.assertEqual(p.text, text)
        p = R.plan("void a() {}\nvoid b() { a(); }\n", {"a__Fv": 0, "b__Fv": 1}, {"a__Fv": 0x40, "b__Fv": 0})
        self.assertEqual(p.text, "void a();\nvoid b() { a(); }\nvoid a() {}\n")

    def test_data_stays_above_its_first_user(self):
        text = "void a() {}\nstatic int t[2] = {1, 2};\nvoid b() { t[0]; }\nvoid c() { t[1]; }\n"
        p = R.plan(text, {"a__Fv": 0, "b__Fv": 2, "c__Fv": 3}, {"c__Fv": 0, "a__Fv": 0x20, "b__Fv": 0x40})
        self.assertEqual(p.text, "static int t[2] = {1, 2};\nvoid c() { t[1]; }\nvoid a() {}\nvoid b() { t[0]; }\n")

    def test_a_pragma_region_around_functions_kept_apart_is_split(self):
        text = "#pragma dont_inline on\nvoid a() {}\nvoid b() {}\n#pragma dont_inline off\nvoid c() {}\n"
        p = R.plan(text, {"a__Fv": 1, "b__Fv": 2, "c__Fv": 4}, {"a__Fv": 0, "c__Fv": 0x20, "b__Fv": 0x40})
        self.assertEqual(p.text, "#pragma dont_inline on\nvoid a() {}\n#pragma dont_inline off\nvoid c() {}\n"
                                 "#pragma dont_inline on\nvoid b() {}\n#pragma dont_inline off\n")

    def test_a_callee_is_kept_out_of_line_with_dont_inline(self):
        text = "namespace N {\nint f() { return 1; }\nint g() { return f(); }\n}\n"
        self.assertEqual(R.wrap_dont_inline(text, {"f__1NFv": 1, "g__1NFv": 2}, ["f__1NFv"]),
                         "namespace N {\n#pragma dont_inline on\nint f() { return 1; }\n#pragma dont_inline reset\n"
                         "int g() { return f(); }\n}\n")

    def test_items_never_cross_a_barrier(self):
        text = "void b() {}\n#include \"x.h\"\nvoid a() {}\n"
        p = R.plan(text, {"a__Fv": 2, "b__Fv": 0}, {"a__Fv": 0, "b__Fv": 0x20})
        self.assertEqual(p.text, text)
        self.assertTrue(any("never across" in n for n in p.notes))


def fake_elf(functions: list[tuple[str, int]]) -> Elf:
    text = ElfSection(1, ".text", 1, 0, 0x1000, 0, 0, b"")
    syms = tuple(ElfSymbol(n, off, 4, 2, 1, 1) for n, off in functions)
    return Elf((ElfSection(0, "", 0, 0, 0, 0, 0, b""), text), (ElfSymbol("", 0, 0, 0, 0, 0),) + syms)


class MisplacedCopyTest(unittest.TestCase):
    def test_a_copy_after_the_wrong_function_gets_the_rule_s_fix(self):
        text = ElfSection(1, ".text", 1, 0, 0x100, 0, 0, bytes(0x100))
        null = (ElfSymbol("", 0, 0, 0, 0, 0),)
        target = Elf((ElfSection(0, "", 0, 0, 0, 0, 0, b""), text),
                     null + (ElfSymbol("f1__Fv", 0, 4, 2, 1, 1), ElfSymbol("w__1AFv", 4, 4, 2, 1, 1), ElfSymbol("f2__Fv", 8, 4, 2, 1, 1)))
        ours = Elf((ElfSection(0, "", 0, 0, 0, 0, 0, b""), text),
                   null + (ElfSymbol("f1__Fv", 0, 4, 2, 1, 1), ElfSymbol("f2__Fv", 4, 4, 2, 1, 1), ElfSymbol("w__1AFv", 8, 4, 2, 2, 1)))
        [line] = R.misplaced_copies(target, ours)
        self.assertTrue(line.startswith("w__1AFv: after f2__Fv in ours, after f1__Fv in the target"))
        self.assertIn("reference it from f1__Fv", line)
        self.assertEqual(R.misplaced_copies(target, target), [])


class OrderTest(unittest.TestCase):
    def test_out_of_order_is_what_the_longest_ordered_run_leaves(self):
        ours = fake_elf([("a", 0), ("c", 4), ("b", 8), ("d", 12)])
        self.assertEqual(len(R.out_of_order(ours, {"a": 0, "b": 4, "c": 8, "d": 12})), 1)
        self.assertEqual(R.out_of_order(ours, {"a": 0, "c": 4, "b": 8, "d": 12}), [])


class DeferredTest(unittest.TestCase):
    def test_a_tail_running_backwards_was_emitted_last(self):
        elf = fake_elf([("a", 0), ("b", 4), ("y", 8), ("x", 12)])
        self.assertEqual(R.deferred_functions(elf, {"a": 1, "b": 5, "x": 3, "y": 9}), {"x"})    # y may be the last ordinary one
        self.assertEqual(R.deferred_functions(elf, {"a": 1, "b": 5, "y": 4, "x": 3}), {"x", "y"})
        self.assertEqual(R.deferred_functions(elf, {"a": 1, "b": 5, "y": 7, "x": 9}), set())
        self.assertEqual(R.deferred_functions(elf, {"a": 1, "b": 5, "y": 2, "x": 9}), set())

    def test_deferred_functions_are_defined_in_reverse_target_order(self):
        text = "void x() {}\nvoid a() {}\nvoid y() {}\n"
        lines, offsets = {"x__Fv": 0, "a__Fv": 1, "y__Fv": 2}, {"a__Fv": 0, "x__Fv": 4, "y__Fv": 8}
        self.assertEqual(R.plan(text, lines, offsets, {"x__Fv", "y__Fv"}).text, "void y() {}\nvoid a() {}\nvoid x() {}\n")


class MirrorTest(unittest.TestCase):
    def test_the_variant_replaces_only_its_own_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            (root / "src" / "x").mkdir(parents=True)
            (root / "include").mkdir()
            (root / "src" / "x" / "a.cpp").write_text("old")
            (root / "src" / "x" / "near.h").write_text("h")
            (root / "src" / "other.cpp").write_text("o")
            tree = R.mirror(SimpleNamespace(root=root), "src/x/a.cpp", "new", Path(tmp) / "work")
            self.assertEqual((tree / "src" / "x" / "a.cpp").read_text(), "new")
            self.assertEqual((root / "src" / "x" / "a.cpp").read_text(), "old")
            self.assertTrue((tree / "src" / "x" / "near.h").is_symlink())
            self.assertTrue((tree / "src" / "other.cpp").is_symlink())
            self.assertTrue((tree / "include").is_symlink())


COMPILER = REPO / "build" / "compilers" / "GC" / "1.3" / "mwcceppc.exe"
WIBO = REPO / "build" / "tools" / "wibo"


def compile_cpp(src: str, flags: tuple[str, ...] = ()) -> Elf:
    tmp = Path(tempfile.mkdtemp(dir=REPO / "build"))
    try:
        (tmp / "t.cpp").write_text(src)
        proc = subprocess.run([str(WIBO), str(COMPILER), "-nodefaults", "-proc", "gekko", "-O4,p", "-lang=c++",
                               "-inline", "auto", "-str", "reuse", *flags, "-c", "t.cpp", "-o", "t.o"],
                              cwd=tmp, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return Elf.parse((tmp / "t.o").read_bytes())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


ORIGINAL = """\
extern void use(const char*);
extern void usef(float);
static int twice(int x) { return x * 2; }
void a() { use("alpha"); usef(1.5f); }
static const char* names[] = { "n0", "n1" };
void b() { use(names[1]); usef(2.5f); use("bravo"); }
int c(int x) { use("charlie"); usef(3.5f); return twice(x); }
void d() { c(1); use("delta"); }
"""

SCRAMBLED = """\
extern void use(const char*);
extern void usef(float);
static int twice(int x) { return x * 2; }
// d comes first here
void d() { c(1); use("delta"); }
int c(int x) { use("charlie"); usef(3.5f); return twice(x); }
static const char* names[] = { "n0", "n1" };
void b() { use(names[1]); usef(2.5f); use("bravo"); }
void a() { use("alpha"); usef(1.5f); }
"""


@unittest.skipUnless(COMPILER.is_file() and WIBO.is_file(), "needs the project's compiler (./dk setup)")
class EndToEndTest(unittest.TestCase):
    def test_the_planned_order_compiles_to_the_original_bytes(self):
        target = compile_cpp(ORIGINAL)
        scrambled = SCRAMBLED.replace("void d() {", "int c(int);\nvoid d() {")     # it compiles as written
        ours = compile_cpp(scrambled, ("-sym", "on"))
        p = R.plan(scrambled, R.function_lines(ours), R.text_offsets(target))
        self.assertEqual(p.notes, [])
        rebuilt = compile_cpp(p.text)
        for sec in (".text", ".data", ".sdata", ".sdata2"):
            want, got = target.section(sec), rebuilt.section(sec)
            self.assertEqual(got and got.data, want and want.data, sec)
        order = lambda e: [s.name for s in sorted(e.symbols, key=lambda s: s.value) if s.type == 2 and s.shndx == 1]
        self.assertEqual(order(rebuilt), order(target))


@unittest.skipUnless(COMPILER.is_file() and WIBO.is_file(), "needs the project's compiler (./dk setup)")
class EmissionOrderTest(unittest.TestCase):
    def test_extern_c_functions_that_assign_a_struct_come_last_in_reverse(self):
        src = ("struct S { int a, b, c; };\n"
               'extern "C" void d2(S* o, S* i) { *o = *i; }\n'
               'extern "C" int p1(int x) { return x + 1; }\n'
               'extern "C" void d1(S* o, S* i) { *o = *i; }\n'
               'void d0(S* o, S* i) { *o = *i; }\n')
        elf = compile_cpp(src, ("-sym", "on"))
        order = [s.name for s in sorted(elf.symbols, key=lambda s: s.value) if s.type == 2 and s.shndx == 1]
        self.assertEqual(order, ["p1", "d0__FP1SP1S", "d1", "d2"])
        self.assertEqual(R.deferred_functions(elf, R.function_lines(elf)), {"d1", "d2"})


@unittest.skipUnless(COMPILER.is_file() and WIBO.is_file(), "needs the project's compiler (./dk setup)")
class WeakCopyPlacementTest(unittest.TestCase):
    HEAD = ("extern void use(int);\n"
            "struct B { virtual int v1() { return 1; } virtual int v2() { return 2; } virtual int v3() { return 3; } };\n"
            "static inline int fold(B* b, int k) { if (k != 0) return b->v2(); return 0; }\n")

    def order(self, body: str) -> list[str]:
        elf = compile_cpp(self.HEAD + body)
        return [s.name.split("__")[0] for s in sorted(elf.symbols, key=lambda s: s.value) if s.type == 2 and s.shndx == 1]

    def test_copies_follow_their_first_c_plus_plus_user_last_needed_first(self):
        self.assertEqual(self.order("void f1(B* b) { fold(b, 0); use(b->v3()); }\nvoid f2(B* b) { use(b->v1()); }\n"),
                         ["f1", "v3", "v2", "f2", "v1"])

    def test_after_an_extern_c_user_they_wait_for_the_next_c_plus_plus_declaration(self):
        self.assertEqual(self.order('extern "C" int c1(B* b) { return b->v1(); }\nextern "C" int c2(B* b) { return 4; }\n'
                                    "void x1(B* b) { use(5); }\n"), ["c1", "c2", "x1", "v1"])
        self.assertEqual(self.order('extern "C" int c1(B* b) { return b->v1(); }\nstruct Fwd;\n'
                                    'extern "C" int c2(B* b) { return 4; }\n'), ["c1", "v1", "c2"])

    def test_vtables_end_the_data_in_reverse_order_of_first_need(self):
        head = ("extern void use(int);\nstruct G { virtual ~G() { use(1); } };\n"
                "struct Str { Str() {} virtual ~Str() { use(2); } };\n")
        vt = lambda elf: [s.name for s in sorted(elf.symbols, key=lambda s: s.value)
                          if s.name.startswith("__vt__") and s.shndx == elf.section(".data").index]
        dtor_key = compile_cpp(head + "struct L : G { virtual ~L(); virtual void k(); Str s; };\n"
                               "L::~L() { use(4); }\nvoid L::k() { use(5); }\n", ("-RTTI", "on"))
        other_key = compile_cpp(head + "struct L : G { virtual void k(); virtual ~L(); Str s; };\n"
                                "L::~L() { use(4); }\nvoid L::k() { use(5); }\n", ("-RTTI", "on"))
        self.assertEqual(vt(dtor_key), ["__vt__1G", "__vt__3Str", "__vt__1L"])
        self.assertEqual(vt(other_key), ["__vt__1L", "__vt__1G", "__vt__3Str"])

    def test_code_under_if_0_needs_no_copy(self):
        self.assertEqual(self.order("void f1(B* b) { if (0) use(b->v2()); use(7); }\n"), ["f1"])


@unittest.skipUnless(COMPILER.is_file() and WIBO.is_file(), "needs the project's compiler (./dk setup)")
class LineTableTest(unittest.TestCase):
    def test_debug_lines_map_functions_to_their_definitions(self):
        src = "void b() {}\n\n// a\nint a(int x) {\n    return x + 1;\n}\ninline int h() { return 2; }\nint c() { return h(); }\n"
        tmp = Path(tempfile.mkdtemp(dir=REPO / "build"))
        try:
            (tmp / "t.cpp").write_text(src)
            proc = subprocess.run([str(WIBO), str(COMPILER), "-nodefaults", "-proc", "gekko", "-O4,p", "-lang=c++",
                                   "-sym", "on", "-c", "t.cpp", "-o", "t.o"], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            lines = R.function_lines(Elf.parse((tmp / "t.o").read_bytes()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(lines, {"b__Fv": 0, "a__Fi": 4, "c__Fv": 7})     # a body's first line, inside its item


if __name__ == "__main__":
    unittest.main()
