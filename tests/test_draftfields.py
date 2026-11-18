import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc.draftfields import Layouts, clean_structors, name_fields
from tools.dk.platforms.gc_mwcc.elf import Elf
from tools.dk.platforms.gc_mwcc.llmtry import unit_notes
from tools.dk.platforms.gc_mwcc.retrieval import Index, grams, normalized, render, similarity

REPO = Path(__file__).resolve().parents[1]

CLASSES = """struct Vec { float x, y; };
struct Base { int m0; int m4; };
class Car : public Base {
public:
    int m8;
    Vec m0C;
    Car* mNext;
    unsigned int flags : 3;
    int mArr[4];
    void f(Car* other);
};
void Car::f(Car* other) { m8 = other->m4 + (int)m0C.y + mArr[1] + flags; mNext = other; }
"""


class FieldNameTest(unittest.TestCase):
    """m2c's `p->unkNN` named from the real compiler's DWARF layouts (skipped without the tools)."""

    @classmethod
    def setUpClass(cls):
        mwcc = REPO / "build" / "compilers" / "GC" / "1.3.2" / "mwcceppc.exe"
        wibo = REPO / "build" / "tools" / "wibo"
        if not mwcc.exists() or not wibo.exists():
            raise unittest.SkipTest("compiler not installed (./dk setup)")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "t.cpp").write_text(CLASSES)
            cmd = [str(wibo), str(mwcc), "-nodefaults", "-proc", "gekko", "-O4,p", "-fp", "hardware", "-lang=c++",
                   "-sym", "on", "-c", "t.cpp", "-o", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            cls.lay = Layouts(Elf.parse((Path(tmp) / "t.o").read_bytes()))

    def test_offsets_become_members(self):
        draft = ("void Car::f(Car *arg0) {\n    this->unk8 = arg0->unk4 + (s32) this->unk10 + this->unk20;\n"
                 "    this->unk14->unk8 = arg0->unkC;\n    this->unk18 = 0;\n    temp_r3->unk4 = this->unk1C;\n}\n")
        got = name_fields(draft, self.lay, "Car", ["Car*"])
        self.assertEqual(got, "void Car::f(Car *arg0) {\n    this->m8 = arg0->m4 + (s32) this->m0C.y + this->unk20;\n"
                              "    this->mNext->m8 = arg0->m0C.x;\n    this->unk18 = 0;\n    temp_r3->unk4 = this->mArr[0];\n}\n")

    def test_unknown_bases_stay(self):
        draft = "void g(void) {\n    lbl_80401000->unk8 = 1;\n    lbl_80402000->unk8 = 2;\n}\n"
        got = name_fields(draft, self.lay, None, [], "extern Car* lbl_80401000;\nextern int* lbl_80402000;\n")
        self.assertEqual(got, "void g(void) {\n    lbl_80401000->m8 = 1;\n    lbl_80402000->unk8 = 2;\n}\n")


class StructorTest(unittest.TestCase):
    def test_constructor(self):
        draft = ("XImage::XImage(Wnd *arg0) {\n    XWidget::XWidget((XWidget *) this, arg0);\n"
                 "    this->vtable0 = &__vt__6XImage;\n    this->vtable0 = (struct __vt__6XImage *) &__vt__9XImageBox;\n"
                 "    this->m114 = 0;\n    return this;\n}\n")
        self.assertEqual(clean_structors(draft, "__ct__6XImageFP3Wnd"),
                         "XImage::XImage(Wnd *arg0) : XWidget(arg0) {\n    this->m114 = 0;\n}\n")
        self.assertEqual(clean_structors(draft, "fn_80401000"), draft)          # only constructors and destructors

    def test_destructor(self):
        draft = ("Cam::~Cam(short destroyFlag) {\n    void *temp_r3;\n\n    if (this != 0) {\n"
                 "        this->vtable0 = &__vt__3Cam;\n        temp_r3 = this->unk4;\n        if (temp_r3 != 0) {\n"
                 "            operator delete[](temp_r3);\n        }\n        if (destroyFlag > 0) {\n"
                 "            operator delete(this);\n        }\n    }\n    return this;\n}\n")
        self.assertEqual(clean_structors(draft, "__dt__3CamFv"),
                         "Cam::~Cam() {\n    void *temp_r3;\n\n    temp_r3 = this->unk4;\n    if (temp_r3 != 0) {\n"
                         "        operator delete[](temp_r3);\n    }\n}\n")


class NotesTest(unittest.TestCase):
    class Entry:
        def __init__(self, calls, refs=()):
            self.calls, self.refs = list(calls), list(refs)

    def test_only_what_the_function_needs(self):
        self.assertEqual(unit_notes(self.Entry(["fn_80401000"]), "x = 1;", ""), "")
        notes = unit_notes(self.Entry(["__nw__FUlPCci"], ["__vt__3Car", "lbl_80670370"]), "sp8 = &__vt__3Car;",
                           "extern unsigned char lbl_80670370[12];\nextern Car* lbl_80401000;\n")
        self.assertIn("new (\"name\", heap) T(args)", notes)
        self.assertIn("never assign `__vt__...`", notes)
        self.assertIn("`lbl_80670370` is declared `unsigned char lbl_80670370[12]`: an array", notes)
        self.assertNotIn("lbl_80401000", notes)


def asm(*insns: str) -> str:
    return "\n".join(f"/* {0x80401000 + 4 * i:08X} {4 * i:08X}  00 00 00 00 */\t{x}" for i, x in enumerate(insns))


class RetrievalTest(unittest.TestCase):
    A = asm("stwu r1, -0x10(r1)", "lwz r3, 0x8(r3)", "addi r3, r3, 0x1", "stw r3, 0x8(r4)", "blr")
    B = asm("stwu r1, -0x20(r1)", "lwz r5, 0x10(r4)", "addi r5, r5, 0x7", "stw r5, 0xc(r6)", "blr")
    C = asm("fmuls f1, f1, f2", "fadds f1, f1, f3", "frsp f1, f1", "blr")

    def test_normalized_ignores_registers_and_numbers(self):
        self.assertEqual(normalized(self.A), normalized(self.B))
        self.assertEqual(normalized(self.C)[0], "fmuls r, r, r")
        self.assertGreater(similarity(grams(normalized(self.A)), grams(normalized(self.B)), 20, 20),
                           similarity(grams(normalized(self.A)), grams(normalized(self.C)), 20, 16))

    def test_top_ranks_excludes_and_fits_the_budget(self):
        recs = [{"unit": "u", "function": f, "size": 20, "compiler": comp, "asm": "x" * n, "source": f"int {f}();",
                 "tokens": normalized(a)} for f, a, comp, n in
                (("same", self.B, "GC/1.3.2", 100), ("float", self.C, "GC/1.3.2", 100), ("big", self.B, "GC/1.3.2", 5000),
                 ("other_compiler", self.B, "GC/1.3", 100), ("query", self.A, "GC/1.3.2", 100))]
        ix = Index(recs)
        got = [x.function for x in ix.top(self.A, 20, "GC/1.3.2", {"query"}, k=3, budget=1000)]
        self.assertEqual(got, ["same", "float", "other_compiler"])           # own compiler first; too big skipped
        self.assertIn("### same\n```\n", render(ix.top(self.A, 20, "GC/1.3.2", {"query"}, k=1)))
        self.assertEqual(render([]), "")


if __name__ == "__main__":
    unittest.main()
