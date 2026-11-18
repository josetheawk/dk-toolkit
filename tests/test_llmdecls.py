import unittest

from tools.dk.platforms.gc_mwcc import llmdecls
from tools.dk.platforms.gc_mwcc.llmdecls import (FULL, SLIM, TOUCHED, Index, build_index, context, draft_types, gaps,
                                                 merge, render_type, statements, strip_comments)
from tools.dk.platforms.gc_mwcc.llmtry import Context, Ingredients, build_messages

ENTITY_H = """// entity.h
#ifndef ENTITY_H
#define ENTITY_H
class Archetype {
public:
    int mKind;
};
class Entity : public Archetype {
public:
    virtual void Update(float dt);
    int GetKind() const { return mKind; }   // short inline: body kept
    void Untouched(int a);
    int mFlags;
    struct Pos { float x, y; } mPos;
};
typedef struct Wheel Wheel;
struct Wheel {
    float mSpin;
    int mGrip;
};
struct Racer;
struct Racer {
    Wheel* mWheel;
    int m14;
};
extern "C" {
Racer* RacerAt(int i);
void SetGrip(Wheel* w, int grip);
}
template <class T> inline T Clamp(T v, T lo, T hi) { return v < lo ? lo : v > hi ? hi : v; }
extern Racer* gLeader;
#endif
"""

CAR_H = """#include "entity.h"
class Car : public Entity {
public:
    void Drive(Racer* r, int n);
    void Honk();
    int m20;
    Racer* mRacer;
};
class Horn {
public:
    void Blow(int n);
    const char* Name() { return "h{;}"; }
    int mVolume;
};
const char* HornName(const char* s = "x;{");
"""

UNIT = """#include "car.h"
static inline int Twice(int x) { return x + x; }
void Car::Honk() { }
void Car::Drive(Racer* r, int n) {
    m20 = n;
}
"""


def index() -> Index:
    span = (UNIT.index("void Car::Drive"), len(UNIT))
    return build_index([("include/entity.h", ENTITY_H), ("include/car.h", CAR_H)], "src/car.cpp", UNIT, span)


class FakeLayouts:
    """draftfields.Layouts' surface draft_types reads: by_name, size(ref), at(ref, offset)."""

    def __init__(self, structs: dict[str, tuple[int, list[int]]]):
        self.by_name = {n: i for i, n in enumerate(sorted(structs))}
        self.info = {self.by_name[n]: v for n, v in structs.items()}

    def size(self, ref):
        return self.info[ref][0]

    def at(self, ref, off):
        return ("m", None) if off in self.info[ref][1] else None


class ParseTest(unittest.TestCase):
    def test_statements_and_index(self):
        s = strip_comments(ENTITY_H)
        self.assertNotIn("body kept", s)
        self.assertEqual(len(s), len(ENTITY_H))
        kinds = [s[a:b].split()[0] for a, b in statements(s)]
        self.assertIn("class", kinds)
        ix = index()
        self.assertEqual(ix.types["Entity"].bases, ["Archetype"])
        self.assertEqual(ix.types["Car"].bases, ["Entity"])
        self.assertIn("{", ix.types["Wheel"].text)          # the body beats `typedef struct Wheel Wheel;`
        self.assertIn("{", ix.types["Racer"].text)          # and a forward declaration
        self.assertEqual(ix.funcs["RacerAt"], ("include/entity.h", "Racer* RacerAt(int i);"))   # inside extern "C"
        self.assertTrue(ix.inlines["Clamp"][1].startswith("template <class T> inline T Clamp("))   # a function template
        self.assertEqual(ix.globals["gLeader"], ("include/entity.h", "extern Racer* gLeader;"))
        self.assertTrue(ix.inlines["Twice"][2])             # the unit's own inline helper
        self.assertNotIn("Car::Drive", ix.funcs)            # out-of-class definitions are not prototypes
        self.assertIn("Twice", ix.unit_calls | {"Twice"})

    def test_render_type_modes(self):
        ix = index()
        full = render_type(ix.types["Entity"], {"Update"}, FULL)
        self.assertIn("int mFlags;", full)
        self.assertIn("virtual void Update(float dt);", full)
        self.assertIn("{ return mKind; }", full)
        self.assertNotIn("Untouched", full)                 # neither virtual, inline nor called
        self.assertNotIn("//", full)
        touched = render_type(ix.types["Entity"], {"mPos"}, TOUCHED)
        self.assertNotIn("mFlags", touched)
        self.assertIn("mPos", touched)
        self.assertIn("other data members not shown", touched)
        slim = render_type(ix.types["Horn"], {"Blow"}, SLIM)
        self.assertNotIn("mVolume", slim)
        self.assertIn("void Blow(int n);", slim)
        self.assertIn('const char* Name() { return "h{;}"; }', slim)    # literals parsed blanked, shown as written
        self.assertEqual(ix.funcs["HornName"][1], 'const char* HornName(const char* s = "x;{");')
        self.assertEqual(slim.count("\n\n"), 0)             # a block never holds a blank line (merge splits on them)


class ContextTest(unittest.TestCase):
    CALLS = ["RacerAt", "Blow__4HornFi", "SetGrip"]
    REFS = ["gLeader"]

    def build(self, budget=llmdecls.CONTEXT_BUDGET, code="this->m20 = arg1;"):
        return context(index(), "Drive__3CarFP5Racer i".replace(" ", ""), code, self.CALLS, self.REFS,
                       {"m20", "RacerAt", "SetGrip", "gLeader"}, budget)

    def test_owner_bases_callees_globals(self):
        ctx = self.build()
        for want in ("class Car : public Entity", "class Entity : public Archetype", "class Archetype",   # bases
                     "struct Racer {",                     # the parameter's type, a global's and a callee's result
                     "struct Wheel {",                     # a callee's parameter type (a local passed to it)
                     "Racer* RacerAt(int i);", "void SetGrip(Wheel* w, int grip);", "extern Racer* gLeader;",
                     "class Horn", "void Blow(int n);",    # a called method's class, slim
                     "static inline int Twice(int x) { return x + x; }"):   # the unit's own inline helper
            self.assertIn(want, ctx)
        self.assertNotIn("mVolume", ctx)
        self.assertNotIn("m20 = n", ctx)                    # never the target's body
        self.assertTrue(ctx.startswith("// include/entity.h\nRacer* RacerAt"))   # prototypes first: the cheapest
        for block in ctx.split("\n\n"):
            self.assertTrue(block.startswith("// "))
        self.assertEqual(ctx, self.build())                 # deterministic

    def test_budget(self):
        small = self.build(budget=400)
        self.assertLessEqual(len(small), 400)
        self.assertIn("RacerAt", small)
        self.assertEqual(self.build(budget=0), "")

    def test_draft_types(self):
        lay = FakeLayouts({"Racer": (0x20, [0x0, 0x14]), "Big": (0x100, [0x14, 0x30]), "Tiny": (0x8, [0x0, 0x4])})
        self.assertEqual(draft_types("x = temp_r3->unk14; y = temp_r3->unk0;", lay), ["Racer"])
        self.assertEqual(draft_types("x = p->unk14;", lay), ["Racer", "Big"])      # smaller first
        self.assertEqual(draft_types("x = p->unk0;", lay), [])                      # offset 0 alone says nothing
        self.assertEqual(draft_types("x = this->unk14;", lay), [])                  # `this` is typed already
        self.assertEqual(draft_types("x = p->unk14;", None), [])


class MergeTest(unittest.TestCase):
    def test_merge_drops_what_decls_define(self):
        ctx = ("// include/a.h\nint Foo(int a);\nextern int gBar;\n\n// include/a.h\nstruct Wheel {\n    int m0;\n};"
               "\n\n// include/b.h\nclass Horn\n{\n};")
        decls = '#include "a.h"\nstruct Wheel { int m0; };\nint Foo(int a);\n'
        out = merge(ctx, decls)
        self.assertNotIn("struct Wheel {\n", out)
        self.assertIn("extern int gBar;", out)
        self.assertEqual(out.count("int Foo(int a);"), 1)
        self.assertTrue(out.endswith(decls))
        self.assertIn("class Horn", out)
        self.assertEqual(merge("", decls), decls)

    def test_build_messages_merges_context(self):
        ctx = Context("Drive__3CarFP5Racer i".replace(" ", ""), "asm", '#include "car.h"\n', "", "", "")
        v6 = build_messages(ctx, "", 0.0, [], Ingredients(), True)
        ctx.context = "// include/entity.h\nRacer* RacerAt(int i);"
        v7 = build_messages(ctx, "", 0.0, [], Ingredients(), True)
        self.assertEqual(v6[0], v7[0])                      # the same rules
        head6 = v6[1]["content"].split("## Declarations")[0]
        self.assertEqual(head6, v7[1]["content"].split("## Declarations")[0])     # the same prefix up to decls
        self.assertIn("Racer* RacerAt(int i);\n\n#include \"car.h\"", v7[1]["content"])

    def test_export_render_uses_the_views_context(self):
        from tools.dk.platforms.gc_mwcc.llmdata import render
        rec = {"function": "fn_80001000", "asm": "asm", "decls": '#include "x.h"\n', "m2c": "", "exact": "void f() {}",
               "wrong": "void f() { g(); }", "wrong_percent": 50.0, "rows": []}
        view = {"m2c": "", "decls": '#include "x.h"\n', "notes": "", "context": "// include/x.h\nvoid g();"}
        repair = render(rec, view=view)["messages"][1]["content"]
        self.assertIn("void g();\n\n#include \"x.h\"", repair)
        direct = render({**rec, "wrong": ""}, view=view)["messages"][1]["content"]
        self.assertIn("void g();\n\n#include \"x.h\"", direct)
        self.assertNotIn("void g();", render({**rec, "wrong": ""}, view={**view, "context": ""})["messages"][1]["content"])


class GapsTest(unittest.TestCase):
    def test_gaps(self):
        answer = 'void f(Car* c) { c->m14 = c->mWheel->m8; Helper(c); fn_80001234(); /* c->mGone */ Log("a(b)"); }'
        g = gaps(answer, "draft: c->unk8 Log", "x = c->unk8;", {"f"})
        self.assertEqual(g["members"], ["m14", "m8", "mWheel"])
        self.assertEqual(g["field"], ["m14"])               # m8 is predictable from the draft's unk8
        self.assertEqual(g["calls"], ["Helper", "fn_80001234"])
        self.assertEqual(g["helper"], ["Helper"])


if __name__ == "__main__":
    unittest.main()
