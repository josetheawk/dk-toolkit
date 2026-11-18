import unittest

from tools.dk.core import cbody
from tools.dk.core.cbody import apply_edits, function_at_line, functions, parse_body, parse_expression, statements

SRC = """\
#include "x.h"
struct Vec { float x, y; };
typedef int Count;

static inline float Abs(float v) { return v < 0.0f ? -v : v; }

class Car {
public:
    Car(int n) : mN(n), mK(0) { mFlags = 0; }
    int Get() const { return mN; }
    int mN, mK, mFlags;
};

int Car::Update(Vec* v, int n) {
    // a comment with { braces }
    Count c = 0, *p = &c;
    float s[4] __attribute__((aligned(32)));
    for (int i = 0; i < n; i++) {
        if (v[i].x > 0.5f) c += 2; else if (!v) break;
        else { c--; }
    }
    while (c = Next(c)) { }
    do { c >>= 1; } while (c > 0);
    switch (n) {
    case 1:
    case 2 + 1:
        c = (Count)n * 2;
        break;
    default:
        goto out;
    }
out:
    const char* name = "a" "b";
    Vec& e = v[n];
    e.x = e.y = 0.0f;
    return c ? (int)s[0] : sizeof(Vec);
}

bool operator|(const Vec& a, const Vec& b) { return a.x < b.x; }
"""


def body(src: str = SRC, name: str = "Car::Update"):
    f = next(f for f in functions(src) if f.name == name)
    return parse_body(src, f.open, cbody.type_names(src))


class TokenTest(unittest.TestCase):
    def test_comments_strings_and_directives(self):
        toks = cbody.tokens('#define A 1 \\\n  2\nx = "a;b" /* c */ + \'}\'; // d\n')
        self.assertEqual([(t.kind, t.text) for t in toks],
                         [("pp", "#define A 1 \\\n  2"), ("id", "x"), ("op", "="), ("str", '"a;b"'), ("op", "+"),
                          ("chr", "'}'"), ("op", ";")])

    def test_numbers_and_longest_operators(self):
        toks = cbody.tokens("a<<=0x1Fu>>=1.5e-3f->*b")
        self.assertEqual([t.text for t in toks], ["a", "<<=", "0x1Fu", ">>=", "1.5e-3f", "->*", "b"])


class FunctionTest(unittest.TestCase):
    def test_definitions_in_classes_and_at_top_level(self):
        self.assertEqual([f.name for f in functions(SRC)], ["Abs", "Car", "Get", "Car::Update", "operator|"])

    def test_function_at_a_line(self):
        lines = SRC.splitlines()
        inside = next(i for i, ln in enumerate(lines) if "goto out" in ln)
        self.assertEqual(function_at_line(SRC, inside).name, "Car::Update")
        head = next(i for i, ln in enumerate(lines) if ln.startswith("int Car::Update"))
        self.assertEqual(function_at_line(SRC, head).name, "Car::Update")
        ctor = next(i for i, ln in enumerate(lines) if "Car(int n)" in ln)
        self.assertEqual(function_at_line(SRC, ctor).name, "Car")


class StatementTest(unittest.TestCase):
    def test_statement_kinds_in_order(self):
        b = body()
        self.assertEqual([s.kind for s in b.kids],
                         ["decl", "decl", "for", "while", "do", "switch", "label", "decl", "decl", "expr", "return"])
        self.assertNotIn("other", [s.kind for s in statements(b)])

    def test_declarations(self):
        b = body()
        d = b.kids[0].decl
        self.assertEqual(d.base_type(SRC), "Count")
        self.assertEqual([(x.name, x.ptr, x.init.text(SRC)) for x in d.declarators], [("c", "", "0"), ("p", "*", "&c")])
        arr = b.kids[1].decl.declarators[0]
        self.assertEqual((arr.name, arr.array), ("s", "[4]"))
        ref = b.kids[8].decl
        self.assertEqual((ref.base_type(SRC), ref.declarators[0].ptr, ref.declarators[0].init.text(SRC)), ("Vec", "&", "v[n]"))
        self.assertEqual(b.kids[7].decl.declarators[0].init.kind, "str")

    def test_for_header_and_else_if(self):
        loop = body().kids[2]
        self.assertEqual(loop.init.kind, "decl")
        self.assertEqual(loop.init.decl.declarators[0].name, "i")
        self.assertEqual((loop.cond.text(SRC), loop.step.text(SRC)), ("i < n", "i++"))
        self.assertEqual(SRC[loop.head[0]:loop.head[1]], "int i = 0; i < n; i++")
        first = loop.kids[0].kids[0]
        self.assertEqual(first.kind, "if")
        self.assertEqual(len(first.kids), 2)
        self.assertEqual(first.kids[1].kind, "if")              # else if
        self.assertEqual(first.kids[1].kids[1].kind, "block")

    def test_switch_cases_and_labels(self):
        sw = body().kids[5]
        kinds = [s.kind for s in sw.kids[0].kids]
        self.assertEqual(kinds, ["case", "case", "expr", "break", "default", "goto"])
        self.assertEqual(sw.kids[0].kids[1].expr.text(SRC), "2 + 1")
        self.assertEqual(body().kids[6].label, "out")

    def test_while_with_an_assignment_and_do_while(self):
        b = body()
        self.assertEqual((b.kids[3].expr.kind, b.kids[3].expr.op), ("assign", "="))
        self.assertEqual(b.kids[4].expr.text(SRC), "c > 0")

    def test_typeof_declarations(self):
        src = "void f() { __typeof__(((u8*)b)[i] * 2)& e = g; __typeof__(x) t = x; }"
        b = parse_body(src, src.index("{"))
        self.assertEqual([s.kind for s in b.kids], ["decl", "decl"])
        d = b.kids[0].decl
        self.assertEqual((d.base_type(src), d.declarators[0].ptr, d.declarators[0].name), ("__typeof__(((u8*)b)[i] * 2)", "&", "e"))

    def test_unreadable_statement_is_kept_whole(self):
        src = "void f() { a = 1; ((void (*)(int))g)(2); b = 3; }"
        b = parse_body(src, src.index("{"))
        self.assertEqual([s.kind for s in b.kids], ["expr", "other", "expr"])
        self.assertEqual(b.kids[1].text(src), "((void (*)(int))g)(2);")


class ExpressionTest(unittest.TestCase):
    def tree(self, s: str, types=frozenset()):
        def show(e):
            if not e.kids:
                return s[e.start:e.end]
            return f"{e.kind}{e.op}(" + ", ".join(show(k) for k in e.kids) + ")"
        return show(parse_expression(s, types))

    def test_precedence_and_associativity(self):
        self.assertEqual(self.tree("a + b * c - d"), "binary-(binary+(a, binary*(b, c)), d)")
        self.assertEqual(self.tree("a = b = c"), "assign=(a, assign=(b, c))")
        self.assertEqual(self.tree("!a || b && c"), "binary||(unary!(a), binary&&(b, c))")
        self.assertEqual(self.tree("x = c ? a : b"), "assign=(x, ternary?(c, a, b))")

    def test_casts_versus_parentheses(self):
        self.assertEqual(self.tree("(float)n * 2"), "binary*(cast()(n), 2)")
        self.assertEqual(self.tree("(a) - b"), "binary-(paren()(a), b)")
        self.assertEqual(self.tree("(T*)p->q", {"T"}), "cast()(member->(p))")
        self.assertEqual(self.tree("(Count)-1", {"Count"}), "cast()(unary-(1))")
        self.assertEqual(self.tree("static_cast<int>(f)"), "caststatic_cast(f)")

    def test_postfix_calls_members_and_types_as_arguments(self):
        self.assertEqual(self.tree("p->a[i].f(x, y)"), "call()(member.(index[](member->(p), i)), x, y)")
        self.assertEqual(self.tree("va_arg(ap, unsigned long)"), "call()(va_arg, ap, unsigned long)")
        self.assertEqual(self.tree("sizeof(int) + sizeof x"), "binary+(sizeof(int), sizeofsizeof(x))")
        self.assertEqual(self.tree("A::B::c(1)"), "call()(A::B::c, 1)")

    def test_new_and_delete(self):
        e = parse_expression('new (0, "Car") Car(parent)')
        self.assertEqual((e.kind, e.op, len(e.kids)), ("unary", "new", 2))
        self.assertEqual(parse_expression("delete[] p").op, "delete")

    def test_spans_cover_the_source(self):
        s = "  x->m[i] += (int)(a * 0.5f) "
        e = parse_expression(s)
        self.assertEqual(s[e.start:e.end], "x->m[i] += (int)(a * 0.5f)")
        self.assertEqual(s[e.kids[1].start:e.kids[1].end], "(int)(a * 0.5f)")


class EditTest(unittest.TestCase):
    def test_a_rewrite_replaces_one_node_and_keeps_everything_else(self):
        b = body()
        ret = b.kids[-1]
        cond = ret.expr.kids[0]
        new = apply_edits(SRC, [(cond.start, cond.end, "c != 0")])
        self.assertIn("return c != 0 ? (int)s[0] : sizeof(Vec);", new)
        self.assertEqual(new.replace("c != 0 ?", "c ?"), SRC)

    def test_overlapping_edits_are_refused(self):
        with self.assertRaises(ValueError):
            apply_edits("abcdef", [(0, 3, "x"), (2, 4, "y")])


if __name__ == "__main__":
    unittest.main()
