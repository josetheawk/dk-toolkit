import re
import unittest

from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc.rewrites import BY_NAME, Site, candidates
from tools.dk.platforms.gc_mwcc.shaperules import accessors


def site(body: str, cplusplus: bool = True, head: str = "void f(int a, int b, float x, S* p)", acc=None) -> Site:
    src = head + " {\n" + body + "}\n"
    return Site(src, cbody.parse_body(src, src.index("{")), None, cplusplus, accessors=acc)


def outs(rule: str, body: str, **kw) -> list[str]:
    s = site(body, **kw)
    return [new[new.index("{") + 2:-3] for _, _, new in candidates(s, [BY_NAME[rule]], groups=False)]


def norm(t: str) -> str:
    return " ".join(t.split())


def has(got: list[str], want: str) -> bool:
    return norm(want) in [norm(g) for g in got]


class ElseReturnTest(unittest.TestCase):
    def test_jump_then_rest_both_ways(self):
        got = outs("else-return", "    if (a) {\n        return;\n    }\n    b = 1;\n    p->m = 2;\n")
        self.assertTrue(has(got, "if (!a) { b = 1; p->m = 2; } else { return; }"), got)
        self.assertTrue(has(got, "if (a) { return; } else { b = 1; p->m = 2; }"), got)
        back = outs("else-return", "    if (!a) {\n        b = 1;\n    } else {\n        return;\n    }\n")
        self.assertTrue(has(back, "if (a) { return; } b = 1;"), back)

    def test_dead_else_and_empty_then(self):
        got = outs("else-return", "    if (a) {\n        b = 2;\n        return;\n    } else {\n        b = 1;\n    }\n")
        self.assertTrue(has(got, "if (a) { b = 2; return; } b = 1;"), got)
        got = outs("else-return", "    if (a != 0) {\n    } else {\n        return;\n    }\n")
        self.assertTrue(has(got, "if (a == 0) { return; }"), got)
        inv = outs("else-return", "    if (a == 0) b = 1;\n")
        self.assertTrue(has(inv, "if (a != 0) { } else { b = 1; }"), inv)

    def test_never_moves_case_labels(self):
        got = outs("else-return", "    switch (a) {\n    case 1:\n        if (b) break;\n        b = 2;\n    case 2:\n        b = 3;\n    }\n")
        self.assertFalse(any("case 2" in g.split("else")[0] and "if (!b)" in g for g in got))


class NameValueTest(unittest.TestCase):
    def test_names_a_call_in_a_condition_and_ahead_of_declarations(self):
        got = outs("name-value", "    int r = 4;\n    if (g(a) != 0) {\n        r = 0;\n    }\n")
        self.assertTrue(has(got, "int r = 4; __typeof__(g(a)) t0 = g(a); if (t0 != 0) { r = 0; }"), got)
        self.assertTrue(has(got, "__typeof__(g(a)) t0 = g(a); int r = 4; if (t0 != 0) { r = 0; }"), got)

    def test_names_a_load_argument(self):
        got = outs("name-value", "    x = h(p->m * x);\n")
        self.assertTrue(has(got, "__typeof__(p->m) t0 = p->m; x = h(t0 * x);"), got)

    def test_folds_a_single_use_local(self):
        got = outs("name-value", "    int k = g(a);\n    int r = 4;\n    if (k != 0) r = 0;\n")
        self.assertTrue(has(got, "int r = 4; if (g(a) != 0) r = 0;"), got)

    def test_never_names_the_right_of_an_and(self):
        got = outs("name-value", "    if (a && g(b)) b = 1;\n")
        self.assertFalse(any("t0" in g for g in got))


class InlineLocalTest(unittest.TestCase):
    def test_writes_a_pointer_out_and_names_it_back(self):
        got = outs("inline-local", "    T* e = p->m;\n    e[a].x = 1;\n    e[a].y = 2;\n")
        self.assertTrue(has(got, "p->m[a].x = 1; p->m[a].y = 2;"), got)
        back = outs("inline-local", "    p->m[a].x = 1;\n    p->m[a].y = 2;\n")
        self.assertTrue(any("t0 = p->m[a]" in g or "t0 = p->m;" in g for g in back), back)

    def test_recalls_a_call_with_pure_arguments(self):
        got = outs("inline-local", "    T* q = g(a, b);\n    q->f();\n    q->h();\n")
        self.assertTrue(has(got, "g(a, b)->f(); g(a, b)->h();"), got)

    def test_not_when_the_value_changes(self):
        self.assertEqual(outs("inline-local", "    T* q = p->m;\n    q->f();\n    p->m = 0;\n    q->h();\n"), [])

    def fold_and_back(self, rule, body, mark, decl, name, **kw):
        """The fold of `body`'s local `decl` (its type and name), then the naming moves of the result,
        both written with the new local as `@ t0` (shape-synth compares up to the new local's type and name)."""
        start = site(body, **kw).src
        folds = [new for _, label, new in candidates(site(body, **kw), [BY_NAME[rule]], groups=False) if mark in label]
        self.assertTrue(folds)
        s = site(folds[0][folds[0].index("{") + 2:-2], **kw)
        want = norm(re.sub(rf"\b{name}\b", "t0", start.replace(decl, "@ " + name)))
        back = [norm(re.sub(r"__typeof__\([^;]*?\) t0", "@ t0", new)) for _, _, new in candidates(s, groups=False)]
        return want, back

    def test_naming_restores_the_declarations_place(self):
        body = "    T* q = g(a);\n    b = 1;\n    x = 2.0f;\n    q->f();\n    q->h();\n"
        want, back = self.fold_and_back("inline-local", body, "written out", "T* q", "q")
        self.assertIn(want, back)
        got = outs("inline-local", "    b = 1;\n    g(a)->f();\n    g(a)->h();\n")
        self.assertTrue(has(got, "__typeof__(g(a)) t0 = g(a); b = 1; t0->f(); t0->h();"), got)

    def test_naming_never_crosses_a_write_of_what_the_value_reads(self):
        got = outs("inline-local", "    a = 1;\n    b = 2;\n    g(a)->f();\n    g(a)->h();\n")
        self.assertFalse(any(norm(g).startswith("__typeof__") for g in got), got)
        self.assertTrue(any(norm(g).startswith("a = 1; __typeof__") for g in got), got)

    def test_c_names_at_the_top_initialized(self):
        body = "    int k = 0;\n    T* q = g(a);\n    int r = 1;\n    q->f();\n    q->h();\n"
        want, back = self.fold_and_back("inline-local", body, "written out", "T* q", "q", cplusplus=False)
        self.assertIn(want, back)


class ValueFormTest(unittest.TestCase):
    """T37 (a): the naming moves name the value forms a fold writes out, so the fold reverses."""

    def round_trip(self, rule: str, body: str, folded: str, named: str):
        got = outs(rule, body)
        self.assertTrue(has(got, folded), got)
        back = outs(rule, "    " + folded + "\n")
        self.assertTrue(has(back, named), back)

    def test_binary_written_out_and_named_back(self):
        self.round_trip("inline-local", "    float u0 = a * x;\n    g(u0);\n    h(u0 + 1);\n",
                        "g((a * x)); h((a * x) + 1);", "__typeof__(a * x) t0 = a * x; g(t0); h(t0 + 1);")

    def test_pointer_cast_keeps_its_parentheses(self):
        self.round_trip("inline-local", "    T* q = (T*)p;\n    q->a = 1;\n    q->b = 2;\n",
                        "((T*)p)->a = 1; ((T*)p)->b = 2;", "__typeof__((T*)p) t0 = (T*)p; t0->a = 1; t0->b = 2;")

    def test_address_named_back_from_its_member_uses(self):
        self.round_trip("inline-local", "    T* r = &s;\n    r->a = 1;\n    g(r);\n",
                        "s.a = 1; g(&s);", "__typeof__(&s) t0 = &s; t0->a = 1; g(t0);")

    def test_name_value_parenthesized_forms(self):
        self.round_trip("name-value", "    float k = a * x;\n    g(k);\n", "g((a * x));",
                        "__typeof__(a * x) t0 = a * x; g(t0);")
        self.round_trip("name-value", "    T* q = &s;\n    q->a = 1;\n", "(&s)->a = 1;",
                        "__typeof__(&s) t0 = &s; t0->a = 1;")
        got = outs("name-value", "    int y = (-a);\n")
        self.assertTrue(has(got, "__typeof__(-a) t0 = -a; int y = t0;"), got)

    def test_not_a_value_that_changes_or_a_constant(self):
        got = outs("inline-local", "    g(a * x);\n    a = 2;\n    h(a * x);\n")
        self.assertFalse(any("t0" in g for g in got), got)
        got = outs("inline-local", "    g(-1);\n    h(-1);\n")
        self.assertFalse(any("t0" in g for g in got), got)
        got = outs("name-value", "    g((a + b));\n    (a + b);\n")
        self.assertFalse(any("t0;" in g.split("g(")[-1] for g in got), got)

    def test_address_not_named_across_a_write_of_its_pointer(self):
        got = outs("inline-local", "    p->m.a = 1;\n    p = 0;\n    p->m.b = 2;\n")
        self.assertFalse(any("&p->m" in g for g in got), got)


class NameValuePlaceTest(unittest.TestCase):
    def test_fold_restored_between_declarations(self):
        body = "    int r = 4;\n    int k = g(a);\n    int s = 5;\n    if (k != 0) r = s;\n"
        want, back = InlineLocalTest.fold_and_back(self, "name-value", body, "folded", "int k", "k")
        self.assertIn(want, back)


class LocalReuseTest(unittest.TestCase):
    def test_dead_parameter_reused_and_back(self):
        got = outs("local-reuse", "    int sh;\n    b = 1;\n    sh = a << 16;\n    p->m = sh;\n")
        self.assertTrue(has(got, "b = 1; a = a << 16; p->m = a;"), got)
        back = outs("local-reuse", "    b = 1;\n    a = a << 16;\n    p->m = a;\n")
        self.assertTrue(has(back, "b = 1; __typeof__(a) t0 = a << 16; p->m = t0;"), back)

    def test_not_when_the_parameter_is_read_later_or_in_a_loop(self):
        got = outs("local-reuse", "    int sh = a << 16;\n    p->m = sh + a;\n")
        self.assertFalse(any(g.startswith("    a = a << 16") for g in got), got)
        got = outs("local-reuse", "    while (b) {\n        int sh = a + 1;\n        p->m = sh;\n    }\n")
        self.assertFalse(any("a = a + 1" in g for g in got))


class DefaultElseTest(unittest.TestCase):
    def test_default_moves_into_an_else(self):
        body = "    float s = 1.0f;\n    if (x > 0.0f) {\n        if (x < 1.0f) {\n            s = x;\n        }\n        p->f = s;\n    }\n"
        got = outs("default-else", body)
        self.assertTrue(has(got, "if (x > 0.0f) { float s; if (x < 1.0f) { s = x; } else { s = 1.0f; } p->f = s; }"), got)

    def test_ternary_and_back(self):
        got = outs("default-else", "    int f = 1;\n    if (a != 0) f = 3;\n    b = f;\n")
        self.assertTrue(has(got, "int f = a != 0 ? 3 : 1; b = f;"), got)
        back = outs("default-else", "    int f = a != 0 ? 3 : 1;\n    b = f;\n")
        self.assertTrue(has(back, "int f = 1; if (a != 0) { f = 3; } b = f;"), back)
        back = outs("default-else", "    if (a) {\n        b = a;\n    } else {\n        b = 0;\n    }\n")
        self.assertTrue(has(back, "b = 0; if (a) { b = a; }"), back)


class UnsignedCompareTest(unittest.TestCase):
    def test_literals_and_counters(self):
        got = outs("unsigned-compare", "    if (*p == '\\n') a = 0;\n")
        self.assertTrue(has(got, "if (*p == 10u) a = 0;"), got)
        self.assertTrue(has(outs("unsigned-compare", "    if (a < 5u) b = 0;\n"), "if (a < 5) b = 0;"))
        got = outs("unsigned-compare", "    int i;\n    for (i = 0; i < b; i++) p->m = i;\n")
        self.assertTrue(has(got, "unsigned int i; for (i = 0; i < b; i++) p->m = i;"), got)


class AccessorTest(unittest.TestCase):
    ACC = accessors(["struct W { T* Get(int i) { return (T*)&m4[i]; } int& At(int i) { return mV[i]; }\n"
                     "    T* Cop(int i) { return mC[i]; } };\nstruct H {\n    W* mW;\n};\nextern H* lbl_1;\n"])
    W = "void f(int a, int b, float x, W* p)"

    def test_accessors_with_their_class(self):
        self.assertEqual(dict(self.ACC), {"m4": [("Get", "ptr-cast", "W")], "mV": [("At", "ref", "W")],
                                          "mC": [("Cop", "value", "W")]})
        self.assertEqual(self.ACC.fields[("H", "mW")], {"W"})
        self.assertEqual(self.ACC.globals["lbl_1"], {"H"})

    def test_index_to_accessor_and_back(self):
        got = outs("accessor", "    x = p->m4[a].f;\n", acc=self.ACC, head=self.W)
        self.assertTrue(has(got, "x = p->Get(a)->f;"), got)
        got = outs("accessor", "    x = p->Get(a)->f;\n    b = p->mV[a];\n", acc=self.ACC, head=self.W)
        self.assertTrue(has(got, "x = p->m4[a].f; b = p->mV[a];"), got)
        self.assertTrue(has(got, "x = p->Get(a)->f; b = p->At(a);"), got)

    def test_only_the_accessors_class(self):
        self.assertEqual(outs("accessor", "    x = p->m4[a].f;\n", acc=self.ACC), [])      # p is an S*
        got = outs("accessor", "    x = lbl_1->mW->m4[a].f;\n", acc=self.ACC)
        self.assertTrue(has(got, "x = lbl_1->mW->Get(a)->f;"), got)
        got = outs("accessor", "    W* w = q();\n    x = w->m4[a].f;\n", acc=self.ACC)
        self.assertTrue(has(got, "W* w = q(); x = w->Get(a)->f;"), got)

    def test_value_accessor_only_read(self):
        got = outs("accessor", "    T* c = p->mC[a];\n    p->mC[b] = c;\n", acc=self.ACC, head=self.W)
        self.assertEqual([norm(g) for g in got], ["T* c = p->Cop(a); p->mC[b] = c;"])
        got = outs("accessor", "    g(p->Cop(a));\n", acc=self.ACC, head=self.W)
        self.assertTrue(has(got, "g(p->mC[a]);"), got)

    def test_no_index_without_accessors(self):
        self.assertEqual(outs("accessor", "    x = p->m4[a].f;\n"), [])


class SharedLabelsTest(unittest.TestCase):
    def test_merge_and_split_only_existing_labels(self):
        body = "    switch (a) {\n    case 1:\n        b = 2;\n        break;\n    case 3:\n        b = 2;\n        break;\n    }\n"
        got = outs("shared-labels", body)
        self.assertTrue(has(got, "switch (a) { case 1: case 3: b = 2; break; }"), got)
        back = outs("shared-labels", "    switch (a) {\n    case 1:\n    case 3:\n        b = 2;\n        break;\n    }\n")
        self.assertTrue(has(back, "switch (a) { case 1: b = 2; break; case 3: b = 2; break; }"), back)
        for g in got + back:                            # never a case value the switch did not have
            self.assertTrue(set(re.findall(r"case (\d+)", g)) <= {"1", "3"})


if __name__ == "__main__":
    unittest.main()


class CachedChainTest(unittest.TestCase):
    def test_first_use_read_through_the_chain_and_back(self):
        body = "    S* s = p->q;\n    if (s->m == 0) {\n        b = 1;\n    }\n    a = s->n;\n"
        got = outs("cached-chain", body)
        self.assertTrue(has(got, "if (p->q->m == 0) { b = 1; } S* s = p->q; a = s->n;"), got)
        back = outs("cached-chain", "    if (p->q->m == 0) {\n        b = 1;\n    }\n    S* s = p->q;\n    a = s->n;\n")
        self.assertTrue(has(back, "S* s = p->q; if (s->m == 0) { b = 1; } a = s->n;"), back)

    def test_skips_declarations_between(self):
        body = "    S* s = p->q;\n    int lim = 4;\n    if (s->m == 0) {\n        lim = 2;\n    }\n    a = s->n + lim;\n"
        got = outs("cached-chain", body)
        self.assertTrue(has(got, "int lim = 4; if (p->q->m == 0) { lim = 2; } S* s = p->q; a = s->n + lim;"), got)
        back = outs("cached-chain", "    int lim = 4;\n    if (p->q->m == 0) {\n        lim = 2;\n    }\n    S* s = p->q;\n    a = s->n + lim;\n")
        self.assertTrue(has(back, "int lim = 4; S* s = p->q; if (s->m == 0) { lim = 2; } a = s->n + lim;"), back)

    def test_needs_a_later_use_and_a_stable_chain(self):
        got = outs("cached-chain", "    S* s = p->q;\n    if (s->m == 0) {\n        b = 1;\n    }\n")
        self.assertFalse(any("p->q->m" in g for g in got), got)
        got = outs("cached-chain", "    S* s = p->q;\n    if (s->m == 0) {\n        p = 0;\n    }\n    a = s->n;\n")
        self.assertFalse(any("p->q->m" in g for g in got), got)
        got = outs("cached-chain", "    int k = p->q->m;\n    a = k;\n")
        self.assertEqual(got, [])


class BoolIndexTest(unittest.TestCase):
    def test_cast_to_select_in_an_index_and_back(self):
        got = outs("bool-index", "    a = p->t[(int)(b >= 2)];\n")
        self.assertTrue(has(got, "a = p->t[b >= 2 ? 1 : 0];"), got)
        back = outs("bool-index", "    a = p->t[b >= 2 ? 1 : 0];\n")
        self.assertTrue(has(back, "a = p->t[(int)(b >= 2)];"), back)

    def test_select_is_parenthesised_outside_an_index(self):
        got = outs("bool-index", "    a = (int)(b > a) + 1;\n")
        self.assertTrue(has(got, "a = (b > a ? 1 : 0) + 1;"), got)
        self.assertEqual(outs("bool-index", "    a = (int)x;\n"), [])
        self.assertEqual(outs("bool-index", "    a = b ? 1 : 0;\n"), [])
