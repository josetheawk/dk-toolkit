"""dk llm-agent --menu (llmmenu.py): the legal-move menu, its JSON schema and GBNF grammar, the reply parser
and the agent turn with a menu (fake Ollama, fake compiler). Synthetic only: the menu is built with the real
rewrite table on a small parsed function and checked against llm-agent's own way of applying a move."""

import json
import re
import unittest

from tests.test_llmagent import SRC, FakeEnv, edit
from tests.test_llmtry import FakeOllama
from tools.dk.core import cbody
from tools.dk.platforms.gc_mwcc import llmmenu, rewrites
from tools.dk.platforms.gc_mwcc.llmagent import BLOCKERS, Examples, Settings, Task, anchor_line, run_task
from tools.dk.platforms.gc_mwcc.llmmenu import Move, enumerate_moves, gbnf, json_schema, parse_reply, site_at
from tools.dk.platforms.gc_mwcc.llmtry import Ollama
from tools.dk.platforms.gc_mwcc.trycmd import definition

FN = """int g;
int h(int x);
int f(int a, int n) {
    int b = h(a) + 1;
    if (n == 0) {
        return 0;
    }
    for (int i = 0; i < n; i++) {
        b += h(i) * 2;
    }
    return b;
}
"""


# ---- tiny checkers for the two grammar forms (only the constructs llmmenu writes) ---------------------------

def schema_ok(doc, s) -> bool:
    if "anyOf" in s:
        return any(schema_ok(doc, x) for x in s["anyOf"])
    if "const" in s:
        return doc == s["const"]
    if "enum" in s:
        return doc in s["enum"]
    if s.get("type") == "string":
        return isinstance(doc, str) and len(doc) >= s.get("minLength", 0)
    if s.get("type") == "object":
        if not isinstance(doc, dict) or any(k not in doc for k in s.get("required", [])):
            return False
        if s.get("additionalProperties") is False and set(doc) - set(s["properties"]):
            return False
        return all(schema_ok(v, s["properties"][k]) for k, v in doc.items() if k in s["properties"])
    return False


def gbnf_regex(grammar: str) -> re.Pattern:
    rules = dict(ln.split(" ::= ", 1) for ln in grammar.strip().splitlines())
    tok = re.compile(r'"((?:\\.|[^"\\])*)"|(\[[^\]]*\])|([A-Za-z][\w-]*)|([()|*+?])|\s+')

    def lit(body):
        return re.escape(body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\"))

    def conv(rhs):
        out = []
        for m in tok.finditer(rhs):
            s, cls, ref, op = m.groups()
            if s is not None:
                out.append(lit(s))
            elif cls is not None:
                out.append(cls.replace("\\n", "\n"))
            elif ref is not None:
                out.append("(?:" + conv(rules[ref]) + ")")
            elif op is not None:
                out.append({"(": "(?:"}.get(op, op))
        return "".join(out)
    return re.compile(conv(rules["root"]), re.S)


def parsed(src):
    sp = definition(src, "f")
    body = cbody.parse_body(src, src.index("{", sp[0], sp[1]), cbody.type_names(src))
    return sp, body


class EnumerateTest(unittest.TestCase):
    def test_dedupe_order_and_pick(self):
        src = "x\nint f() {\n  a;\n  b;\n}\n"
        span = (2, len(src) - 1)
        table = {("r1", 2): [("one", "S1"), ("two", "S2")], ("r2", 2): [("dup", "S1")], ("r2", 3): [("three", "S3")],
                 ("r1", 3): [("same", src)]}
        moves = enumerate_moves(src, span, lambda r, l0: table.get((r, l0), []), ["r1", "r2"])
        self.assertEqual([(m.transform, m.anchor, m.pick, m.new) for m in moves],
                         [("r1", 2, 0, "S1"), ("r1", 2, 1, "S2"), ("r2", 3, 0, "S3")])
        self.assertEqual(moves[1].reply(), '{"transform": "r1", "anchor": 2, "pick": 1}')

    def test_real_table_menu_applies_as_the_agent_applies(self):
        sp, body = parsed(FN)
        names = [r.name for r in rewrites.RULES]
        moves = enumerate_moves(FN, sp, site_at(FN, body, True), names)
        self.assertGreater(len(moves), 3)
        for m in moves:                                  # llm-agent: anchor_line, then the rule's candidates there
            line0, why = anchor_line(FN, sp, m.anchor)
            self.assertEqual(why, "")
            site = rewrites.Site(FN, body, {line0}, True)
            got = [new for _r, _l, new in rewrites.candidates(site, [rewrites.BY_NAME[m.transform]], groups=False)]
            self.assertEqual(got[m.pick], m.new)
        self.assertEqual(len({m.new for m in moves}), len(moves))

    def test_rank_keeps_hot_lines(self):
        sp, _ = parsed(FN)
        moves = [Move("operand-swap", k, 0, "", f"s{k}") for k in range(1, 11)]
        first = FN.count("\n", 0, sp[0])

        class Sym:
            lines = {first + 7: 2}                       # file line (1-based) of definition line 7
            signs = {}
        kept = llmmenu.rank(moves, FN, sp, Sym(), limit=3)
        self.assertEqual([m.anchor for m in kept], [6, 7, 8])
        self.assertEqual(llmmenu.rank(moves, FN, sp, None, limit=0), moves)


class GrammarTest(unittest.TestCase):
    MOVES = [Move("name-value", 4, 0), Move("inline-local", 4, 0), Move("if-invert", 5, 1)]

    def test_schema(self):
        s = json_schema(self.MOVES, BLOCKERS)
        for m in self.MOVES:
            self.assertTrue(schema_ok(json.loads(m.reply()), s))
        self.assertFalse(schema_ok({"transform": "name-value", "anchor": 5}, s))      # off the menu
        self.assertFalse(schema_ok({"transform": "if-invert", "anchor": 5}, s))       # pick missing
        self.assertTrue(schema_ok({"stop": "no-idea"}, s))
        self.assertFalse(schema_ok({"stop": "tired"}, s))
        self.assertTrue(schema_ok({"search": "a", "replace": "b"}, s))
        self.assertFalse(schema_ok({"search": "a", "replace": "b"}, json_schema(self.MOVES, BLOCKERS, edits=False)))
        json.dumps(s)

    def test_gbnf(self):
        rx = gbnf_regex(gbnf(self.MOVES, BLOCKERS))
        for m in self.MOVES:
            self.assertTrue(rx.fullmatch(m.reply()), m.reply())
        self.assertFalse(rx.fullmatch('{"transform": "name-value", "anchor": 5}'))
        self.assertTrue(rx.fullmatch("STOP: register-only"))
        self.assertFalse(rx.fullmatch("STOP: tired"))
        self.assertTrue(rx.fullmatch(edit("    int b = a + 1;", "    int b = 1 + a;").rstrip("\n")))
        self.assertFalse(gbnf_regex(gbnf(self.MOVES, BLOCKERS, edits=False)).fullmatch(edit("a", "b").rstrip("\n")))
        self.assertNotIn("move ::=", gbnf([], BLOCKERS))                               # no moves: STOP or edit


class ReplyTest(unittest.TestCase):
    def test_json_forms_and_fallback(self):
        a = parse_reply('{"stop": "register-only"}')
        self.assertEqual((a.kind, a.label), ("stop", "register-only"))
        self.assertIn("unknown STOP label", parse_reply('{"stop": "tired"}').error)
        a = parse_reply('{"search": "a + 1", "replace": "1 + a"}')
        self.assertEqual((a.kind, a.old, a.new), ("edit", "a + 1", "1 + a"))
        self.assertIn("the same", parse_reply('{"search": "x", "replace": "x"}').error)
        a = parse_reply('{"transform": "if-invert", "anchor": 5, "pick": 1}')
        self.assertEqual((a.kind, a.transform, a.anchor, a.pick), ("structured", "if-invert", 5, 1))
        self.assertEqual(parse_reply("STOP: no-idea").kind, "stop")                   # text forms still parse
        self.assertTrue(llmmenu.on_menu(a, GrammarTest.MOVES))
        self.assertFalse(llmmenu.on_menu(parse_reply('{"transform": "if-invert", "anchor": 5}'), GrammarTest.MOVES))

    def test_add_menu_goes_before_the_diff(self):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "def\n1 | x\n\nDiff (target | ours): r\nYour one action:"}]
        out = llmmenu.add_menu(msgs, "MENU")
        self.assertEqual(out[1]["content"], "def\n1 | x\n\nMENU\nDiff (target | ours): r\nYour one action:")
        self.assertEqual(msgs[1]["content"].count("MENU"), 0)


class AgentMenuTest(unittest.TestCase):
    def run_agent(self, replies, table, grammar=True):
        fake = FakeOllama(replies)
        rows = []
        try:
            env = FakeEnv(table, SRC)
            st = Settings("fake", len(replies), 0, "both", menu=40, grammar=grammar)
            out = run_task(Task("t", "u", "f"), env, SRC, Ollama(fake.url), st, Examples([]), rows.append)
        finally:
            fake.close()
        return out, rows, fake.requests

    def test_menu_in_state_and_format_sent(self):
        v1 = SRC.replace("a + 1", "1 + a")               # FakeEnv: operand-swap applies at every line
        out, rows, reqs = self.run_agent(['{"transform": "operand-swap", "anchor": 2}'], {SRC: 50.0, v1: 80.0})
        user = reqs[0]["messages"][1]["content"]
        self.assertIn(llmmenu.HEADER, user)
        self.assertIn('{"transform": "operand-swap", "anchor": 1}', user)          # one listing for the one source
        self.assertNotIn('"anchor": 2}', user)
        self.assertLess(user.index(llmmenu.HEADER), user.index("Diff (target | ours"))
        alts = reqs[0]["format"]["anyOf"]
        self.assertEqual(alts[0]["properties"], {"transform": {"const": "operand-swap"}, "anchor": {"const": 1}})
        turn = rows[0]
        self.assertEqual((turn["menu"], turn["menu_all"], turn["on_menu"], turn["applied"]), (1, 1, False, True))
        self.assertEqual(out["best"], 80.0)

    def test_no_grammar_sends_no_format(self):
        _out, rows, reqs = self.run_agent(['{"stop": "no-idea"}'], {}, grammar=False)
        self.assertNotIn("format", reqs[0])
        self.assertEqual(rows[0]["action"], "stop")


if __name__ == "__main__":
    unittest.main()
