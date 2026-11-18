"""dk llm-agent (llmagent.py).

Synthetic: action parsing (edit, structured, STOP, malformed), the exact-substring rule, anchors, retrieval's
leave-one-out, the turn loop against a fake Ollama and a fake compiler (best state kept, malformed and apply
counts, a judged exact result written as a candidate, STOP), set loading and the summary."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_llmtry import FakeOllama
from tools.dk.platforms.gc_mwcc import llmagent
from tools.dk.platforms.gc_mwcc.llmagent import (Action, Examples, Score, Settings, Task, anchor_line, apply_edit,
                                                 fix_hunk, load_set, no_comments, parse_action, run_task, summarize)
from tools.dk.platforms.gc_mwcc.llmtry import Ollama

REPO = Path(__file__).resolve().parents[1]

SRC = "int g;\nint f(int a) {\n    int b = a + 1;\n    return b * 2;\n}\n"


def edit(old, new):
    return f"<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n"


class ParseTest(unittest.TestCase):
    def test_edit_block(self):
        a = parse_action("Here:\n```\n" + edit("    int b = a + 1;", "    int b = 1 + a;") + "```")
        self.assertEqual((a.kind, a.old, a.new, a.error), ("edit", "    int b = a + 1;", "    int b = 1 + a;", ""))

    def test_structured_and_stop(self):
        a = parse_action('{"transform": "operand-swap", "anchor": 3}')
        self.assertEqual((a.kind, a.transform, a.anchor), ("structured", "operand-swap", 3))
        a = parse_action("STOP: register-only")
        self.assertEqual((a.kind, a.label), ("stop", "register-only"))

    def test_malformed(self):
        self.assertIn("no action", parse_action("I think the loop is wrong.").error)
        self.assertIn("2 actions", parse_action(edit("a", "b") + edit("c", "d")).error)
        self.assertIn("unknown transform", parse_action('{"transform": "make-it-match", "anchor": 1}').error)
        self.assertIn("unknown STOP label", parse_action("STOP: tired").error)
        self.assertIn("no anchor", parse_action('{"transform": "if-invert"}').error)
        self.assertIn("the same", parse_action(edit("x", "x")).error)
        self.assertIn("only edit actions", parse_action('{"transform": "if-invert", "anchor": 2}', "edit").error)
        self.assertIn("only structured", parse_action(edit("a", "b"), "structured").error)
        self.assertEqual(parse_action("STOP: no-idea", "edit").kind, "stop")       # STOP is always allowed

    def test_think_block_ignored(self):
        self.assertEqual(parse_action("<think>STOP: no-idea</think>" + edit("a", "b")).kind, "edit")


class ApplyTest(unittest.TestCase):
    span = (SRC.index("int f"), SRC.index("}\n") + 1)

    def test_exact_substring_only(self):
        new, why = apply_edit(SRC, self.span, "a + 1", "1 + a")
        self.assertEqual((new, why), (SRC.replace("a + 1", "1 + a"), ""))
        self.assertIsNone(apply_edit(SRC, self.span, "a+1", "1 + a")[0])          # whitespace differs: rejected
        self.assertIn("not an exact substring", apply_edit(SRC, self.span, "int g;", "int h;")[1])   # outside the function
        self.assertIn("occurs 2 times", apply_edit(SRC, self.span, "b", "c")[1])

    def test_anchor(self):
        self.assertEqual(anchor_line(SRC, self.span, 2), (2, ""))
        self.assertEqual(anchor_line(SRC, self.span, "return b * 2;"), (3, ""))
        self.assertEqual(anchor_line(SRC, self.span, "return  b * 2;")[0], 3)      # whitespace-folded
        self.assertIn("outside", anchor_line(SRC, self.span, 9)[1])
        self.assertIn("matches no line", anchor_line(SRC, self.span, "while (1)")[1])


class RetrievalTest(unittest.TestCase):
    def rec(self, fn, shape="s", cls="shape", rows=("    0  lwz r3, 0x4(r3)    op  lbz r3, 0x4(r3)",)):
        return {"function": fn, "asm_shape": shape, "diff_class": cls, "rows": list(rows), "before_percent": 90.0,
                "before": "void f() {\n    // a note\n    x = 1;\n}", "after": "void f() {\n    x = 2;\n}",
                "fix_kind": ["literal"]}

    def test_leave_one_out(self):
        ex = Examples([self.rec("a", "s1"), self.rec("b", "s2"), self.rec("c", "s3", "register")])
        got = [r["function"] for r in ex.top("a", "s9", ["    0  lwz r3, 0x4(r3)    op  lbz r3"], False, 3)]
        self.assertNotIn("a", got)                       # never the same function
        self.assertEqual(got[0], "b")                    # same diff class first
        got = [r["function"] for r in ex.top("x", "s2", [], False, 3)]
        self.assertNotIn("b", got)                       # nor the same asm shape

    def test_render_strips_comments(self):
        text = llmagent.render_example(1, self.rec("b"))
        self.assertNotIn("a note", text)
        self.assertIn("<<<<<<< SEARCH\n    x = 1;\n=======\n    x = 2;\n>>>>>>> REPLACE", text)
        self.assertEqual(no_comments("a; // n\n/* m */b;"), "a;\nb;")

    def test_fix_hunk_unique(self):
        before = "{\n    x = 0;\n    y = 1;\n    x = 0;\n}"
        after = "{\n    x = 0;\n    y = 1;\n    x = 5;\n}"
        old, new = fix_hunk(before, after)
        self.assertEqual(before.count(old), 1)
        self.assertEqual(before.replace(old, new), after)


class FakeEnv:
    """Scores by a table: text -> percent; the judge accepts unless the text holds `volatile`."""

    def __init__(self, table, start):
        self.table, self.start, self.calls = table, start, []

    def definition(self, text):
        a = text.index("int f")
        return a, text.index("\n}\n", a) + 2

    def score(self, text):
        self.calls.append(text)
        if "syntax error" in text:
            return Score(False, error="error: syntax")
        pct = self.table.get(text, 50.0)
        return Score(True, pct, (-(pct == 100.0), -pct), ["  ...", "    0  add r3, r3, r4   arg  add r3, r4, r3"])

    def oracle_text(self, text, score):
        return "oracle: 3 values, 2 of 3 coloured as the target"

    def structured(self, text, score, rule, line0):
        return [text.replace("a + 1", "1 + a")] if rule == "operand-swap" else []

    def judge(self, text, score):
        bad = "volatile" in text
        return (not bad), (["adds volatile"] if bad else []), text


class LoopTest(unittest.TestCase):
    def run_agent(self, replies, table, turns=8, action="both"):
        fake = FakeOllama(replies)
        rows = []
        try:
            env = FakeEnv(table, SRC)
            out = run_task(Task("t", "u", "f"), env, SRC, Ollama(fake.url), Settings("fake", turns, 0, action),
                           Examples([]), rows.append)
        finally:
            fake.close()
        return out, rows, fake.requests

    def test_best_kept_counts_and_exact(self):
        v1 = SRC.replace("a + 1", "1 + a")              # 80%: kept
        v2 = v1.replace("b * 2", "2 * b")               # 60%: worse, reverted (the next edit applies to v1)
        v3 = v1.replace("int b = 1 + a;", "volatile int b = 1 + a;")      # 100% but a device
        v4 = v1.replace("return b * 2;", "return b + b;")                 # exact
        replies = ['{"transform": "operand-swap", "anchor": 2}', edit("b * 2", "2 * b"), "nothing useful",
                   edit("int b = a + 1;", "int b = 2;"),                  # SEARCH not in the best state: malformed
                   edit("    int b = 1 + a;", "    volatile int b = 1 + a;"), edit("return b * 2;", "return b + b;")]
        out, rows, reqs = self.run_agent(replies, {SRC: 50.0, v1: 80.0, v2: 60.0, v3: 100.0, v4: 100.0})
        self.assertEqual((out["result"], out["exact"], out["turns"], out["text"]), ("exact", True, 6, v4))
        turns = [r for r in rows if r["kind"] == "turn"]
        self.assertEqual([r["malformed"] for r in turns], [False, False, True, True, False, False])
        self.assertEqual([r["applied"] for r in turns], [True, True, False, False, True, True])
        self.assertEqual(turns[1]["kept"], False)
        self.assertEqual(turns[4]["judge"], ["adds volatile"])
        self.assertIn("worse than the best 80.00%: reverted", reqs[2]["messages"][1]["content"])
        self.assertIn("1 + a", reqs[3]["messages"][1]["content"])           # the state is the best (v1)
        s = summarize(rows + [{"kind": "task", "set": "t", **out}])["t"]
        self.assertEqual((s["exact"], s["replies"], s["malformed"], s["actions"], s["applied"]), (1, 6, 2, 6, 4))

    def test_stop_and_compile_error(self):
        out, rows, _ = self.run_agent([edit("return b * 2;", "return b * 2 syntax error;"), "STOP: header-type"], {})
        self.assertEqual((out["result"], out["label"], out["turns"]), ("stop", "header-type", 2))
        self.assertEqual(rows[0]["compiled"], False)

    def test_structured_not_applicable_is_not_malformed(self):
        out, rows, _ = self.run_agent(['{"transform": "if-invert", "anchor": 2}'], {}, turns=1)
        self.assertEqual((rows[0]["malformed"], rows[0]["applied"]), (False, False))
        self.assertEqual(out["result"], "miss")

    def test_sft_render_matches_rlgate(self):
        v1 = SRC.replace("a + 1", "1 + a")
        fake = FakeOllama(['{"transform": "operand-swap", "anchor": 2}', "STOP: no-idea"])
        try:
            env = FakeEnv({SRC: 50.0, v1: 80.0}, SRC)
            st = Settings("fake", 2, 0, "both", render="sft")
            run_task(Task("t", "u", "f"), env, SRC, Ollama(fake.url), st, Examples([]), lambda r: None)
        finally:
            fake.close()
        from tools.dk.platforms.gc_mwcc import rlgate
        s0 = env.score(SRC)
        a, b = env.definition(SRC)
        want = rlgate.render("f", SRC[a:b], s0, [])
        self.assertEqual(fake.requests[0]["messages"], want)
        second = fake.requests[1]["messages"][1]["content"]
        self.assertNotIn("Oracle:", second)
        self.assertIn("80.00%, 1 rows shown differ (kept: the new best)", second)
        self.assertNotIn("oracle", "".join(m["content"] for r in fake.requests for m in r["messages"]).lower())

    def test_edit_only_mode_prompt(self):
        _out, _rows, reqs = self.run_agent(["STOP: no-idea"], {}, action="edit")
        system = reqs[0]["messages"][0]["content"]
        self.assertIn("<<<<<<< SEARCH", system)
        self.assertNotIn('"transform"', system)


class SetTest(unittest.TestCase):
    def test_kinds_and_filter(self):
        with tempfile.TemporaryDirectory() as d:
            fixes = Path(d) / "fixes.jsonl"
            fixes.write_text(json.dumps({"unit": "u", "function": "a", "parent": "p1", "path": "src/a.cpp",
                                         "before": "B", "after": "A"}) + "\n")
            gate = Path(d) / "gate.json"
            gate.write_text(json.dumps({"functions": [
                {"unit": "u", "function": "g1", "commit": "c1", "source": "src/g.cpp", "before": "G", "why": "gave-up",
                 "source_blob": "abc"},
                {"unit": "u", "function": "g2", "commit": "c1", "source": "src/g.cpp", "before": "H", "why": "never-tried"},
                {"unit": "u", "function": "plain"}]}))
            a = load_set(str(fixes))
            self.assertEqual((a[0].rev, a[0].before, a[0].after), ("p1", "B", "A"))
            g = load_set(f"{gate}#why=gave-up")
            self.assertEqual([(t.function, t.rev, t.blob, t.set) for t in g], [("g1", "c1", "abc", "gate#why=gave-up")])
            self.assertIsNone(load_set(str(gate))[2].rev)     # a plain row: the repo's function now


if __name__ == "__main__":
    unittest.main()
