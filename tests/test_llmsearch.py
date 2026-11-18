import argparse
import unittest

from tests.test_llmtry import FakeOllama, fence
from tools.dk.platforms.gc_mwcc import llmsearch
from tools.dk.platforms.gc_mwcc.llmsearch import Budget, Node, allocate, node_key, select
from tools.dk.platforms.gc_mwcc.llmtry import Context, Ollama, Settings, _configure_try, search, settings_from


def v(n: int) -> str:
    return f"int f(int a) {{\n    return a + {n};\n}}"


SCORES = {"a + 1": 80.0, "a + 2": 70.0, "a + 3": 70.0, "a + 4": 85.0, "a + 5": 90.0, "a + 6": 95.0, "a + 9": 40.0}
EXACT = "int f(int a) {\n    return 1 + a;\n}"


def evaluate(code, slot):
    """Scripted scores; two variants at one score have the same diff rows (the same asm)."""
    if "1 + a" in code:
        return {"compiled": True, "percent": 100.0, "devices": [], "exact": True, "rows": [], "text": "whole"}
    for k, pct in SCORES.items():
        if k in code:
            return {"compiled": True, "percent": pct, "devices": [], "exact": False, "rows": [f"    0  row {pct}"]}
    return {"compiled": False, "error": "undefined identifier 'x'", "devices": []}


def run(replies, st, start=("int f(int a) {\n    return a;\n}\n", 50.0)):
    fake = FakeOllama(replies)
    rows = []
    try:
        ctx = Context("f", "    0  blr", "", "", start[0])
        out = search(ctx, start[0], start[1], [], evaluate, Ollama(fake.url), st, rows.append, jobs=2,
                     say=lambda *_: None)
    finally:
        fake.close()
    return out, rows, fake.requests


def settings(**kw):
    st = Settings(model="fake", think=False, prefill=False, **{k: x for k, x in kw.items() if k in ("n", "rounds")})
    st.search, st.beam, st.adaptive = kw.get("search", "greedy"), kw.get("beam", 3), kw.get("adaptive", False)
    st.budget = kw.get("budget")
    return st


class BeamTest(unittest.TestCase):
    def test_select_keeps_the_top_k_distinct_and_drops_an_empty_start(self):
        nodes = [Node("start", "", 0.0, [], "start", -1), Node("a", "x", 70.0, [], "k70", 0),
                 Node("b", "y", 70.0, [], "k70", 1), Node("c", "z", 80.0, [], "k80", 2), Node("d", "w", 60.0, [], "k60", 3)]
        self.assertEqual([n.id for n in select(nodes, 3)], ["c", "a", "d"])     # b: a's asm again
        self.assertEqual([n.id for n in select(nodes[:1], 3)], ["start"])      # nothing compiled yet
        self.assertEqual([n.id for n in allocate(select(nodes, 2), 5)], ["c", "a", "c", "a", "c"])

    def test_node_key(self):
        self.assertEqual(node_key({"percent": 70.0, "rows": ["r"]}), node_key({"percent": 70.0, "rows": ["r"]}))
        self.assertNotEqual(node_key({"percent": 70.0, "rows": ["r"]}), node_key({"percent": 70.0, "rows": ["s"]}))
        self.assertEqual(node_key({"asm_hash": "abc", "percent": 1.0}), "abc")

    def test_budget(self):
        b = Budget()
        self.assertFalse(b.draw(4))
        b.deposit(8)
        self.assertTrue(b.draw(4))
        self.assertEqual((b.bank, b.saved, b.spent), (4, 8, 4))


class TreeTest(unittest.TestCase):
    def test_tree_branches_the_next_round_across_the_beam(self):
        replies = [fence(v(1)), fence(v(2)), fence(v(3)), "no code",           # round 0: 80, 70, 70 (same asm), none
                   fence(v(4)), fence(v(9)), fence(v(4)), fence(v(5))]          # round 1
        out, rows, req = run(replies, settings(n=4, rounds=2, search="tree", beam=2))
        att = [r for r in rows if r["kind"] == "attempt"]
        self.assertEqual(len(att), 8)                                           # equal attempts to greedy
        self.assertEqual([a["parent"] for a in att[:4]], ["start"] * 4)
        self.assertEqual([a["parent"] for a in att[4:]], ["r0v0", "r0v1", "r0v0", "r0v1"])   # v2 kept, v3 its twin
        self.assertIn("Best attempt so far (scores 80.00%)", req[4]["messages"][1]["content"])
        self.assertIn("return a + 2;", req[5]["messages"][1]["content"])
        self.assertIn("scores 70.00%", req[5]["messages"][1]["content"])
        self.assertEqual([r["options"]["seed"] for r in req[4:]], [1001, 1002, 1003, 1004])
        prompts = [r for r in rows if r["kind"] == "prompt"]
        self.assertEqual([(p["round"], p["parent"]) for p in prompts], [(0, "start"), (1, "r0v0"), (1, "r0v1")])
        self.assertEqual(att[6]["duplicate_of"], 0)              # the same code from another parent: r1v0 reused
        self.assertEqual(out.best, 90.0)
        done = [r for r in rows if r["kind"] == "search"][0]
        self.assertEqual((done["rounds_run"], done["attempts"], done["beam_ids"]), (2, 8, ["r1v3", "r1v0"]))

    def test_an_exact_variant_stops_the_tree(self):
        out, rows, _ = run([fence(v(1)), fence(v(2)), fence(EXACT), fence(v(4))], settings(n=2, rounds=3, search="tree"))
        self.assertEqual(out.exact_code.strip(), EXACT)
        self.assertEqual(sum(1 for r in rows if r["kind"] == "attempt"), 4)

    def test_the_default_logs_as_before(self):
        out, rows, _ = run([fence(v(1)), fence(v(2)), fence(v(4)), fence(v(5))], settings(n=2, rounds=2))
        self.assertFalse(any("parent" in r or "search" in r for r in rows))
        self.assertFalse(any(r["kind"] == "search" for r in rows))
        self.assertEqual(out.best, 90.0)


class AdaptiveTest(unittest.TestCase):
    def test_a_stalled_function_banks_its_rounds_for_an_improving_one(self):
        bank = Budget()
        st = settings(n=2, rounds=2, adaptive=True, budget=bank)
        _, stalled, _ = run([fence(v(9)), "no code"], st)                     # 40% < the start's 50%: no progress
        self.assertEqual(sum(1 for r in stalled if r["kind"] == "attempt"), 2)
        self.assertEqual(bank.bank, 2)
        replies = [fence(v(2)), fence(v(9)), fence(v(1)), fence(v(9)), fence(v(4)), fence(v(6))]
        out, rows, _ = run(replies, st)                                        # 70, 80, then 95 in the extra round
        att = [r for r in rows if r["kind"] == "attempt"]
        self.assertEqual(len(att), 6)
        self.assertEqual(out.best, 95.0)
        self.assertEqual((bank.bank, bank.saved, bank.spent), (0, 2, 2))
        done = [r for r in rows if r["kind"] == "search"][0]
        self.assertEqual((done["rounds_run"], done["planned"], done["bank"]), (3, 3, 0))

    def test_settings_from_the_command_line(self):
        p = argparse.ArgumentParser()
        _configure_try(p)
        st = settings_from(p.parse_args(["u", "f"]))
        self.assertEqual((st.search, st.adaptive, st.budget), ("greedy", False, None))
        self.assertEqual(llmsearch.tag(st), {})
        st = settings_from(p.parse_args(["u", "f", "--search", "tree", "--beam", "4", "--adaptive"]))
        self.assertEqual((st.search, st.beam, st.adaptive), ("tree", 4, True))
        self.assertIsInstance(st.budget, Budget)
        self.assertEqual(llmsearch.tag(st), {"search": "tree", "beam": 4, "adaptive": True})


if __name__ == "__main__":
    unittest.main()
