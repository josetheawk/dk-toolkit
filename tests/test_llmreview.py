"""The overseer task (llmreview.py): format, labels, balance, eval scoring and the grind plumbing.
Every fixture is synthetic."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_project
from tools.dk.platforms.gc_mwcc import llmgrind, llmreview as R
from tools.dk.platforms.gc_mwcc.judge import review

ASM = ".fn f, global\n    0  stwu r1, -0x10(r1)\n    4  lwz r3, 0x4(r3)\n    8  bl g\n    C  fadds f1, f1, f2\n   10  blr\n"
DECLS = "struct Thing {\n    int m4;\n    int m8;\n};\nint g(Thing* t);\nstruct Other {\n    int z;\n};\n"
CODE = "int f(Thing* t, int a) {\n    int n = 0;\n    t->m4 = a;\n    t->m8 = n;\n    return g(t) + a;\n}"


def item(**kw):
    base = {"unit": "main/game/a", "function": "f", "path": "src/game/a.cpp", "asm": ASM, "decls": DECLS,
            "before_def": "", "after_def": CODE, "exact": True, "before": 0.0, "after": 100.0, "flags": []}
    return {**base, **kw}


class FormatTest(unittest.TestCase):
    def test_prompt_holds_the_facts_compactly(self):
        sys_msg, user = R.build_review_messages(item(flags=["adds a comma operator"]))
        self.assertIn("VERDICT: clean|needs-judgment|reject", sys_msg["content"])
        u = user["content"]
        self.assertIn("5 instructions (0x14 bytes)", u)
        self.assertIn("calls: g", u)
        self.assertIn("exact (100%); the function was unwritten", u)
        self.assertIn("Judge flags: adds a comma operator", u)
        self.assertIn("struct Thing", u)
        self.assertNotIn("struct Other", u)               # declarations cut to what the diff uses
        self.assertIn("+    t->m4 = a;", u)
        self.assertNotIn("+++", u)
        hidden = R.build_review_messages(item(flags=["x"]), show_flags=False)[1]["content"]
        self.assertIn("Judge flags: (not shown)", hidden)

    def test_score_line_of_a_repair(self):
        u = R.build_review_messages(item(before_def=CODE.replace("+ a", "+ 1"), exact=False, before=90.0, after=95.5))[1]
        self.assertIn("not exact (95.50%); the current source scores 90.00%", u["content"])
        self.assertIn("-    return g(t) + 1;", u["content"])

    def test_answer_round_trips_and_parsing_is_lenient(self):
        rec = {**item(), "verdict": "needs-judgment", "rank": 42, "why": "check:  volatile\n local"}
        out = R.render(rec, True)["messages"][-1]["content"]
        self.assertEqual(out, "VERDICT: needs-judgment\nRANK: 42\nWHY: check: volatile local")
        self.assertEqual(R.parse_verdict(out), {"verdict": "needs-judgment", "rank": 42, "why": "check: volatile local"})
        got = R.parse_verdict("<think>hm</think>\n```\n**VERDICT:** Needs judgment\nRANK: 140\nWHY: odd\n```")
        self.assertEqual((got["verdict"], got["rank"]), ("needs-judgment", 100))
        self.assertEqual(R.parse_verdict("VERDICT: REJECT")["rank"], None)
        self.assertIsNone(R.parse_verdict("looks fine to me"))

    def test_ranks_order_the_verdicts(self):
        self.assertEqual(R.label_rank("clean", 100.0, True), 100)
        self.assertEqual(R.label_rank("clean", 99.9, False), 89)
        self.assertEqual(R.label_rank("clean", 92.0, False), 83)
        self.assertEqual(R.label_rank("clean", 92.0, False, before=95.0), 20)   # faithful but worse than now
        self.assertEqual(R.label_rank("needs-judgment", 100.0, True), 60)
        self.assertEqual(R.label_rank("reject", 100.0, True), 10)
        self.assertEqual(R.label_rank("reject", 40.0, False), 4)


class LabelTest(unittest.TestCase):
    DOC = ("| Unit | Function | Rule | Verdict | Reason |\n|---|---|---|---|---|\n"
           "| main/a | `f1` | comma operator | **device** | `a, b` in an if |\n"
           "| main/a | `f2` | cached member or global read | allowed | read once in the original |\n"
           "| main/a | `f3` | unused local | open | unclear |\n")

    def row(self, fn, flags, triage=None):
        r = {"unit": "main/a", "function": fn, "flags": flags, "source": "src/a.cpp"}
        if triage:
            r["triage"] = triage
        return r

    def test_triage_reasons_and_audit_labels(self):
        reasons = R.triage_reasons(self.DOC)
        self.assertEqual(reasons[("main/a", "f1")], ("device", "a, b in an if"))
        self.assertEqual(R.audit_label(self.row("f1", ["adds a comma operator"], {"comma operator": "device"}), reasons),
                         ("reject", "device: a, b in an if"))
        v, why = R.audit_label(self.row("f2", ["new local x only caches m4 (a member or global nothing changes in "
                                               "between)"], {"cached member or global read": "allowed"}), reasons)
        self.assertEqual(v, "clean")
        self.assertIn("read once in the original", why)
        self.assertEqual(R.audit_label(self.row("f3", ["new local x is never read"], {"unused local": "open"}), reasons)[0],
                         "needs-judgment")
        self.assertIsNone(R.audit_label(self.row("f4", ["new local x is never read"]), reasons))   # nobody decided
        self.assertIsNone(R.audit_label(self.row("f5", ["adds a cast to a raw pointer type"]), reasons))

    def test_judged_labels(self):
        self.assertEqual(R.judged_label("clean", [], ["#pragma"], True, 100.0)[0], "reject")
        self.assertEqual(R.judged_label("needs-judgment", ["volatile x"], [], True, 100.0), ("needs-judgment", "check: volatile x"))
        self.assertEqual(R.judged_label("clean", [], [], False, 93.5)[1], "faithful but not exact (93.50%)")

    def test_split_is_fixed_and_about_one_in_ten(self):
        units = [f"main/game/u{i}" for i in range(2000)]
        picked = [u for u in units if R.review_split(u)]
        self.assertEqual(picked, [u for u in units if R.review_split(u)])
        self.assertTrue(150 < len(picked) < 250)

    def test_latest_label_wins_and_failed_landings_are_no_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            ddir = Path(d)
            (ddir / "candidates").mkdir()
            rows = [{"id": "a", "label": "rejected", "why": "pragma"}, {"id": "b", "label": "land-failed"},
                    {"id": "c", "label": "land-failed"}, {"id": "c", "label": "approved"}]
            (ddir / "candidates" / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            got = R.latest_labels(ddir)
            self.assertEqual({k: v["label"] for k, v in got.items()}, {"a": "rejected", "c": "approved"})


class TriageFileTest(unittest.TestCase):
    """The triage label file (TRIAGE_LABELS): keyed (unit, function), strictest wins, the coordinator overrides."""

    def test_strictest_wins_and_the_coordinator_overrides(self):
        tri = {"unit": "u", "function": "f", "verdict": "reject", "why": "device: alias", "label_source": "qt3-triage"}
        self.assertEqual(R.merge_label(("clean", "allowed"), tri, None), ("reject", "device: alias", "triage", "triage"))
        self.assertEqual(R.merge_label(("reject", "device"), {**tri, "verdict": "clean"}, None)[:3], ("reject", "device", "audit"))
        self.assertEqual(R.merge_label(("needs-judgment", "open"), {**tri, "verdict": "needs-judgment"}, None)[2], "audit")
        self.assertEqual(R.merge_label(None, {**tri, "verdict": "needs-judgment"}, None)[0], "needs-judgment")
        self.assertIsNone(R.merge_label(None, None, None))
        self.assertEqual(R.merge_label(("reject", "d"), tri, ("clean", "approved"))[::3], ("clean", "coordinator"))
        self.assertEqual(R.merge_label(("reject", "d"), {**tri, "verdict": "clean", "label_source": "coordinator"}, None)[0],
                         "clean")

    def test_new_triage_labels_never_push_items_out_of_the_eval(self):
        def rec(i, src, verdict, set_="heldout-final"):
            return {"rid": f"r{i}", "unit": "u", "function": f"f{i}", "source": src, "verdict": verdict, "set": set_}
        ev = [rec(i, "heldout-exact", "clean") for i in range(4)] + [rec(10 + i, "heldout-attempt", "reject") for i in range(4)]
        before, _ = R.balance(ev, 0.5, R.EVAL_FIRST)
        after, _ = R.balance(ev + [rec(20 + i, "triage", "clean", "review-units") for i in range(5)], 0.5, R.EVAL_FIRST)
        self.assertEqual({r["rid"] for r in after if r["set"] == "heldout-final"}, {r["rid"] for r in before})

    def test_file_newest_row_per_function_and_coordinator_by_function(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = [{"unit": "u", "function": "f", "verdict": "clean", "time": 2},
                    {"unit": "u", "function": "f", "verdict": "reject", "time": 1},
                    {"unit": "u", "function": "g", "verdict": "maybe", "time": 3}]
            (d / R.TRIAGE_LABELS).write_text("".join(json.dumps(r) + "\n" for r in rows))
            got = R.triage_file_labels(d / R.TRIAGE_LABELS)
            self.assertEqual({k: v["verdict"] for k, v in got.items()}, {("u", "f"): "clean"})
            self.assertEqual(R.triage_file_labels(d / "missing.jsonl"), {})
            (d / "candidates" / "2026-11-02").mkdir(parents=True)
            for cid, fn in (("a1", "f"), ("b2", "g")):
                (d / "candidates" / "2026-11-02" / f"{cid}.json").write_text(json.dumps({"id": cid, "unit": "u", "function": fn}))
            labels = {"a1": {"label": "approved", "time": 5}, "b2": {"label": "rejected", "why": "cast", "time": 6}}
            self.assertEqual(R.coordinator_by_function(d, labels),
                             {("u", "f"): ("clean", "approved by the coordinator"),
                              ("u", "g"): ("reject", "rejected by the coordinator: cast")})


class InjectTest(unittest.TestCase):
    def test_each_device_kind_and_the_judge_sees_most(self):
        code = "int f(Thing* t, int a) {\n    int i;\n    int n = 0;\n    t->m4 = a;\n    t->m8 = n;\n    return n + a;\n}\n"
        seen = {}
        for kind in R.KINDS:
            bad = R.inject(code, kind, 1)
            self.assertIsNotNone(bad, kind)
            self.assertNotEqual(bad, code)
            stub = "struct Thing;\n"
            seen[kind] = review(stub, stub + bad, "f")
        self.assertIn("volatile int i;", R.inject(code, "volatile"))
        self.assertIn("register int i;", R.inject(code, "register"))
        self.assertIn("t->m4 = a, t->m8 = n;", R.inject(code, "comma"))
        alias = R.inject(code, "alias")
        self.assertIn("Thing* t2 = t;", alias)
        self.assertIn("t2->m4 = a;", alias)
        self.assertTrue(R.inject(code, "pragma", 0).startswith("#pragma dont_inline on\n"))
        for kind in ("dummy", "pragma", "register", "comma"):
            self.assertNotEqual(seen[kind][0], "clean", kind)
        self.assertIsNone(R.inject("int f() {\n    return 0;\n}\n", "volatile"))
        self.assertIsNone(R.inject("void f(int a) {\n    a++;\n}\n", "alias"))     # a written parameter: not an alias


class BalanceTest(unittest.TestCase):
    def test_people_and_real_attempts_first(self):
        def rec(i, src, verdict):
            return {"rid": f"r{i}", "unit": "u", "function": f"f{i}", "source": src, "verdict": verdict}
        recs = [rec(i, "exact", "clean") for i in range(20)] + [rec(100, "audit", "clean"), rec(101, "attempt", "clean")]
        recs += [rec(200 + i, "synthetic", "reject") for i in range(3)]
        out, cut = R.balance(recs, 0.5)
        clean = [r for r in out if r["verdict"] == "clean"]
        self.assertEqual(len(clean), 3)
        self.assertEqual(cut, 19)
        self.assertEqual({r["source"] for r in clean} >= {"audit", "attempt"}, True)
        self.assertEqual(R.balance(recs, 1.0)[1], 0)


class FakeClient:
    """Answers each review with a script keyed by the function name (no server, no GPU)."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def chat(self, model, messages, options, think=None, keep_alive="15m"):
        self.calls.append((model, messages, options, think))
        fn = messages[1]["content"].split("`")[1]
        return {"message": {"role": "assistant", "content": self.answers.get(fn, "no idea")}}

    def loaded(self):
        return []


class BenchTest(unittest.TestCase):
    def test_scores_per_verdict_and_reject_recall(self):
        rows = [{"label": "reject", "got": "reject", "rank": 5, "rank_label": 10, "source": "s"},
                {"label": "reject", "got": "clean", "rank": 90, "rank_label": 10, "source": "s"},
                {"label": "clean", "got": "clean", "rank": 100, "rank_label": 100, "source": "e"},
                {"label": "needs-judgment", "got": None, "rank": None, "rank_label": 60, "source": "e"}]
        s = R.scores(rows)
        self.assertEqual((s["n"], s["parsed"], s["accuracy"], s["reject_recall"]), (4, 3, 0.5, 0.5))
        self.assertEqual(s["reject_missed_as_clean"], 1)
        self.assertEqual(s["per_verdict"]["clean"], {"labels": 1, "said": 2, "precision": 0.5, "recall": 1.0})
        self.assertEqual(s["rank_mae"], round((5 + 80 + 0) / 3, 1))
        self.assertEqual(s["by_source"]["s"]["reject_recall"], 0.5)

    def test_run_against_a_fake_backend(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            root.mkdir()
            cfg = make_project(root)
            ddir = Path(d) / "llm-data"
            ddir.mkdir()
            recs = [{**item(function=f"f{i}", after_def=CODE.replace("int f(", f"int f{i}(")),
                     "rid": f"r{i}", "source": "heldout-attempt", "set": "heldout", "verdict": v, "rank": rk, "why": "x"}
                    for i, (v, rk) in enumerate([("clean", 100), ("reject", 10), ("needs-judgment", 60)])]
            (ddir / R.EVAL_FILE).write_text(json.dumps({"made": "now", "units": ["main/game/a"], "functions": recs}))
            client = FakeClient({"f0": "VERDICT: clean\nRANK: 97\nWHY: fine", "f1": "VERDICT: reject\nRANK: 3\nWHY: pragma"})
            parser = argparse.ArgumentParser()
            from tools.dk.platforms.gc_mwcc import llmtry
            llmtry._configure_bench(parser)
            args = parser.parse_args(["run", "--set", "review", "--model", "fake", "--run", "rev-test"])
            with mock.patch.dict(os.environ, {"DK_LLM_DATA": str(ddir)}), mock.patch("builtins.print"):
                self.assertEqual(R.bench_review(cfg, args, client=client), 0)
            rows = [json.loads(x) for x in (ddir / "rev-test.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["set"], "review")
            got = {r["function"]: (r["label"], r["got"]) for r in rows if r["kind"] == "review"}
            self.assertEqual(got, {"f0": ("clean", "clean"), "f1": ("reject", "reject"), "f2": ("needs-judgment", None)})
            self.assertEqual(client.calls[0][2]["temperature"], 0.0)
            self.assertFalse(client.calls[0][3])                     # thinking off
            self.assertEqual(R.report_rows(rows)["reject_recall"], 1.0)


SRC = "int g;\nint f(int a) {\n    return a;\n}\n"


class GrindPlumbingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "proj"
        self.root.mkdir()
        self.cfg = make_project(self.root, files={"src/game/a.cpp": SRC})
        self.ddir = Path(self.tmp.name) / "llm-data"
        env = mock.patch.dict(os.environ, {"DK_LLM_DATA": str(self.ddir)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)
        self.cdir = self.ddir / "candidates" / "2026-11-02"

    def cand(self, text):
        return {"unit": "main/game/a", "source": "src/game/a.cpp", "function": "f", "size": 0x10, "kind": "register",
                "route": "cpu", "before": 98.5, "after": 100.0, "model": None, "before_text": SRC, "text": text,
                "judge": [], "time": "2026-11-02 03:00"}

    def test_overseer_never_gates_and_is_shown_and_measured(self):
        client = FakeClient({"f": "VERDICT: reject\nRANK: 4\nWHY: looks like a device"})
        cand = self.cand(SRC.replace("return a;", "return a + 1;"))
        with mock.patch.object(R, "Facts", lambda cfg, by: (lambda u, f: {"asm": ASM, "decls": DECLS})):
            cand["overseer"] = llmgrind.ask_overseer(self.cfg, client, "ov", cand)
        self.assertEqual((cand["overseer"]["verdict"], cand["overseer"]["rank"], cand["overseer"]["judge"]),
                         ("reject", 4, "clean"))
        cid = llmgrind.write_candidate(self.cdir, cand)
        with mock.patch("builtins.print"):
            llmgrind.run_review(self.cfg, argparse.Namespace(date="", reject=None, approve=None, max_diff=40))
        digest = (self.cdir / "digest.md").read_text()
        self.assertIn(f"## {cid} clean", digest)                # the judge decides the category
        self.assertIn("Overseer (ov, never the gate): reject, rank 4: looks like a device", digest)
        with mock.patch("builtins.print"):
            llmgrind.run_review(self.cfg, argparse.Namespace(date="", reject=None, approve=[cid], why="ok", max_diff=40))
            out = R.agreement(self.ddir, say=lambda *_: None)
        label = json.loads((self.ddir / "candidates" / "labels.jsonl").read_text())
        self.assertEqual((label["label"], label["how"]), ("approved", "review"))
        self.assertEqual((out["candidates"], out["agrees_with_judge"], out["labelled"], out["agrees_with_label"]),
                         (1, 0, 1, 0))

    def test_the_overseer_model_does_not_pause_the_grind(self):
        class Loaded:
            def loaded(self):
                return [{"name": "ov"}, {"name": "worker"}]
        self.assertEqual(llmgrind.other_models(Loaded(), "worker"), ["ov"])
        self.assertEqual(llmgrind.other_models(Loaded(), "worker", ("ov",)), [])

    def test_new_flags_are_off_by_default(self):
        parser = argparse.ArgumentParser()
        llmgrind._configure_grind(parser)
        self.assertEqual(parser.parse_args([]).overseer, "")
        from tools.dk.platforms.gc_mwcc import llmdata
        parser = argparse.ArgumentParser()
        llmdata._configure(parser)
        self.assertFalse(parser.parse_args(["export"]).never_review_eval)    # the frozen worker export is unchanged
        self.assertFalse(parser.parse_args(["export"]).with_review)          # joint only when asked
        self.assertEqual(parser.parse_args(["export", "--with-review"]).review_share, 0.05)
        self.assertFalse(parser.parse_args(["review"]).export)

    def test_eval_never_reads_the_eval_set(self):
        (self.ddir).mkdir(parents=True, exist_ok=True)
        (self.ddir / R.EVAL_FILE).write_text(json.dumps({"units": ["main/u"], "functions": [
            {"unit": "main/u", "function": "f", "asm_shape": "abc"}, {"unit": "main/u", "function": "g"}]}))
        self.assertEqual(R.eval_never(self.cfg), ({"main/u"}, {"f", "g"}, {"abc"}))


def rrec(i, unit="main/game/a", source="exact", verdict="clean", split="train", **kw):
    """A review record as `dataset review` stores it (synthetic)."""
    code = CODE.replace("int f(", f"int f{i}(")
    return {**item(function=f"f{i}", unit=unit, after_def=code, before_def=code if source != "exact" else "",
                   asm_shape=f"shape{i}", size=0x20 + i),
            "rid": f"r{i}", "source": source, "label_source": {"audit": "triage", "attempt": "judge"}.get(source, "exact"),
            "verdict": verdict, "rank": R.label_rank(verdict, 100.0, True), "why": "a reason", "split": split, **kw}


class JointExportTest(unittest.TestCase):
    """`dataset export --with-review`: the overseer's rows appended to the worker's (LoRA 1 is joint)."""
    EV = {"units": ["main/game/ev"], "functions": [
        {"unit": "main/game/ev", "function": "g1", "asm_shape": "evshape", "set": "review-units"},
        {"unit": "main/game/held", "function": "h1", "asm_shape": "hshape", "set": "heldout-final"},
        {"unit": "main/game/held", "function": "h2", "set": "heldout"}]}
    NEVER = ({"main/game/held", "main/game/bench"}, {"h1", "h2", "b1"}, {"hshape"})

    def recs(self):
        out = [rrec(i) for i in range(6)] + [rrec(10 + i, source="synthetic", verdict="reject") for i in range(6)]
        out += [rrec(30, source="audit", verdict="needs-judgment"), rrec(31, source="attempt", verdict="reject")]
        return out + [rrec(40, unit="main/game/ev", split="eval")]          # review-dev: never trained

    def test_rows_are_what_the_overseer_is_sent_tagged_review_weight_one(self):
        rows, card = R.joint_review_rows(self.recs(), self.EV, self.NEVER, 1e9, 0.05, 2.2, 16384)
        self.assertEqual(card["dropped"]["eval split"], 1)
        self.assertEqual((card["rows"], card["dropped"]["share cap"]), (14, 0))     # under the cap: all of them
        self.assertTrue(all(r["tags"]["task"] == "review" and r["tags"]["sample_weight"] == 1.0 for r in rows))
        self.assertNotIn("f40", {r["tags"]["function"] for r in rows})
        by_fn = {r["function"]: r for r in self.recs()}
        client = FakeClient({})
        for row in rows:
            rec = by_fn[row["tags"]["function"]]
            R.ask(client, "m", rec, row["tags"]["flags_shown"])           # what llm-review/llm-grind send Ollama
            self.assertEqual(client.calls[-1][1], row["messages"][:-1])
            got = R.parse_verdict(row["messages"][-1]["content"])
            self.assertEqual((got["verdict"], got["rank"]), (rec["verdict"], rec["rank"]))
            self.assertEqual(row["tags"]["size"], rec["size"])

    def test_share_cap_keeps_people_and_real_attempts_first(self):
        rows, _ = R.joint_review_rows(self.recs(), self.EV, self.NEVER, 1e9, 1.0, 2.2, 16384)
        tokens = sum(r["tags"]["approx_tokens"] for r in rows)
        worker = tokens * 3                                     # the cap allows a third of the rows' tokens
        got, card = R.joint_review_rows(self.recs(), self.EV, self.NEVER, worker, 0.1, 2.2, 16384)
        self.assertLessEqual(card["tokens"], 0.1 * (worker + card["tokens"]))
        self.assertLessEqual(card["share"], 0.1)
        self.assertGreater(card["dropped"]["share cap"], 0)
        self.assertTrue({"audit", "attempt"} <= {r["tags"]["source"] for r in got})
        self.assertEqual(R.joint_review_rows(self.recs(), self.EV, self.NEVER, worker, 0.0, 2.2, 16384)[1]["rows"], 0)

    def test_a_record_of_a_never_or_eval_unit_function_or_shape_is_refused(self):
        for bad in (rrec(50, unit="main/game/held"), rrec(51, unit="main/game/ev"), rrec(52, asm_shape="evshape"),
                    {**rrec(53), "function": "b1"}):
            with self.assertRaises(ValueError):
                R.joint_review_rows(self.recs() + [bad], self.EV, self.NEVER, 1e9, 0.05, 2.2, 16384)

    def test_overlap_review_final_must_be_clean_review_dev_is_counted(self):
        worker = [{"tags": {"unit": "main/game/ev", "function": "g1", "asm_shape": "evshape"}},
                  {"tags": {"unit": "main/game/a", "function": "f0", "asm_shape": "x"}}]
        o = R.worker_overlap(worker, self.EV)
        self.assertTrue(o["review_final_clean"])
        self.assertEqual((o["review-dev"]["units_in_worker_training"], o["review-dev"]["items_in_those_units"],
                          o["review-dev"]["worker_rows_in_those_units"]), (1, 1, 1))
        o = R.worker_overlap(worker + [{"tags": {"unit": "main/game/z", "function": "h1"}}], self.EV)
        self.assertFalse(o["review_final_clean"])

    def test_join_export_totals_both_tasks_and_rechecks(self):
        (self.ddir / "dataset" / "records").mkdir(parents=True)
        (self.ddir / R.EVAL_FILE).write_text(json.dumps(self.EV))
        (self.ddir / "dataset" / "records" / R.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in self.recs()))
        worker = [{"messages": [], "tags": {"unit": "main/game/a", "function": "w", "approx_tokens": 1000,
                                            "sample_weight": 2.0}}]
        card = {"records": 1, "rows_written": 1, "tokens_per_epoch": 1000, "effective_tokens_per_epoch": 100000,
                "effective_tokens_by_cap": {"8192": 90000, "16384": 100000}, "sources": {"base": {"answer_tokens": 300}}}
        rows, card = R.join_export(self.cfg, self.ddir / "dataset", self.NEVER, worker, card, 0.05, 16384)
        rev = card["tasks"]["review"]
        self.assertEqual(rows[0], worker[0])
        self.assertEqual(card["records"], 1 + rev["records"])
        self.assertEqual(card["effective_tokens_per_epoch"], 100000 + rev["effective_tokens"])
        self.assertEqual(card["effective_tokens_by_cap"]["8192"], 90000 + rev["effective_tokens"])
        self.assertAlmostEqual(card["tasks"]["worker"]["effective_share"] + rev["effective_share"], 1.0, places=3)
        self.assertEqual(R.joint_recheck(rows, self.ddir / R.EVAL_FILE), 0)
        leak = {"messages": [], "tags": {"task": "review", "unit": "main/game/ev", "function": "g9"}}
        self.assertEqual(R.joint_recheck(rows + [leak], self.ddir / R.EVAL_FILE), 1)
        bad = worker + [{"messages": [], "tags": {"unit": "main/game/held", "function": "h1"}}]
        with self.assertRaises(ValueError):                    # review-final met in the worker rows
            R.join_export(self.cfg, self.ddir / "dataset", self.NEVER, bad, dict(card, records=1, rows_written=1,
                          tokens_per_epoch=1000, effective_tokens_per_epoch=100000), 0.05, 16384)

    setUp = GrindPlumbingTest.setUp


if __name__ == "__main__":
    unittest.main()
