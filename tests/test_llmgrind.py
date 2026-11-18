import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_project
from tests.test_llmtry import FakeOllama
from tools.dk.platforms.gc_mwcc import llmgrind
from tools.dk.platforms.gc_mwcc.judge import review
from tools.dk.platforms.gc_mwcc.llmtry import Ollama

SRC = "int g;\nint f(int a) {\n    return a;\n}\n"


def fn(name, size, pct, unit="main/game/a", source="src/game/a.cpp", address=0):
    return {"unit": unit, "source": source, "function": name, "address": address, "size": size, "fuzzy": pct}


class QueueTest(unittest.TestCase):
    def test_exclusions_and_size_order(self):
        open_fns = [fn("big", 0x300, 0.0, address=1), fn("small", 0x20, 0.0, address=2),
                    fn("bench", 0x20, 0.0, unit="main/game/b"), fn("busy", 0x20, 0.0, source="src/game/c.cpp"),
                    fn("@123", 0x20, 0.0), fn("__sinit_x", 0x20, 0.0), fn("far", 0x40, 40.0),
                    fn("near", 0x40, 95.0), fn("asm", 0x20, 0.0, source="src/game/a.s"), fn("huge", 0x2000, 0.0)]
        items = llmgrind.queue(open_fns, {"main/game/b"}, {"src/game/c.cpp"}, {})
        self.assertEqual([i["function"] for i in items], ["small", "near", "big"])
        self.assertEqual(items[1]["kind"], "near")

    def test_rates_rank_bytes_times_success(self):
        rates = {("unwritten", 0x40): 0.5, ("unwritten", 0x400): 0.01, ("near", 0x80): 0.2}
        items = llmgrind.queue([fn("u", 0x40, 0.0), fn("big", 0x400, 0.0), fn("n", 0x80, 90.0)], set(), set(), rates)
        self.assertEqual([i["function"] for i in items], ["u", "n", "big"])   # 32, 25.6, 4.1
        self.assertEqual(items[0]["priority"], 32.0)

    def test_success_rates_from_logs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "night.jsonl"
            rows = [{"kind": "function", "function": "a", "bench_kind": "heldout", "exact": True},
                    {"kind": "function", "function": "b", "bench_kind": "heldout", "exact": False},
                    {"kind": "function", "function": "c", "bench_kind": "register", "exact": False},
                    {"kind": "attempt", "function": "a"}]
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            rates = llmgrind.success_rates([p], {"a": 0x30, "b": 0x38, "c": 0x100})
        self.assertAlmostEqual(rates[("unwritten", 0x40)], 1.2 / 4)
        self.assertAlmostEqual(rates[("near", 0x100)], 0.2 / 3)


class GpuTest(unittest.TestCase):
    def test_waits_only_for_other_models(self):
        fake = FakeOllama([])                       # /api/ps lists the model "fake"
        try:
            client = Ollama(fake.url)
            self.assertTrue(llmgrind.wait_gpu(client, "fake", 0, say=lambda *_: None, nap=0.01))
            import time
            self.assertFalse(llmgrind.wait_gpu(client, "mine", time.time() + 0.05, say=lambda *_: None, nap=0.1))
        finally:
            fake.close()


class ReviewTest(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(review(SRC, SRC.replace("return a;", "return a + 1;"), "f"), ("clean", []))
        self.assertEqual(review("int g;\n", SRC, "f"), ("clean", []))       # unwritten: new definition only
        cat, flags = review(SRC, SRC.replace("return a;", "int t = a;;\n    return t;"), "f")
        self.assertEqual(cat, "reject")
        self.assertIn("adds an empty statement", flags)
        cat, flags = review(SRC, SRC.replace("int g;", "int g;\nint h;").replace("return a;", "return a + h;"), "f")
        self.assertEqual(cat, "needs-judgment")
        cat, flags = review(SRC, SRC.replace("int g;", "int g;\nstatic inline int h(int x) { return x; }")
                            .replace("return a;", "return h(a);"), "f")
        self.assertEqual((cat, flags), ("needs-judgment", ["new inline helper h"]))
        self.assertEqual(review(SRC, SRC.replace("int g;", "int g;\nint k(void) { return 0; }"), "f")[0], "reject")
        self.assertEqual(review(SRC, SRC.replace("int g;", "int g = 3;"), "f")[0], "reject")
        cat, flags = review(SRC, SRC.replace("return a;", "return (a, 2);"), "f")
        self.assertIn("adds a comma operator", flags)
        cat, flags = review(SRC, SRC.replace("return a;", "int i; for (i = 0, a = 1;; i++) {}\n"
                                             "    return *(int*)((char*)&a + 4);"), "f")
        self.assertEqual(flags[0], "hard-coded offset cast")
        self.assertNotIn("adds a comma operator", flags)
        self.assertNotIn("adds an empty statement", flags)
        cat, flags = review(SRC, SRC.replace("return a;", "volatile int x = a + 1;\n    return x;"), "f")
        self.assertEqual(cat, "needs-judgment")


def candidate(text, before=SRC, fn_name="f"):
    return {"unit": "main/game/a", "source": "src/game/a.cpp", "function": fn_name, "size": 0x10, "kind": "register",
            "route": "cpu", "before": 98.5, "after": 100.0, "model": None, "before_text": before, "text": text,
            "judge": [], "time": "2026-11-02 03:00"}


class CandidateTest(unittest.TestCase):
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

    def test_review_digest_clean_first_and_labels(self):
        good = llmgrind.write_candidate(self.cdir, candidate(SRC.replace("return a;", "return a + 1;")))
        bad = llmgrind.write_candidate(self.cdir, candidate(SRC.replace("return a;", "int t = a;\n    return t;")))
        with mock.patch("builtins.print"):
            self.assertEqual(llmgrind.run_review(self.cfg, argparse.Namespace(date="", reject=None, max_diff=40)), 0)
        text = (self.cdir / "digest.md").read_text()
        self.assertLess(text.index(f"## {good} clean"), text.index(f"## {bad} reject"))
        self.assertIn("+    return a + 1;", text)
        self.assertIn("98.50% -> 100%", text)
        with mock.patch("builtins.print"):
            llmgrind.run_review(self.cfg, argparse.Namespace(date="", reject=[bad[:5]], why="copy local", max_diff=40))
        label = json.loads((self.ddir / "candidates" / "labels.jsonl").read_text())
        self.assertEqual((label["id"], label["label"], label["why"]), (bad, "rejected", "copy local"))

    def land(self, ids, rc_check=0, rc_unit=0):
        calls = []

        def fake_dk(cfg, *argv):
            calls.append(argv)
            if argv[0] == "check":
                return rc_check if "--function" in argv else rc_unit
            return 0
        with mock.patch("builtins.print"):
            rc = llmgrind.run_land(self.cfg, argparse.Namespace(ids=ids, no_commit=False), run_dk=fake_dk)
        return rc, calls

    def test_land_saves_or_accepts(self):
        new = SRC.replace("return a;", "return a + 1;")
        cid = llmgrind.write_candidate(self.cdir, candidate(new))
        rc, calls = self.land([cid], rc_unit=1)
        self.assertEqual(rc, 0)
        self.assertEqual((self.root / "src/game/a.cpp").read_text(), new)
        self.assertEqual(calls[-1][0], "save")
        (self.root / "src/game/a.cpp").write_text(SRC)
        rc, calls = self.land([cid], rc_unit=0)
        self.assertEqual(calls[-1], ("accept", "main/game/a"))

    def test_land_restores_a_miss_and_resplices_clean_ones(self):
        new = SRC.replace("return a;", "return a + 1;")
        cid = llmgrind.write_candidate(self.cdir, candidate(new))
        rc, calls = self.land([cid], rc_check=1)
        self.assertEqual(rc, 1)
        self.assertEqual((self.root / "src/game/a.cpp").read_text(), SRC)
        self.assertEqual(len(calls), 1)                   # no save after a failed check
        moved = "int h;\n" + SRC                           # the file moved since the grind
        (self.root / "src/game/a.cpp").write_text(moved)
        rc, _ = self.land([cid])
        self.assertEqual((self.root / "src/game/a.cpp").read_text(), "int h;\n" + new)
        dirty = llmgrind.write_candidate(self.cdir, candidate(SRC.replace("int g;", "int g;\nint z;")
                                                              .replace("return a;", "return z;")))
        rc, calls = self.land([dirty])                      # not clean and the file moved: refused
        self.assertEqual((rc, calls), (1, []))


class NearTest(unittest.TestCase):
    def test_wants_near(self):
        self.assertTrue(llmgrind.wants_near("unwritten", 0.0, 91.0, 90.0, 5.0))
        self.assertFalse(llmgrind.wants_near("unwritten", 0.0, 89.9, 90.0, 5.0))
        self.assertFalse(llmgrind.wants_near("shape", 92.0, 95.0, 90.0, 5.0))      # a written one needs a big gain
        self.assertTrue(llmgrind.wants_near("register", 92.0, 97.0, 90.0, 5.0))

    def test_near_ok_only_judge_clean(self):
        ok = {"compiled": True, "percent": 95.0, "devices": [], "worse": []}
        good = SRC.replace("return a;", "return a + 1;")
        self.assertEqual(llmgrind.near_ok({**ok, "text": good}, SRC, "f", 50.0, 90.0), [])
        self.assertIsNone(llmgrind.near_ok({**ok, "text": good, "devices": ["volatile"]}, SRC, "f", 50.0, 90.0))
        self.assertIsNone(llmgrind.near_ok({**ok, "text": good, "worse": ["g"]}, SRC, "f", 50.0, 90.0))
        self.assertIsNone(llmgrind.near_ok({**ok, "text": good}, SRC, "f", 96.0, 90.0))       # no gain
        self.assertIsNone(llmgrind.near_ok({**ok, "text": good, "percent": 89.0}, SRC, "f", 0.0, 90.0))
        comma = SRC.replace("return a;", "return (a, 2);")                                     # reject
        self.assertIsNone(llmgrind.near_ok({**ok, "text": comma}, SRC, "f", 50.0, 90.0))
        judged = SRC.replace("int g;", "int g;\nint h;").replace("return a;", "return a + h;")  # needs judgment
        self.assertIsNone(llmgrind.near_ok({**ok, "text": judged}, SRC, "f", 50.0, 90.0))


class NearWorkTest(unittest.TestCase):
    def run_work(self, best, best_code, ev):
        from types import SimpleNamespace
        from tools.dk.platforms.gc_mwcc import llmtry
        ctx = SimpleNamespace(current="", m2c="")
        scorer = mock.Mock(forbidden=[], work=Path("/nonexistent"))
        scorer.baseline.return_value = (0.0, [])
        scorer.evaluate.return_value = ev
        parser = argparse.ArgumentParser()
        llmgrind._configure_grind(parser)
        args = parser.parse_args([])
        item = fn("f", 0x40, 0.0)
        out = llmtry.Outcome(0.0, best, best_code, "", "", 16)
        with mock.patch.object(llmtry, "prepare", return_value=({"name": "main/game/a"}, Path("src/game/a.cpp"), SRC,
                                                                 (0, 0), None, ctx)), \
                mock.patch.object(llmtry, "Scorer", return_value=scorer), \
                mock.patch.object(llmtry, "search", return_value=out), \
                mock.patch.object(llmgrind, "wait_gpu", return_value=True), \
                mock.patch("tools.dk.platforms.gc_mwcc.judge.unit_probe", return_value=None):
            cfg = mock.Mock(root=Path("/nonexistent"), rel=lambda p: str(p))
            return llmgrind.work(cfg, item, args, llmtry.Settings(), None, lambda r: None, 0.0, say=lambda *_: None)

    def test_a_judge_clean_unwritten_miss_becomes_a_near_candidate(self):
        good = SRC.replace("return a;", "return a + 1;")
        ok = {"compiled": True, "percent": 93.5, "devices": [], "worse": [], "text": good}
        res = self.run_work(93.5, "int f(int a) { return a + 1; }", ok)
        self.assertEqual((res["result"], res["candidate"]["kind"], res["candidate"]["after"]), ("near", "near", 93.5))
        self.assertEqual(res["candidate"]["text"], good)
        self.assertEqual(self.run_work(85.0, "x", ok)["result"], "miss")             # under --keep-near
        bad = {**ok, "text": SRC.replace("return a;", "return (a, 1);")}
        self.assertEqual(self.run_work(93.5, "x", bad)["result"], "miss")             # the judge rejects it


class NearCandidateTest(unittest.TestCase):
    setUp = CandidateTest.setUp
    def near(self, text, after=93.25):
        return {**candidate(text), "kind": "near", "class": "unwritten", "route": "llm", "before": 0.0,
                "after": after, "model": "m"}

    def test_digest_puts_near_after_exact(self):
        exact = llmgrind.write_candidate(self.cdir, candidate(SRC.replace("return a;", "return a + 1;")))
        near = llmgrind.write_candidate(self.cdir, self.near(SRC.replace("return a;", "return a + 2;")))
        with mock.patch("builtins.print"):
            llmgrind.run_review(self.cfg, argparse.Namespace(date="", reject=None, max_diff=40))
        text = (self.cdir / "digest.md").read_text()
        self.assertIn("1 exact: 1 clean", text.splitlines()[0])
        self.assertIn("; 1 near)", text.splitlines()[0])
        self.assertLess(text.index(f"## {exact} "), text.index("# Near candidates"))
        self.assertLess(text.index("# Near candidates"), text.index(f"## {near} "))
        self.assertIn("0.00% -> 93.25%, near", text)

    def land_near(self, ids, before=60.0, after=93.25, others=None, rc_unit=0):
        calls = []

        def fake_dk(cfg, *argv):
            calls.append(argv)
            return rc_unit if argv[0] == "check" else 0

        def fake_score(cfg, rec, text):
            got = after if text != SRC else before
            return {"f": got, "g": (others or {}).get(text != SRC, 100.0)}
        with mock.patch("builtins.print"):
            rc = llmgrind.run_land(self.cfg, argparse.Namespace(ids=ids, no_commit=False), run_dk=fake_dk,
                                   score=fake_score)
        return rc, calls

    def test_near_lands_only_through_save(self):
        new = SRC.replace("return a;", "return a + 2;")
        cid = llmgrind.write_candidate(self.cdir, self.near(new))
        rc, calls = self.land_near([cid])                   # even when the unit check would pass
        self.assertEqual(rc, 0)
        self.assertEqual((self.root / "src/game/a.cpp").read_text(), new)
        self.assertEqual([c[0] for c in calls], ["save"])
        self.assertIn("f 60.00% -> 93.25% (near: llm-grind candidate " + cid, calls[0][-1])
        labels = (self.ddir / "candidates" / "labels.jsonl").read_text()
        self.assertIn('"approved"', labels)

    def test_near_refused_without_gain_or_when_others_drop(self):
        new = SRC.replace("return a;", "return a + 2;")
        cid = llmgrind.write_candidate(self.cdir, self.near(new))
        rc, calls = self.land_near([cid], before=95.0, after=93.25)
        self.assertEqual((rc, calls), (1, []))
        self.assertEqual((self.root / "src/game/a.cpp").read_text(), SRC)
        rc, calls = self.land_near([cid], others={True: 90.0})
        self.assertEqual((rc, calls), (1, []))
        self.assertEqual((self.root / "src/game/a.cpp").read_text(), SRC)
        self.assertEqual(llmgrind.near_check({"function": "f"}, None, {"f": 1.0}), "does not compile here")


class GrindLoopTest(unittest.TestCase):
    def test_resume_skips_done_and_writes_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            root.mkdir()
            cfg = make_project(root)
            ddir = Path(d) / "llm-data"
            report = {"units": [{"name": "main/game/a", "metadata": {"source_path": "src/game/a.cpp"},
                                 "functions": [{"name": n, "size": 0x20, "fuzzy_match_percent": 0.0,
                                                "metadata": {"virtual_address": i}} for i, n in enumerate("fgh")]}]}
            seen = []

            def fake_work(cfg, item, args, st, client, log, until, say=print):
                seen.append(item["function"])
                if item["function"] == "g":
                    return {"unit": item["unit"], "function": "g", "result": "exact", "route": "llm",
                            "candidate": candidate(SRC.replace("return a;", "return 2;"), fn_name="f")}
                return {"unit": item["unit"], "function": item["function"], "result": "miss"}
            parser = argparse.ArgumentParser()
            llmgrind._configure_grind(parser)
            with mock.patch.dict(os.environ, {"DK_LLM_DATA": str(ddir)}), \
                    mock.patch("tools.dk.core.progress.load_report", return_value=report), \
                    mock.patch.object(llmgrind.llmtry, "never_train", return_value=(set(), set(), set())), \
                    mock.patch.object(llmgrind.llmtry, "retrieval_for", return_value=None), \
                    mock.patch.object(llmgrind, "busy_sources", return_value=set()), \
                    mock.patch.object(llmgrind.llmtry, "model_ready", return_value=True), \
                    mock.patch.object(llmgrind, "work", fake_work), mock.patch("builtins.print"):
                llmgrind.run_grind(cfg, parser.parse_args(["--max", "2", "--size-order"]))
                self.assertEqual(seen, ["f", "g"])
                llmgrind.run_grind(cfg, parser.parse_args(["--resume"]))
                self.assertEqual(seen, ["f", "g", "h"])
            state = json.loads((ddir / "grind-state.json").read_text())
            self.assertEqual(sorted(state["done"]), ["main/game/a::f", "main/game/a::g", "main/game/a::h"])
            cands = list((ddir / "candidates" / state["date"]).glob("*.json"))
            self.assertEqual(len(cands), 1)
            self.assertEqual(state["done"]["main/game/a::g"]["id"], cands[0].stem)


if __name__ == "__main__":
    unittest.main()
