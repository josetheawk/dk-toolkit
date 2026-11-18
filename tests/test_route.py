import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tests.helpers import make_project
from tools.dk.platforms.gc_mwcc import compilersweep, llmgrind, route


def ins(text, diff=None, reloc=False):
    row = {"instruction": {"formatted": text}}
    if reloc:
        row["instruction"]["relocation"] = {"type_name": "R_PPC_ADDR16_LO"}
    if diff:
        row["diff_kind"] = diff
    return row


def fn(name, size, pct, unit="main/game/a", source="src/game/a.cpp", address=0):
    return {"unit": unit, "source": source, "function": name, "address": address, "size": size, "fuzzy": pct}


class DiffKindTest(unittest.TestCase):
    def test_register_only(self):
        left = [ins("lwz r3, 0(r4)"), ins("mr r5, r3", "DIFF_ARG_MISMATCH"), ins("blr")]
        right = [ins("lwz r3, 0(r4)"), ins("mr r6, r3", "DIFF_ARG_MISMATCH"), ins("blr")]
        self.assertEqual(route.diff_kind(left, right), "register")

    def test_same_instructions_in_another_order_is_scheduling(self):
        left = [ins("lwz r3, 0(r4)", "DIFF_REPLACE"), ins("lwz r5, 4(r4)", "DIFF_REPLACE"), ins("blr")]
        right = [ins("lwz r5, 4(r4)", "DIFF_REPLACE"), ins("lwz r3, 0(r4)", "DIFF_REPLACE"), ins("blr")]
        self.assertEqual(route.diff_kind(left, right), "scheduling")

    def test_other_instructions_or_lengths_are_shape(self):
        self.assertEqual(route.diff_kind([ins("li r3, 1", "DIFF_REPLACE")], [ins("li r3, 2", "DIFF_REPLACE")]), "shape")
        self.assertEqual(route.diff_kind([ins("blr")], [ins("nop"), ins("blr")]), "shape")
        self.assertEqual(route.diff_kind([ins("stwu r1, -0x20(r1)", "DIFF_ARG_MISMATCH")],
                                         [ins("stwu r1, -0x10(r1)", "DIFF_ARG_MISMATCH")]), "shape")

    def test_a_relocated_operand_that_differs_is_data(self):
        same = [ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH", reloc=True)]
        self.assertEqual(route.diff_kind(same, [ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH", reloc=True)]), "data")
        left = [ins("lis r4, lbl_8067CFA0@ha", "DIFF_ARG_MISMATCH", reloc=True), ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH")]
        right = [ins("lis r3, ...data.0@ha", "DIFF_ARG_MISMATCH", reloc=True), ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH")]
        self.assertEqual(route.diff_kind(left, right), "data")             # symbol and register: the data comes first
        self.assertEqual(route.diff_kind([ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH", reloc=True)],
                                         [ins("addi r4, r31, 0x0", "DIFF_ARG_MISMATCH", reloc=True)]), "data")
        # the same immediate change on an instruction with no relocation is the function's own shape
        self.assertEqual(route.diff_kind([ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH")],
                                         [ins("addi r4, r31, 0x0", "DIFF_ARG_MISMATCH")]), "shape")


class ClassifyTest(unittest.TestCase):
    def test_first_match_wins(self):
        c = route.classify
        self.assertEqual(c("__sinit_a_cpp", 0x40, 99.0, "register", False), "sinit")
        self.assertEqual(c("@12", 0x40, 0.0, None, False), "sinit")
        self.assertEqual(c("big", 0x401, 99.9, "register", True), "giant")
        self.assertEqual(c("big", 0x400, 99.9, "register", False), "near-register")
        self.assertEqual(c("new", 0x40, 0.0, None, True), "fresh")
        self.assertEqual(c("f", 0x40, 90.0, "register", True), "compiler-version")
        self.assertEqual(c("f", 0x40, 90.0, "data", False), "unit-data")
        self.assertEqual(c("f", 0x40, 79.9, "shape", False), "far")
        self.assertEqual(c("f", 0x40, 80.0, "scheduling", False), "near-scheduling")
        self.assertEqual(c("f", 0x40, 95.0, "shape", False), "near-shape")
        self.assertEqual(c("f", 0x40, 95.0, None, False), "near-shape")


class RowsTest(unittest.TestCase):
    def test_workers_attempts_promotion_and_context(self):
        reg = ([ins("mr r5, r3", "DIFF_ARG_MISMATCH")], [ins("mr r6, r3", "DIFF_ARG_MISMATCH")])
        data = ([ins("addi r4, r31, 0x30", "DIFF_ARG_MISMATCH", reloc=True)],
                [ins("addi r4, r31, 0x0", "DIFF_ARG_MISMATCH", reloc=True)])
        diffs = {"main/game/a": ({"r": reg, "r2": reg, "d": data}, [".sdata2"])}
        fns = [fn("r", 0x40, 99.0, address=1), fn("r2", 0x40, 98.0, address=2), fn("d", 0x40, 99.0, address=3),
               fn("w", 0x40, 99.0, unit="main/game/w", source="src/game/w.cpp"), fn("g", 0x800, 99.0),
               fn("n", 0x40, 0.0)]
        attempts = {"main/game/a::r2": {"attempts": 2, "idle": 2, "best": 98.0, "candidate": None},
                    "main/game/a::r": {"attempts": 1, "idle": 1, "best": 99.0, "candidate": "abc"}}
        rows = {r["function"]: r for r in route.route_rows(fns, diffs, {"game/w.cpp": "GC/1.3.2 C++"}, attempts)}
        self.assertEqual((rows["r"]["class"], rows["r"]["worker"]), ("near-register", "cpu"))
        self.assertEqual(rows["r"]["candidate"], "abc")
        self.assertEqual((rows["r2"]["class"], rows["r2"]["worker"], rows["r2"]["idle"]), ("near-register", "claude", 2))
        self.assertEqual((rows["d"]["class"], rows["d"]["worker"]), ("unit-data", "blocked"))
        self.assertIn(".sdata2 open", rows["d"]["why"])
        self.assertEqual((rows["w"]["class"], rows["w"]["worker"]), ("compiler-version", "blocked"))
        self.assertIn("GC/1.3.2 C++", rows["w"]["why"])
        self.assertEqual((rows["g"]["class"], rows["g"]["worker"]), ("giant", "claude"))
        self.assertEqual((rows["n"]["class"], rows["n"]["worker"]), ("fresh", "lora"))
        self.assertEqual(rows["r"]["data_open"], [".sdata2"])
        s = route.summary(list(rows.values()))
        self.assertEqual(s["functions"], 6)
        self.assertEqual(s["classes"]["near-register"], {"functions": 2, "bytes": 0x80})
        self.assertEqual(s["workers"]["blocked"]["functions"], 2)

    def test_a_function_without_an_object_is_near_shape_and_says_so(self):
        rows = route.route_rows([fn("f", 0x40, 95.0)], {}, {}, {})
        self.assertEqual(rows[0]["class"], "near-shape")
        self.assertIn("no object to diff", rows[0]["why"])

    def test_select_by_class_or_worker(self):
        rows = route.route_rows([fn("f", 0x40, 95.0), fn("n", 0x40, 0.0), fn("g", 0x800, 1.0)], {}, {}, {})
        self.assertEqual([r["function"] for r in route.select(rows, "fresh")], ["n"])
        self.assertEqual([r["function"] for r in route.select(rows, "lora,claude")], ["f", "n", "g"])
        self.assertEqual(len(route.select(rows, "")), 3)
        with self.assertRaises(ValueError):
            route.select(rows, "nope")


class InputsTest(unittest.TestCase):
    def test_attempts_from_state_files(self):
        with tempfile.TemporaryDirectory() as d:
            ddir = Path(d)
            (ddir / "grind-state.json").write_text(json.dumps({"run": "a", "done": {
                "u::f": {"result": "miss", "before": 90.0, "best": 90.0, "id": None},
                "u::g": {"result": "miss", "before": 90.0, "best": 94.0, "id": None},
                "u::s": {"result": "skip", "before": 100.0},
                "u::e": {"result": "exact", "before": 90.0, "best": 100.0, "id": "c1"}}}))
            (ddir / "grind-state-pass2.json").write_text(json.dumps({"run": "b", "done": {
                "u::f": {"result": "miss", "before": 90.0, "best": None, "id": None}}}))
            (ddir / "grind-state-bad.json").write_text("{not json")
            a = route.load_attempts(ddir)
            self.assertEqual((a["u::f"]["attempts"], a["u::f"]["idle"], a["u::f"]["best"]), (2, 2, 90.0))
            self.assertEqual((a["u::g"]["attempts"], a["u::g"]["idle"], a["u::g"]["best"]), (1, 0, 94.0))
            self.assertEqual(a["u::e"]["candidate"], "c1")
            self.assertNotIn("u::s", a)
            self.assertEqual(route.load_attempts(ddir / "nothing"), {})

    def test_sweep_cache_names_the_gaining_variant_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sweep.json"
            p.write_text(json.dumps([
                {"unit": "game/a.cpp", "variants": [{"try": "GC/1.3 C++", "gains": False, "lost": []},
                                                    {"try": "GC/1.3.2 C++", "gains": True, "lost": []}]},
                {"unit": "game/b.cpp", "variants": [{"try": "GC/1.3 C++", "gains": True, "lost": ["x"]}]},
                {"unit": "game/c.cpp", "variants": []}]))
            self.assertEqual(route.load_sweep(p), {"game/a.cpp": "GC/1.3.2 C++"})
            self.assertEqual(route.load_sweep(Path(d) / "missing.json"), {})
            self.assertEqual(compilersweep.CACHE, "build/dk/sweep.json")


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "proj"
        root.mkdir()
        self.cfg = make_project(root)
        self.ddir = Path(self.tmp.name) / "llm-data"
        self.ddir.mkdir()
        meta = {"source_path": "src/game/a.cpp"}
        self.report = {"units": [
            {"name": "main/game/a", "metadata": meta, "functions": [
                {"name": "near", "size": 0x40, "fuzzy_match_percent": 99.0, "metadata": {"virtual_address": 1}},
                {"name": "fresh", "size": 0x20, "fuzzy_match_percent": 0.0, "metadata": {"virtual_address": 2}},
                {"name": "done", "size": 0x20, "fuzzy_match_percent": 100.0, "metadata": {"virtual_address": 3}}]},
            {"name": "main/game/auto", "metadata": {"auto_generated": True}, "functions": []}]}
        self.units = [{"name": "main/game/a", "base_path": "build/a.o", "target_path": "t.o", "metadata": meta}]
        reg = ([ins("mr r5, r3", "DIFF_ARG_MISMATCH")], [ins("mr r6, r3", "DIFF_ARG_MISMATCH")])
        self.patches = [
            mock.patch.dict(os.environ, {"DK_LLM_DATA": str(self.ddir)}),
            mock.patch("tools.dk.core.progress.load_report", return_value=self.report),
            mock.patch("tools.dk.platforms.gc_mwcc.matching.load_objdiff_units", return_value=self.units),
            mock.patch.object(route, "unit_diff", return_value=({"near": reg}, [])),
            mock.patch.object(route, "sweep_path", return_value=self.ddir / "none.json")]
        for p in self.patches:
            p.start()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def run_route(self, *argv):
        parser = argparse.ArgumentParser()
        route._configure(parser)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = route.run(self.cfg, parser.parse_args(argv))
        return rc, out.getvalue()

    def test_json_covers_every_open_function(self):
        rc, out = self.run_route("--json")
        doc = json.loads(out)
        self.assertEqual(rc, 0)
        self.assertEqual([(r["function"], r["class"], r["worker"]) for r in doc["rows"]],
                         [("near", "near-register", "cpu"), ("fresh", "fresh", "lora")])
        self.assertEqual(doc["functions"], 2)
        self.assertEqual(doc["workers"]["cpu"]["functions"], 1)

    def test_json_rows_have_blocker_and_version_and_fresh_modes(self):
        rc, out = self.run_route("--json")
        doc = json.loads(out)
        self.assertTrue(doc["playbook_version"])
        near = doc["rows"][0]
        self.assertEqual((near["kind"], near["blocker"], near["fresh"]), ("register", None, True))
        rc, out = self.run_route("--fresh", "--depletion", "--per-day", "1")
        self.assertIn("near 0x40 99.00% register", out.replace("near 0x40", "near 0x40"))
        self.assertIn("fresh: 1 function(s)", out)
        self.assertIn("1.0 day(s) left", out)
        self.assertIn("under one day", self.run_route("--depletion", "--per-day", "5")[1])
        self.assertEqual(self.run_route("--record", "near", "temp-temp", "--note", "x")[0], 0)
        self.assertEqual(self.run_route("--record", "near", "Bad Tag")[0], 2)
        doc = json.loads(self.run_route("--json")[1])
        self.assertEqual((doc["rows"][0]["blocker"], doc["rows"][0]["fresh"]), ("temp-temp", False))
        self.assertIn("fresh: 0 function(s)", self.run_route("--fresh")[1])

    def test_text_class_filter_and_errors(self):
        rc, out = self.run_route()
        self.assertEqual(rc, 0)
        self.assertIn("near-register", out)
        self.assertIn("2 open function(s)", out)
        rc, out = self.run_route("--class", "fresh", "--json")
        self.assertEqual([r["function"] for r in json.loads(out)["rows"]], ["fresh"])
        self.assertEqual(self.run_route("--class", "bogus")[0], 2)

    def test_grind_class_filter(self):
        seen = []

        def fake_work(cfg, item, args, st, client, log, until, say=print):
            seen.append(item["function"])
            return {"unit": item["unit"], "function": item["function"], "result": "miss"}
        parser = argparse.ArgumentParser()
        llmgrind._configure_grind(parser)
        report = {"units": [{**self.report["units"][0], "functions": [
            {**f, "metadata": {"virtual_address": i}} for i, f in enumerate(self.report["units"][0]["functions"])]}]}
        with mock.patch("tools.dk.core.progress.load_report", return_value=report), \
                mock.patch.object(llmgrind.llmtry, "never_train", return_value=(set(), set(), set())), \
                mock.patch.object(llmgrind.llmtry, "retrieval_for", return_value=None), \
                mock.patch.object(llmgrind, "busy_sources", return_value=set()), \
                mock.patch.object(llmgrind.llmtry, "model_ready", return_value=True), \
                mock.patch.object(llmgrind, "work", fake_work), mock.patch("builtins.print"):
            llmgrind.run_grind(self.cfg, parser.parse_args(["--size-order", "--class", "near-register"]))
            self.assertEqual(seen, ["near"])
            self.assertEqual(llmgrind.run_grind(self.cfg, parser.parse_args(["--class", "bogus"])), 2)


if __name__ == "__main__":
    unittest.main()
